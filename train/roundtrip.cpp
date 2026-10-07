// yue2mtp-roundtrip: GGUF round-trip check — load a trainer checkpoint as an eagle3 draft in
// libllama (against the target via ctx_other), feed a window of cache rows the way
// common/speculative.cpp draft-eagle3 process() does (encoder on the target features,
// then (t_{P+1}, g_P) pairs at pos P), and compare the draft logits at the d2t ids with
// the trainer's own depth-0 forward on the same rows.
//
//   yue2mtp-roundtrip --draft CKPT.gguf --target GGUF [--cache DIR] [--song I] [--offset R]
//                     [--n N<=512] [--llama-backend cpu|vulkan] [--threads T]
#include "y2t.h"

#include "common.h"
#include "llama.h"
#include "llama-ext.h"

#undef die   // common.h has its own; use the trainer's

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

int main(int argc, char ** argv)
{
	std::map<std::string, std::string> a;
	for (int i = 1; i + 1 < argc; i += 2)
	{
		a[argv[i]] = argv[i + 1];
	}
	auto gets = [&](const char * k, const char * d) { return a.count(k) ? a[k] : std::string(d); };
	auto geti = [&](const char * k, int d) { return a.count(k) ? atoi(a[k].c_str()) : d; };
	const std::string draft  = gets("--draft", "");
	const std::string target = gets("--target", "");
	const std::string cdir   = gets("--cache", "data/cache");
	const std::string lbe    = gets("--llama-backend", "cpu");
	const int song = geti("--song", 0), offset = geti("--offset", 0), n = geti("--n", 256), threads = geti("--threads", 8);
	if (draft.empty() || target.empty() || n > 512)
	{
		fprintf(stderr, "usage: yue2mtp-roundtrip --draft CKPT.gguf --target GGUF [--cache DIR] [--song I] [--offset R] [--n N<=512] [--llama-backend cpu|vulkan]\n");
		return 2;
	}
	ggml_backend_load_all();

	// ---- trainer forward (CPU, F32)
	Cache cache;
	cache.open(cdir, "train", song + 1);
	Hparams hp;
	hp.layers = cache.layers;
	const CacheSong & s = cache.songs[song];
	if (offset + n + 1 > s.n_pos)
	{
		die("window %d+%d past the song (%d rows)", offset, n, s.n_pos);
	}
	ChunkRef cr = {song, offset, n};
	Batch b;
	build_batch(cache, cr, hp, n, 1, false, b);

	std::vector<float> lg_tr((size_t) n * hp.n_vocab);
	{
		Backends be;
		be.init("cpu", threads);
		TargetGguf tg;
		tg.open(target);
		Model m;
		m.alloc(hp, be.main, false);
		model_init_from_target(m, tg, "struct", 1);
		model_load_checkpoint(m, draft);
		Inputs in;
		in.alloc(hp, n, 1, false, be.main);
		batch_upload(b, in);
		ggml_init_params ip = {ggml_tensor_overhead() * 4096 + ggml_graph_overhead(), nullptr, true};
		ggml_context * ctx = ggml_init(ip);
		ggml_cgraph * gf = ggml_new_graph(ctx);
		GraphOut go = build_forward(ctx, m, in, false, false);
		ggml_set_output(go.logits[0]);
		ggml_build_forward_expand(gf, go.logits[0]);
		ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.main));
		ggml_gallocr_alloc_graph(ga, gf);
		ggml_backend_graph_compute(be.main, gf);
		ggml_backend_tensor_get(go.logits[0], lg_tr.data(), 0, lg_tr.size() * 4);
		ggml_gallocr_free(ga);
		ggml_free(ctx);
		in.free_all();
		m.free_all();
		tg.close();
		be.free_all();
	}

	// ---- libllama: target (for ctx_other: tok_embd) + eagle3 draft
	llama_backend_init();
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = lbe == "vulkan" ? 999 : 0;
	llama_model * mt = llama_model_load_from_file(target.c_str(), mp);
	if (!mt)
	{
		die("cannot load target");
	}
	llama_context_params cpt = llama_context_default_params();
	cpt.n_ctx = 512;
	cpt.n_batch = cpt.n_ubatch = 512;
	cpt.n_threads = cpt.n_threads_batch = threads;
	if (lbe != "vulkan")
	{
		cpt.op_offload = false;
		cpt.offload_kqv = false;
	}
	llama_context * ct = llama_init_from_model(mt, cpt);

	llama_model * md = llama_model_load_from_file(draft.c_str(), mp);
	if (!md)
	{
		die("cannot load draft %s", draft.c_str());
	}
	llama_context_params cpd = cpt;
	cpd.n_ctx = 16384;
	if (gets("--kv", "f32") == "f32")
	{
		// F32 KV, no flash-attn: the comparison the spec's 1e-3 bound is about
		// (--kv default = llama's F16 cache + auto FA, i.e. what draft-eagle3 runs)
		cpd.type_k = GGML_TYPE_F32;
		cpd.type_v = GGML_TYPE_F32;
		cpd.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
	}
	cpd.ctx_other = ct;
	llama_context * cd = llama_init_from_model(md, cpd);
	if (!cd)
	{
		die("cannot create draft context");
	}
	llama_set_embeddings_nextn(cd, true, true);
	const int E = llama_model_n_embd(md);
	printf("draft loaded: n_embd %d, target layers %u, n_vocab %d\n", E, llama_model_target_layer_ids_n(md),
	       llama_vocab_n_tokens(llama_model_get_vocab(md)));

	// encoder: g = fc · feat
	common_batch be_(cd);
	be_.clear();
	std::vector<llama_pos> pos(n);
	for (int i = 0; i < n; i++)
	{
		pos[i] = i;
		be_.add_embd({b.feat.data() + (size_t) i * hp.n_feat, 1, (size_t) hp.n_feat}, &pos[i], 0, true);
	}
	if (llama_process(cd, LLAMA_PROCESS_TYPE_ENCODE, be_.get()) != 0)
	{
		die("encode failed");
	}
	std::vector<float> g((size_t) n * E);
	memcpy(g.data(), llama_get_embeddings_nextn(cd), g.size() * 4);

	// decoder: row P carries (t_{P+1} = tok[P], g_P) at the target position of P
	common_batch bd(cd);
	bd.clear();
	for (int i = 0; i < n; i++)
	{
		const int idx = bd.add(b.tok[i] + D2T_BASE, b.pos[i], 0, true);
		bd.set_embd(idx, {g.data() + (size_t) i * E, 1, (size_t) E});
	}
	if (llama_process(cd, LLAMA_PROCESS_TYPE_DECODE, bd.get()) != 0)
	{
		die("decode failed");
	}
	double maxd = 0, sumd = 0, maxl = 0;
	int agree = 0;
	for (int i = 0; i < n; i++)
	{
		const float * lg = llama_get_logits_ith(cd, i);
		int am_l = 0, am_t = 0;
		for (int j = 0; j < hp.n_vocab; j++)
		{
			const double x = lg[D2T_BASE + j], y = lg_tr[(size_t) i * hp.n_vocab + j];
			const double d = fabs(x - y);
			maxd = std::max(maxd, d);
			sumd += d;
			maxl = std::max(maxl, fabs(y));
			if (lg[D2T_BASE + j] > lg[D2T_BASE + am_l])
			{
				am_l = j;
			}
			if (lg_tr[(size_t) i * hp.n_vocab + j] > lg_tr[(size_t) i * hp.n_vocab + am_t])
			{
				am_t = j;
			}
		}
		agree += am_l == am_t;
		if (i < 3)
		{
			printf("row %d: llama %.5f %.5f %.5f | trainer %.5f %.5f %.5f\n", i, lg[D2T_BASE], lg[D2T_BASE + 1], lg[D2T_BASE + 2],
			       lg_tr[(size_t) i * hp.n_vocab], lg_tr[(size_t) i * hp.n_vocab + 1], lg_tr[(size_t) i * hp.n_vocab + 2]);
		}
	}
	// ids outside d2t must be -inf
	const float * l0 = llama_get_logits_ith(cd, 0);
	const bool outside_inf = std::isinf(l0[0]) && l0[0] < 0 && std::isinf(l0[D2T_BASE - 1]) && std::isinf(l0[D2T_BASE + hp.n_vocab]);
	printf("round-trip (%s llama, %d rows x %d draft ids): max|diff| %.3e  mean|diff| %.3e  max|logit| %.2f  argmax agree %d/%d  outside-d2t -inf: %s\n",
	       lbe.c_str(), n, hp.n_vocab, maxd, sumd / ((double) n * hp.n_vocab), maxl, agree, n, outside_inf ? "yes" : "NO");
	printf("%s\n", maxd <= 1e-3 ? "PASS (<= 1e-3)" : "FAIL (> 1e-3)");

	llama_free(cd);
	llama_model_free(md);
	llama_free(ct);
	llama_model_free(mt);
	llama_backend_free();
	cache.close();
	return maxd <= 1e-3 ? 0 : 1;
}
