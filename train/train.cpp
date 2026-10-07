// yue2mtp-train: EAGLE-3 draft-head trainer on ggml + ggml_opt (see README).
#include "y2t.h"

#include "ggml-cpu.h"
#include "ggml-opt.h"
#ifdef Y2T_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <future>
#include <map>
#include <string>
#include <sys/stat.h>

namespace
{

struct Opts
{
	std::string target;
	std::string cache   = "data/cache";
	std::string out;
	std::string backend = "vulkan";
	std::string resume;
	std::string labels  = "shaped";
	std::string init    = "struct";
	int         epochs  = 1;
	float       lr      = 1e-4f;
	int         warmup  = 200;
	int         unroll  = 4;
	int         chunk   = 2048;
	int         accum   = 8;
	bool        freeze_head = false;
	int         eval_every  = 250;
	int         eval_chunks = 48;
	int         save_every  = 1000;
	int         log_every   = 10;
	int         threads     = 8;
	int         max_steps   = 0;    // opt steps, 0 = epochs decide
	int         songs       = 0;    // limit train songs (overfit tests)
	int         min_rows    = 256;
	uint64_t    seed        = 42;
	bool        fixed_offset = false;
	bool        eval_only    = false;
	std::string precision    = "f32";  // f32: Vulkan matmuls without fp16/coopmat; f16: coopmat + loss scaling
	float       loss_scale   = 0.0f;   // 0 = auto (1 for f32, 1024 for f16)
};

FILE * g_log = nullptr;

void logf(const char * fmt, ...)
{
	char buf[4096];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	fputs(buf, stdout);
	fflush(stdout);
	if (g_log)
	{
		fputs(buf, g_log);
		fflush(g_log);
	}
}

struct LrState
{
	int64_t step   = 0;   // completed optimiser steps
	int64_t total  = 1;
	int     warmup = 200;
	float   peak   = 1e-4f;
	float   mult   = 1.0f;
	float   last   = 0.0f;
	float   scale  = 1.0f;   // loss scale: AdamW is invariant to it except eps
};

double lr_at(const LrState & s)
{
	if (s.step < s.warmup)
	{
		return s.peak * (double) (s.step + 1) / s.warmup;
	}
	const double prog = std::min(1.0, (double) (s.step - s.warmup) / std::max<int64_t>(1, s.total - s.warmup));
	return s.peak * (0.1 + 0.9 * 0.5 * (1.0 + cos(M_PI * prog)));
}

ggml_opt_optimizer_params lr_cb(void * ud)
{
	LrState * s = (LrState *) ud;
	ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
	s->last = (float) (lr_at(*s) * s->mult);
	p.adamw.alpha = s->last;
	p.adamw.beta1 = 0.9f;
	p.adamw.beta2 = 0.95f;
	p.adamw.eps   = 1e-8f * s->scale;
	p.adamw.wd    = 0.0f;
	return p;
}

void usage()
{
	fprintf(stderr,
		"usage: yue2mtp-train --target AR.gguf --out DIR [--cache DIR] [--backend vulkan|cpu] [--resume CKPT.gguf]\n"
		"         [--epochs N] [--lr X] [--warmup N] [--unroll K] [--chunk N] [--accum B] [--labels shaped|raw]\n"
		"         [--init struct|random] [--freeze-head] [--eval-every N] [--eval-chunks N] [--save-every N]\n"
		"         [--log-every N] [--threads N] [--max-steps N] [--songs N] [--min-rows N] [--seed S] [--fixed-offset]\n"
		"         [--precision f32|f16] [--loss-scale S] [--eval-only]\n");
	exit(2);
}

Opts parse(int argc, char ** argv)
{
	Opts o;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		auto next = [&]() -> std::string
		{
			if (i + 1 >= argc)
			{
				usage();
			}
			return argv[++i];
		};
		if (a == "--target") o.target = next();
		else if (a == "--cache") o.cache = next();
		else if (a == "--out") o.out = next();
		else if (a == "--backend") o.backend = next();
		else if (a == "--resume") o.resume = next();
		else if (a == "--labels") o.labels = next();
		else if (a == "--init") o.init = next();
		else if (a == "--epochs") o.epochs = std::stoi(next());
		else if (a == "--lr") o.lr = std::stof(next());
		else if (a == "--warmup") o.warmup = std::stoi(next());
		else if (a == "--unroll") o.unroll = std::stoi(next());
		else if (a == "--chunk") o.chunk = std::stoi(next());
		else if (a == "--accum") o.accum = std::stoi(next());
		else if (a == "--freeze-head") o.freeze_head = true;
		else if (a == "--eval-every") o.eval_every = std::stoi(next());
		else if (a == "--eval-chunks") o.eval_chunks = std::stoi(next());
		else if (a == "--save-every") o.save_every = std::stoi(next());
		else if (a == "--log-every") o.log_every = std::stoi(next());
		else if (a == "--threads") o.threads = std::stoi(next());
		else if (a == "--max-steps") o.max_steps = std::stoi(next());
		else if (a == "--songs") o.songs = std::stoi(next());
		else if (a == "--min-rows") o.min_rows = std::stoi(next());
		else if (a == "--seed") o.seed = std::stoull(next());
		else if (a == "--fixed-offset") o.fixed_offset = true;
		else if (a == "--eval-only") o.eval_only = true;
		else if (a == "--precision") o.precision = next();
		else if (a == "--loss-scale") o.loss_scale = std::stof(next());
		else usage();
	}
	if (o.out.empty() || o.target.empty() || (o.labels != "shaped" && o.labels != "raw") || o.unroll < 1 || o.accum < 1 ||
	    (o.precision != "f32" && o.precision != "f16"))
	{
		usage();
	}
	return o;
}

void mkdirs(const std::string & p)
{
	for (size_t i = 1; i <= p.size(); i++)
	{
		if (i == p.size() || p[i] == '/')
		{
			mkdir(p.substr(0, i).c_str(), 0755);
		}
	}
}

std::string vram_str(const Backends & be)
{
	if (!be.is_vk)
	{
		return "";
	}
#ifdef Y2T_VULKAN
	size_t fr = 0, tot = 0;
	ggml_backend_vk_get_device_memory(0, &fr, &tot);
	char b[64];
	snprintf(b, sizeof(b), " vram_used %.2f GB", (tot - fr) / 1e9);
	return b;
#else
	return "";
#endif
}

struct Evaluator
{
	ggml_context *       ctx = nullptr;
	ggml_cgraph *        gf = nullptr;
	ggml_backend_sched_t sched = nullptr;
	EvalOut              eo;
	std::vector<ChunkRef> chunks;

	void init(const Backends & be, const Model & m, const Inputs & in, const Cache & held, int max_chunks, int min_rows)
	{
		ggml_init_params ip = {ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true};
		ctx = ggml_init(ip);
		gf = ggml_new_graph_custom(ctx, 8192, false);
		GraphOut go = build_forward(ctx, m, in, false, false);
		eo = build_eval(ctx, m, in, go);
		for (int k = 0; k < in.K; k++)
		{
			ggml_build_forward_expand(gf, eo.lse[k]);
			ggml_build_forward_expand(gf, eo.am[k]);
			ggml_build_forward_expand(gf, eo.lsup[k]);
		}
		std::vector<ggml_backend_t> bl = be.list();
		sched = ggml_backend_sched_new(bl.data(), nullptr, (int) bl.size(), 8192, false, false);
		std::mt19937_64 rng(0);
		std::vector<ChunkRef> all = make_chunks(held, in.n, min_rows, false, rng);
		if (max_chunks > 0 && (int) all.size() > max_chunks)
		{
			// evenly spaced subset (covers every song in index order)
			for (int i = 0; i < max_chunks; i++)
			{
				chunks.push_back(all[(size_t) i * all.size() / max_chunks]);
			}
		} else {
			chunks = all;
		}
	}

	std::string run(const Cache & held, const Hparams & hp, Inputs & in, double & secs, double & a0)
	{
		const double t0 = now_s();
		EvalAcc acc;
		acc.init(in.K);
		Batch b;
		const int n = in.n, K = in.K, L = in.L;
		std::vector<std::vector<int32_t>> am(K, std::vector<int32_t>(n));
		std::vector<std::vector<float>> lse(K, std::vector<float>(n)), lsup(K, std::vector<float>((size_t) n * L));
		for (const ChunkRef & cr : chunks)
		{
			build_batch(held, cr, hp, n, K, false, b);
			batch_upload(b, in);
			ggml_backend_sched_reset(sched);
			if (!ggml_backend_sched_alloc_graph(sched, gf))
			{
				die("eval graph alloc failed");
			}
			ggml_backend_sched_graph_compute(sched, gf);
			for (int k = 0; k < K; k++)
			{
				ggml_backend_tensor_get(eo.am[k], am[k].data(), 0, (size_t) n * 4);
				ggml_backend_tensor_get(eo.lse[k], lse[k].data(), 0, (size_t) n * 4);
				ggml_backend_tensor_get(eo.lsup[k], lsup[k].data(), 0, (size_t) n * L * 4);
			}
			acc.add(b, hp.n_vocab, am, lse, lsup);
		}
		secs = now_s() - t0;
		a0 = acc.accA[0] / std::max(1.0, acc.n[0]);
		return acc.report();
	}

	void free_all()
	{
		if (sched)
		{
			ggml_backend_sched_free(sched);
		}
		if (ctx)
		{
			ggml_free(ctx);
		}
	}
};

} // namespace

int main(int argc, char ** argv)
{
	Opts o = parse(argc, argv);
	mkdirs(o.out);
	g_log = fopen((o.out + "/train.log").c_str(), "a");
	{
		std::string cmd;
		for (int i = 0; i < argc; i++)
		{
			cmd += std::string(i ? " " : "") + argv[i];
		}
		logf("# %s\n", cmd.c_str());
	}
	if (o.loss_scale <= 0.0f)
	{
		o.loss_scale = o.precision == "f16" ? 1024.0f : 1.0f;
	}
	if (o.precision == "f32")
	{
		// ggml-vulkan stages F32 matmuls through fp16 (coopmat): the backward's tiny logit
		// gradients (~1e-7) underflow there (README "Vulkan workarounds"), so default to
		// exact F32 matmuls
		setenv("GGML_VK_DISABLE_F16", "1", 0);
		setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
	}
	// ggml-vulkan's graph_optimize node reordering corrupts the eval graph's outputs
	// (NaN lse; exact again with it off) — keep it off everywhere (README "Vulkan workarounds")
	setenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "1", 0);
	logf("precision %s, loss scale %g\n", o.precision.c_str(), o.loss_scale);
	ggml_backend_load_all();

	Backends be;
	be.init(o.backend, o.threads);
	TargetGguf tg;
	tg.open(o.target);

	Cache train;
	train.open(o.cache, "train", o.songs);
	if (train.songs.empty())
	{
		die("no train songs in %s", o.cache.c_str());
	}
	Cache held;
	held.open(o.cache, "heldout");
	const bool have_held = !held.songs.empty();

	Hparams hp;
	hp.layers = train.layers;
	const bool raw = o.labels == "raw";
	logf("train: %zu songs, %lld positions; layers %d,%d,%d; backend %s\n", train.songs.size(), (long long) train.total_pos(),
	     hp.layers[0], hp.layers[1], hp.layers[2], o.backend.c_str());

	Model m;
	m.alloc(hp, be.main, raw);
	model_init_from_target(m, tg, o.init, o.seed);
	if (!o.resume.empty())
	{
		model_load_checkpoint(m, o.resume);
		logf("resumed weights from %s (fresh AdamW moments, schedule restarts with warmup %d)\n", o.resume.c_str(), o.warmup);
	}
	int64_t n_params = 0;
	for (ggml_tensor * t : m.trainable())
	{
		if (o.freeze_head && t == m.output)
		{
			continue;
		}
		ggml_set_param(t);
		n_params += ggml_nelements(t);
	}
	logf("trainable parameters: %.1f M%s\n", n_params / 1e6, o.freeze_head ? " (head frozen)" : "");

	Inputs in;
	in.alloc(hp, o.chunk, o.unroll, raw, be.main);

	// ---- ggml_opt with static graphs
	ggml_init_params cip = {ggml_tensor_overhead() * 16384 + 3 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true};
	ggml_context * ctx_compute = ggml_init(cip);
	GraphOut go = build_forward(ctx_compute, m, in, true, true);
	std::vector<ggml_backend_t> bl = be.list();
	ggml_backend_sched_t sched = ggml_backend_sched_new(bl.data(), nullptr, (int) bl.size(), 16384, false, false);

	LrState lrs;
	lrs.peak   = o.lr;
	lrs.warmup = o.warmup;
	lrs.scale  = o.loss_scale;

	ggml_opt_params op = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
	op.ctx_compute     = ctx_compute;
	op.inputs          = in.feat;
	op.outputs         = go.loss;
	op.build_type      = GGML_OPT_BUILD_TYPE_OPT;
	op.opt_period      = o.accum;
	op.get_opt_pars    = lr_cb;
	op.get_opt_pars_ud = &lrs;
	op.optimizer       = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
	ggml_opt_context_t opt = ggml_opt_init(op);

	std::vector<ggml_tensor *> pt;
	for (ggml_tensor * t : m.trainable())
	{
		if (!(o.freeze_head && t == m.output))
		{
			pt.push_back(t);
		}
	}

	Evaluator ev;
	if (have_held)
	{
		ev.init(be, m, in, held, o.eval_chunks, o.min_rows);
		logf("held-out: %zu songs, %zu eval chunks\n", held.songs.size(), ev.chunks.size());
	}

	std::mt19937_64 rng(o.seed);
	{
		std::mt19937_64 r2(o.seed);
		const size_t nc = make_chunks(train, o.chunk, o.min_rows, !o.fixed_offset, r2).size();
		lrs.total = o.max_steps > 0 ? o.max_steps : (int64_t) o.epochs * (int64_t) (nc / o.accum);
		logf("~%zu chunks/epoch, %d micro-steps per opt step, %lld opt steps planned\n", nc, o.accum, (long long) lrs.total);
	}

	auto save = [&](const std::string & path)
	{
		model_write_gguf(m, tg, path, "yue2-mtp eagle3 draft");
		logf("saved %s\n", path.c_str());
	};
	auto do_eval = [&]()
	{
		if (!have_held)
		{
			return;
		}
		double secs = 0, a0 = 0;
		const std::string rep = ev.run(held, hp, in, secs, a0);
		logf("eval @ step %lld (%.1f s):\n%s", (long long) lrs.step, secs, rep.c_str());
	};

	if (o.eval_only)
	{
		do_eval();
		ev.free_all();
		return 0;
	}

	std::vector<double> gn_hist;
	double t_start = now_s(), t_win = now_s();
	int64_t pos_win = 0, micro = 0;
	std::vector<double> ce_sum(o.unroll, 0.0), ce_n(o.unroll, 0.0);
	double loss_sum = 0;
	int loss_n = 0;
	bool done = false;
	bool first = true;

	for (int ep = 0; ep < o.epochs * 1000 && !done; ep++)
	{
		if (o.max_steps == 0 && ep >= o.epochs)
		{
			break;
		}
		std::vector<ChunkRef> chunks = make_chunks(train, o.chunk, o.min_rows, !o.fixed_offset, rng);
		std::shuffle(chunks.begin(), chunks.end(), rng);
		auto make = [&](size_t i)
		{
			Batch b;
			build_batch(train, chunks[i], hp, o.chunk, o.unroll, raw, b);
			for (auto & w : b.w)
			{
				w *= o.loss_scale / (float) o.accum;
			}
			return b;
		};
		std::future<Batch> fut = std::async(std::launch::async, make, (size_t) 0);
		for (size_t ci = 0; ci < chunks.size() && !done; ci++)
		{
			Batch b = fut.get();
			if (ci + 1 < chunks.size())
			{
				fut = std::async(std::launch::async, make, ci + 1);
			}
			batch_upload(b, in);
			const double t0 = now_s();
			ggml_opt_alloc(opt, true);
			ggml_opt_eval(opt, nullptr);
			if (first)
			{
				first = false;
				logf("first step %.2f s; sched buffer %s %.2f GB%s\n", now_s() - t0, ggml_backend_name(be.main),
				     ggml_backend_sched_get_buffer_size(sched, be.main) / 1e9, vram_str(be).c_str());
			}
			micro++;
			pos_win += b.n_real[0];

			std::vector<float> ce(o.unroll);
			ggml_backend_tensor_get(in.ce_out, ce.data(), 0, ce.size() * 4);
			double l = 0;
			for (int k = 0; k < o.unroll; k++)
			{
				if (b.n_real[k] > 0)
				{
					const double c = (double) ce[k] * o.chunk / b.n_real[k];
					ce_sum[k] += c * b.n_real[k];
					ce_n[k] += b.n_real[k];
					l += c / o.unroll;
				}
			}
			loss_sum += l;
			loss_n++;
			if (!std::isfinite(l))
			{
				die("non-finite loss at %s", b.tag.c_str());
			}

			if (micro % o.accum != 0)
			{
				continue;
			}
			// an optimiser step just ran
			lrs.step++;
			double gnorm = NAN;
			if (o.accum > 1)
			{
				std::vector<ggml_tensor *> accs;
				for (ggml_tensor * t : pt)
				{
					accs.push_back(ggml_opt_grad_acc(opt, t));
				}
				double s2 = 0;
				for (double v : sum_sq(be.main, accs))
				{
					s2 += v;
				}
				gnorm = sqrt(s2) / o.loss_scale;
				if (gn_hist.size() >= 20)
				{
					std::vector<double> tmp(gn_hist.end() - std::min<size_t>(gn_hist.size(), 100), gn_hist.end());
					std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
					const double med = tmp[tmp.size() / 2];
					if (gnorm > 10.0 * med)
					{
						lrs.mult *= 0.5f;
						logf("grad-norm spike %.3e (median %.3e) at step %lld: lr multiplier -> %.3g\n", gnorm, med,
						     (long long) lrs.step, lrs.mult);
					}
				}
				gn_hist.push_back(gnorm);
			}
			if (lrs.step % o.log_every == 0 || lrs.step == 1)
			{
				const double dt = now_s() - t_win;
				std::string dep;
				for (int k = 0; k < o.unroll; k++)
				{
					char t[32];
					snprintf(t, sizeof(t), " %.4f", ce_sum[k] / std::max(1.0, ce_n[k]));
					dep += t;
				}
				logf("step %6lld ep %d lr %.3e loss %.4f ce/depth%s gnorm %.3e pos/s %.0f elapsed %.0f s%s\n",
				     (long long) lrs.step, ep, lrs.last, loss_sum / std::max(1, loss_n), dep.c_str(), gnorm, pos_win / dt,
				     now_s() - t_start, vram_str(be).c_str());
				std::fill(ce_sum.begin(), ce_sum.end(), 0.0);
				std::fill(ce_n.begin(), ce_n.end(), 0.0);
				loss_sum = 0;
				loss_n = 0;
				pos_win = 0;
				t_win = now_s();
			}
			if (o.eval_every > 0 && lrs.step % o.eval_every == 0)
			{
				do_eval();
				t_win = now_s();
				pos_win = 0;
			}
			if (o.save_every > 0 && lrs.step % o.save_every == 0)
			{
				char p[64];
				snprintf(p, sizeof(p), "/draft-step%06lld.gguf", (long long) lrs.step);
				save(o.out + p);
			}
			if (o.max_steps > 0 && lrs.step >= o.max_steps)
			{
				done = true;
			}
		}
		if (fut.valid())
		{
			fut.wait();
		}
		logf("epoch %d done (%lld opt steps, %.0f s)\n", ep, (long long) lrs.step, now_s() - t_start);
	}
	if (o.eval_every > 0 && lrs.step % o.eval_every != 0)
	{
		do_eval();
	}
	save(o.out + "/draft.gguf");

	ev.free_all();
	ggml_opt_free(opt);
	ggml_backend_sched_free(sched);
	ggml_free(ctx_compute);
	in.free_all();
	m.free_all();
	train.close();
	held.close();
	tg.close();
	be.free_all();
	if (g_log)
	{
		fclose(g_log);
	}
	return 0;
}
