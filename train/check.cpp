// yue2mtp-check: validation helpers for the trainer (README "Validation").
//   synth     DIR [--real-dims] [--songs N]     write a synthetic cache
//   gradcheck [--seed S]                        tiny config, CPU: graph vs naive reference + FD gradients
//   cpuvk     --target GGUF --cache DIR [--chunk N] [--unroll K] [--song I] [--offset R]
//                                               loss + per-tensor grad norms, CPU vs Vulkan
#include "y2t.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>

// ------------------------------------------------------- direct backward ---

struct DirectResult
{
	double loss = 0;
	std::vector<double> ce;                    // per depth (raw graph value)
	std::vector<std::vector<float>> grads;     // per trainable tensor
	std::vector<std::string>        node_desc; // Y2T_NODECMP: every F32 node of the graph
	std::vector<std::vector<float>> node_data;
};

// forward + backward of the loss without ggml_opt (sched over the given backends)
static DirectResult run_direct(const Backends & be, const Model & m, Inputs & in, bool want_grads)
{
	ggml_init_params ip = {ggml_tensor_overhead() * 16384 + 3 * ggml_graph_overhead_custom(16384, true), nullptr, true};
	ggml_context * ctx = ggml_init(ip);
	ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, true);
	GraphOut go = build_forward(ctx, m, in, true, false);
	for (ggml_tensor * t : go.ce)
	{
		ggml_set_output(t);
	}
	ggml_set_output(go.loss);
	ggml_set_loss(go.loss);
	ggml_build_forward_expand(gf, go.loss);
	for (ggml_tensor * t : go.ce)
	{
		ggml_build_forward_expand(gf, t);
	}
	if (getenv("Y2T_EVALGRAPH"))
	{
		EvalOut eo = build_eval(ctx, m, in, go);
		for (int k = 0; k < in.K; k++)
		{
			ggml_build_forward_expand(gf, eo.lse[k]);
			ggml_build_forward_expand(gf, eo.am[k]);
			ggml_build_forward_expand(gf, eo.lsup[k]);
		}
	}
	ggml_cgraph * g = gf;
	if (want_grads)
	{
		g = ggml_graph_dup(ctx, gf, true);
		ggml_build_backward_expand(ctx, g, nullptr);
	}
	if (be.is_vk)
	{
		// ops of the forward+backward graph the Vulkan backend cannot run (would fall back to CPU)
		std::map<std::string, int> fb;
		for (int i = 0; i < ggml_graph_n_nodes(g); i++)
		{
			ggml_tensor * t = ggml_graph_node(g, i);
			if (!ggml_backend_supports_op(be.main, t))
			{
				fb[std::string(ggml_op_desc(t)) + " " + ggml_type_name(t->type)]++;
			}
		}
		printf("graph nodes %d; ops not supported on %s:%s\n", ggml_graph_n_nodes(g), ggml_backend_name(be.main), fb.empty() ? " none" : "");
		for (auto & kv : fb)
		{
			printf("  %s x%d\n", kv.first.c_str(), kv.second);
		}
	}
	const bool nodecmp = getenv("Y2T_NODECMP") != nullptr;
	if (nodecmp)
	{
		for (int i = 0; i < ggml_graph_n_nodes(g); i++)
		{
			ggml_set_output(ggml_graph_node(g, i));
		}
	}
	std::vector<ggml_backend_t> bl = be.list();
	ggml_backend_sched_t sched = ggml_backend_sched_new(bl.data(), nullptr, (int) bl.size(), 16384, false, false);
	if (!ggml_backend_sched_alloc_graph(sched, g))
	{
		die("run_direct: graph alloc failed");
	}
	if (want_grads)
	{
		ggml_graph_reset(g);
	}
	ggml_backend_sched_graph_compute(sched, g);

	DirectResult r;
	float v;
	ggml_backend_tensor_get(go.loss, &v, 0, 4);
	r.loss = v;
	for (ggml_tensor * t : go.ce)
	{
		ggml_backend_tensor_get(t, &v, 0, 4);
		r.ce.push_back(v);
	}
	if (want_grads)
	{
		for (ggml_tensor * p : m.trainable())
		{
			ggml_tensor * gr = ggml_graph_get_grad(g, p);
			std::vector<float> h(ggml_nelements(p), 0.0f);
			if (gr)
			{
				ggml_backend_tensor_get(gr, h.data(), 0, h.size() * 4);
			}
			r.grads.push_back(std::move(h));
		}
	}
	if (nodecmp)
	{
		for (int i = 0; i < ggml_graph_n_nodes(g); i++)
		{
			ggml_tensor * t = ggml_graph_node(g, i);
			if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
			{
				continue;
			}
			char d[256];
			snprintf(d, sizeof(d), "%s %s [%lld,%lld,%lld,%lld] src0=%s", ggml_op_desc(t), t->name, (long long) t->ne[0],
			         (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->src[0] ? ggml_op_desc(t->src[0]) : "-");
			std::vector<float> h(ggml_nelements(t));
			ggml_backend_tensor_get(t, h.data(), 0, h.size() * 4);
			r.node_desc.push_back(d);
			r.node_data.push_back(std::move(h));
		}
	}
	ggml_backend_sched_free(sched);
	ggml_free(ctx);
	return r;
}

// ----------------------------------------------------- naive reference ---

static std::vector<double> get_d(ggml_tensor * t)
{
	std::vector<float> f(ggml_nelements(t));
	ggml_backend_tensor_get(t, f.data(), 0, f.size() * 4);
	return std::vector<double>(f.begin(), f.end());
}

// y[o] = Σ_i W[o][i] x[i], W ggml [in, out]
static std::vector<double> matvec(const std::vector<double> & W, int in, int out, const double * x)
{
	std::vector<double> y(out, 0.0);
	for (int o = 0; o < out; o++)
	{
		for (int i = 0; i < in; i++)
		{
			y[o] += W[(size_t) o * in + i] * x[i];
		}
	}
	return y;
}

static void rmsnorm(std::vector<double> & x, const std::vector<double> & w, double eps)
{
	double s = 0;
	for (double v : x)
	{
		s += v * v;
	}
	const double r = 1.0 / sqrt(s / x.size() + eps);
	for (size_t i = 0; i < x.size(); i++)
	{
		x[i] = x[i] * r * w[i];
	}
}

static void rope_norm(double * x, int hd, int pos, double theta)
{
	for (int i = 0; i < hd; i += 2)
	{
		const double a = pos * pow(theta, -(double) i / hd);
		const double c = cos(a), s = sin(a);
		const double x0 = x[i], x1 = x[i + 1];
		x[i]     = x0 * c - x1 * s;
		x[i + 1] = x0 * s + x1 * c;
	}
}

// independent double-precision forward of the draft model + unrolled objective; returns per-depth mean CE over real rows
static std::vector<double> ref_forward(const Model & m, const Batch & b)
{
	const Hparams & h = m.hp;
	const int n = b.n, K = b.K, E = h.n_embd, V = h.n_vocab, hd = h.head_dim, nh = h.n_head, nkv = h.n_head_kv;
	std::vector<double> fc = get_d(m.fc), an = get_d(m.attn_norm), an2 = get_d(m.attn_norm_2);
	std::vector<double> wq = get_d(m.wq), wk = get_d(m.wk), wv = get_d(m.wv), wo = get_d(m.wo);
	std::vector<double> fn = get_d(m.ffn_norm), fg = get_d(m.ffn_gate), fu = get_d(m.ffn_up), fdn = get_d(m.ffn_down);
	std::vector<double> on = get_d(m.output_norm), ow = get_d(m.output);
	std::vector<uint8_t> q8(ggml_nbytes(m.tok_embd));
	ggml_backend_tensor_get(m.tok_embd, q8.data(), 0, q8.size());
	std::vector<float> emb((size_t) E * V);
	const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
	for (int r = 0; r < V; r++)
	{
		tt->to_float(q8.data() + ggml_row_size(GGML_TYPE_Q8_0, E) * r, emb.data() + (size_t) r * E, E);
	}

	// g per depth/row, K/V per depth/row
	std::vector<std::vector<double>> g(n, std::vector<double>(E));
	for (int i = 0; i < n; i++)
	{
		g[i] = matvec(fc, h.n_feat, E, std::vector<double>(b.feat.begin() + (size_t) i * h.n_feat, b.feat.begin() + (size_t) (i + 1) * h.n_feat).data());
	}
	std::vector<std::vector<std::vector<double>>> Kd(K), Vd(K);
	std::vector<double> ce(K, 0.0);
	for (int k = 0; k < K; k++)
	{
		std::vector<std::vector<double>> Q(n);
		Kd[k].resize(n);
		Vd[k].resize(n);
		for (int i = 0; i < n; i++)
		{
			const int t = b.tok[(size_t) k * n + i];
			std::vector<double> e(emb.begin() + (size_t) t * E, emb.begin() + (size_t) (t + 1) * E);
			rmsnorm(e, an, h.eps);
			std::vector<double> gn = g[i];
			rmsnorm(gn, an2, h.eps);
			std::vector<double> x = e;
			x.insert(x.end(), gn.begin(), gn.end());
			Q[i]     = matvec(wq, 2 * E, nh * hd, x.data());
			Kd[k][i] = matvec(wk, 2 * E, nkv * hd, x.data());
			Vd[k][i] = matvec(wv, 2 * E, nkv * hd, x.data());
			const int pos = b.pos[(size_t) k * n + i];
			for (int hh = 0; hh < nh; hh++)
			{
				rope_norm(Q[i].data() + hh * hd, hd, pos, h.theta);
			}
			for (int hh = 0; hh < nkv; hh++)
			{
				rope_norm(Kd[k][i].data() + hh * hd, hd, pos, h.theta);
			}
		}
		for (int i = 0; i < n; i++)
		{
			// keys: depth-0 rows j <= i, depth-1..k rows of origin i
			std::vector<std::pair<int, int>> keys;
			for (int j = 0; j <= i; j++)
			{
				keys.push_back({0, j});
			}
			for (int d = 1; d <= k; d++)
			{
				keys.push_back({d, i});
			}
			std::vector<double> o(nh * hd, 0.0);
			for (int hh = 0; hh < nh; hh++)
			{
				const int kv = hh / (nh / nkv);
				std::vector<double> s(keys.size());
				double mx = -1e300;
				for (size_t a = 0; a < keys.size(); a++)
				{
					const double * kp = Kd[keys[a].first][keys[a].second].data() + kv * hd;
					double d = 0;
					for (int c = 0; c < hd; c++)
					{
						d += Q[i][hh * hd + c] * kp[c];
					}
					s[a] = d / sqrt((double) hd);
					mx = std::max(mx, s[a]);
				}
				double z = 0;
				for (auto & v : s)
				{
					v = exp(v - mx);
					z += v;
				}
				for (size_t a = 0; a < keys.size(); a++)
				{
					const double * vp = Vd[keys[a].first][keys[a].second].data() + kv * hd;
					for (int c = 0; c < hd; c++)
					{
						o[hh * hd + c] += s[a] / z * vp[c];
					}
				}
			}
			std::vector<double> hin = matvec(wo, nh * hd, E, o.data());
			for (int c = 0; c < E; c++)
			{
				hin[c] += g[i][c];
			}
			std::vector<double> hn = hin;
			rmsnorm(hn, fn, h.eps);
			std::vector<double> ga = matvec(fg, E, h.n_ff, hn.data()), up = matvec(fu, E, h.n_ff, hn.data());
			for (int c = 0; c < h.n_ff; c++)
			{
				ga[c] = ga[c] / (1.0 + exp(-ga[c])) * up[c];
			}
			std::vector<double> f = matvec(fdn, h.n_ff, E, ga.data());
			for (int c = 0; c < E; c++)
			{
				f[c] += hin[c];
			}
			g[i] = f;   // next depth's g (pre-norm)
			if (b.rmask[(size_t) k * n + i] == 0.0f)
			{
				continue;
			}
			std::vector<double> ho = f;
			rmsnorm(ho, on, h.eps);
			std::vector<double> lg = matvec(ow, E, V, ho.data());
			double mx = -1e300, z = 0;
			for (double v : lg)
			{
				mx = std::max(mx, v);
			}
			for (double v : lg)
			{
				z += exp(v - mx);
			}
			const double lse = mx + log(z);
			const size_t r = (size_t) k * n + i;
			for (int j = 0; j < b.L && b.sp_id[r * b.L + j] >= 0; j++)
			{
				ce[k] -= b.sp_p[r * b.L + j] * (lg[b.sp_id[r * b.L + j]] - lse);
			}
		}
		ce[k] /= std::max(1, b.n_real[k]);
	}
	return ce;
}

static double ref_loss(const Model & m, const Batch & b)
{
	double l = 0;
	for (double c : ref_forward(m, b))
	{
		l += c / b.K;
	}
	return l;
}

// ------------------------------------------------------------ gradcheck ---

static int cmd_gradcheck(uint64_t seed, const std::string & scratch, int n, int K)
{
	Hparams hp = Hparams::tiny();
	// a tiny cache with one short song (13 origins -> chunk padding) and one longer one
	const std::string dir = scratch + "/gradcheck_cache";
	write_synthetic_cache(dir, hp, 2, 1, 14, 40, seed);
	Cache cache;
	cache.open(dir, "train");

	Backends be;
	be.init("cpu", 8);
	Model m;
	m.alloc(hp, be.main, false);
	model_init_random(m, seed, 0.3f);
	for (ggml_tensor * t : m.trainable())
	{
		ggml_set_param(t);
	}
	Inputs in;
	in.alloc(hp, n, K, false, be.main);

	int fails = 0;
	for (int ci = 0; ci < (int) cache.songs.size(); ci++)
	{
		ChunkRef cr = {ci, 0, std::min(n, cache.songs[ci].n_pos - 1)};
		Batch b;
		build_batch(cache, cr, hp, n, K, false, b);
		batch_upload(b, in);
		printf("chunk %s: n_real per depth", b.tag.c_str());
		for (int k = 0; k < K; k++)
		{
			printf(" %d", b.n_real[k]);
		}
		printf("\n");

		DirectResult r = run_direct(be, m, in, true);
		std::vector<double> ref = ref_forward(m, b);
		double loss_ref = 0;
		for (int k = 0; k < K; k++)
		{
			const double ce_graph = r.ce[k] * n / std::max(1, b.n_real[k]);
			loss_ref += ref[k] / K;
			const double rel = fabs(ce_graph - ref[k]) / std::max(1e-9, fabs(ref[k]));
			printf("  depth %d: graph CE %.7f  ref CE %.7f  rel %.2e\n", k, ce_graph, ref[k], rel);
			fails += rel > 1e-4;
		}
		printf("  loss graph %.7f  ref %.7f\n", r.loss, loss_ref);
		fails += fabs(r.loss - loss_ref) / loss_ref > 1e-4;

		// central differences on a sample of every trainable tensor
		std::mt19937_64 rng(seed + ci);
		std::vector<ggml_tensor *> ps = m.trainable();
		for (size_t pi = 0; pi < ps.size(); pi++)
		{
			ggml_tensor * p = ps[pi];
			const int64_t ne = ggml_nelements(p);
			std::vector<int64_t> sel;
			if (ne <= 48)
			{
				for (int64_t e = 0; e < ne; e++)
				{
					sel.push_back(e);
				}
			} else {
				for (int s = 0; s < 48; s++)
				{
					sel.push_back((int64_t) (rng() % (uint64_t) ne));
				}
			}
			double num = 0, den = 0, maxabs = 0;
			for (int64_t e : sel)
			{
				float x0;
				ggml_backend_tensor_get(p, &x0, e * 4, 4);
				// FD on the double-precision reference loss (no F32 round-off in the difference)
				const float hstep = 1e-3f * std::max(1.0f, fabsf(x0));
				float xp = x0 + hstep, xm = x0 - hstep;
				ggml_backend_tensor_set(p, &xp, e * 4, 4);
				const double lp = ref_loss(m, b);
				ggml_backend_tensor_set(p, &xm, e * 4, 4);
				const double lm = ref_loss(m, b);
				ggml_backend_tensor_set(p, &x0, e * 4, 4);
				const double fd = (lp - lm) / ((double) xp - (double) xm);
				const double an = r.grads[pi][e];
				num += (fd - an) * (fd - an);
				den += an * an;
				maxabs = std::max(maxabs, fabs(fd - an));
			}
			const double rel = sqrt(num / std::max(den, 1e-30));
			printf("  %-26s n=%3zu  |g|=%.3e  rel.err %.2e  max|fd-an| %.2e %s\n", p->name, sel.size(), sqrt(den), rel, maxabs,
			       rel < 1e-3 ? "ok" : "FAIL");
			fails += rel >= 1e-3;
		}
	}
	printf("gradcheck: %s\n", fails ? "FAIL" : "PASS");
	in.free_all();
	m.free_all();
	be.free_all();
	cache.close();
	return fails ? 1 : 0;
}

// ---------------------------------------------------------------- cpuvk ---

static int cmd_cpuvk(const std::string & target, const std::string & cache_dir, int n, int K, int song, int offset,
                     const std::string & init, int threads)
{
	Hparams hp;
	TargetGguf tg;
	tg.open(target);
	Cache cache;
	cache.open(cache_dir, "train", song + 1);
	hp.layers = cache.layers;
	const CacheSong & s = cache.songs[song];
	ChunkRef cr = {song, offset, std::min(n, s.n_pos - 1 - offset)};
	Batch b;
	build_batch(cache, cr, hp, n, K, false, b);
	printf("chunk %s n=%d K=%d n_real", b.tag.c_str(), n, K);
	for (int k = 0; k < K; k++)
	{
		printf(" %d", b.n_real[k]);
	}
	printf("\n");

	DirectResult res[2];
	for (int pass = 0; pass < 2; pass++)
	{
		Backends be;
		be.init(pass == 0 ? "cpu" : "vulkan", threads);
		Model m;
		m.alloc(hp, be.main, false);
		model_init_from_target(m, tg, init, 1234);
		for (ggml_tensor * t : m.trainable())
		{
			ggml_set_param(t);
		}
		Inputs in;
		in.alloc(hp, n, K, false, be.main);
		batch_upload(b, in);
		auto wsum = [&]()
		{
			double acc = 0;
			for (ggml_tensor * t : m.trainable())
			{
				std::vector<float> h(ggml_nelements(t));
				ggml_backend_tensor_get(t, h.data(), 0, h.size() * 4);
				for (size_t e = 0; e < h.size(); e++)
				{
					acc += h[e] * (double) ((e % 7) + 1);
				}
			}
			return acc;
		};
		const double w0 = wsum();
		const double t0 = now_s();
		// optional loss scaling on the Vulkan pass (Y2T_LOSS_SCALE): grads scaled back after
		const char * lsenv = getenv("Y2T_LOSS_SCALE");
		const float ls = pass == 1 && lsenv ? (float) atof(lsenv) : 1.0f;
		if (ls != 1.0f)
		{
			Batch bs = b;
			for (auto & w : bs.w)
			{
				w *= ls;
			}
			batch_upload(bs, in);
		}
		res[pass] = run_direct(be, m, in, getenv("Y2T_EVALGRAPH") == nullptr);
		if (ls != 1.0f)
		{
			res[pass].loss /= ls;
			for (auto & gv : res[pass].grads)
			{
				for (auto & x : gv)
				{
					x /= ls;
				}
			}
		}
		printf("weights checksum before %.9e after %.9e\n", w0, wsum());
		printf("%s: loss %.7f (%.1f s)\n", pass == 0 ? "CPU" : "Vulkan", res[pass].loss, now_s() - t0);
		in.free_all();
		m.free_all();
		be.free_all();
	}
	printf("loss rel.err %.3e\n", fabs(res[0].loss - res[1].loss) / fabs(res[0].loss));
	for (int k = 0; k < K; k++)
	{
		printf("  depth %d CE: cpu %.7f vk %.7f rel %.3e\n", k, res[0].ce[k], res[1].ce[k],
		       fabs(res[0].ce[k] - res[1].ce[k]) / fabs(res[0].ce[k]));
	}
	static const char * pn[] = {"fc", "output_norm", "output", "attn_norm", "attn_norm_2", "attn_q", "attn_k", "attn_v",
	                            "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"};
	double worst = 0;
	for (size_t i = 0; i < res[0].grads.size(); i++)
	{
		double a = 0, bb = 0, d = 0;
		for (size_t e = 0; e < res[0].grads[i].size(); e++)
		{
			const double x = res[0].grads[i][e], y = res[1].grads[i][e];
			a += x * x;
			bb += y * y;
			d += (x - y) * (x - y);
		}
		const double rel_norm = fabs(sqrt(a) - sqrt(bb)) / std::max(1e-30, sqrt(a));
		const double rel_vec = sqrt(d) / std::max(1e-30, sqrt(a));
		worst = std::max(worst, rel_vec);
		printf("  grad %-12s |g| cpu %.6e vk %.6e  rel(|g|) %.2e  rel(g) %.2e\n", pn[i], sqrt(a), sqrt(bb), rel_norm, rel_vec);
	}
	printf("worst elementwise-vector rel.err %.3e\n", worst);
	int shown = 0;
	std::map<std::string, std::vector<size_t>> vk_at;
	for (size_t i = 0; i < res[1].node_desc.size(); i++)
	{
		vk_at[res[1].node_desc[i]].push_back(i);
	}
	std::map<std::string, size_t> seen;
	for (size_t i = 0; i < res[0].node_data.size() && shown < 40; i++)
	{
		const std::string & key = res[0].node_desc[i];
		const size_t occ = seen[key]++;
		if (!vk_at.count(key) || vk_at[key].size() <= occ)
		{
			continue;
		}
		const auto & x = res[0].node_data[i];
		const auto & y = res[1].node_data[vk_at[key][occ]];
		double d = 0, a = 0;
		for (size_t e = 0; e < x.size(); e++)
		{
			d += (double) (x[e] - y[e]) * (x[e] - y[e]);
			a += (double) x[e] * x[e];
		}
		const double rel = sqrt(d / std::max(a, 1e-30));
		if (rel > 1e-3 || !std::isfinite(rel))
		{
			printf("  node diverges: rel %.2e  #%zu %s\n", rel, i, key.c_str());
			shown++;
		}
	}
	cache.close();
	tg.close();
	return 0;
}

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: yue2mtp-check synth DIR [--real-dims] [--songs N] | gradcheck [--seed S] [--scratch DIR] |\n"
		                "       cpuvk --target GGUF --cache DIR [--chunk N] [--unroll K] [--song I] [--offset R] [--init struct|random]\n");
		return 2;
	}
	const std::string cmd = argv[1];
	std::map<std::string, std::string> a;
	std::vector<std::string> pos;
	for (int i = 2; i < argc; i++)
	{
		std::string s = argv[i];
		if (s.rfind("--", 0) == 0)
		{
			if (s == "--real-dims")
			{
				a[s] = "1";
			} else if (i + 1 < argc) {
				a[s] = argv[++i];
			}
		} else {
			pos.push_back(s);
		}
	}
	auto geti = [&](const char * k, int d) { return a.count(k) ? atoi(a[k].c_str()) : d; };
	auto gets = [&](const char * k, const char * d) { return a.count(k) ? a[k] : std::string(d); };
	setenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "1", 0);   // see train.cpp
	ggml_backend_load_all();
	if (cmd == "synth")
	{
		if (pos.empty())
		{
			die("synth DIR");
		}
		Hparams hp = a.count("--real-dims") ? Hparams() : Hparams::tiny();
		const int ns = geti("--songs", 6);
		write_synthetic_cache(pos[0], hp, ns, std::max(1, ns / 5), geti("--min-len", 1500), geti("--max-len", 5000), geti("--seed", 1));
		printf("wrote %s\n", pos[0].c_str());
		return 0;
	}
	if (cmd == "gradcheck")
	{
		const std::string scratch = gets("--scratch", "/tmp");
		mkdir(scratch.c_str(), 0755);
		return cmd_gradcheck(geti("--seed", 7), scratch, geti("--chunk", 16), geti("--unroll", 3));
	}
	if (cmd == "cpuvk")
	{
		if (!a.count("--target"))
		{
			die("cpuvk needs --target AR.gguf");
		}
		return cmd_cpuvk(a["--target"], gets("--cache", "data/cache"), geti("--chunk", 2048),
		                 geti("--unroll", 4), geti("--song", 0), geti("--offset", 0), gets("--init", "struct"), geti("--threads", 8));
	}
	die("unknown command %s", cmd.c_str());
}
