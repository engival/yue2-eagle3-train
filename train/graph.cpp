// Training graph (README "Model" and "Objective"), inputs, backends, host metrics.
#include "y2t.h"

#include "ggml-cpu.h"
#ifdef Y2T_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

// ------------------------------------------------------------- backends ---

void Backends::init(const std::string & which, int threads)
{
	cpu = ggml_backend_cpu_init();
	ggml_backend_cpu_set_n_threads(cpu, threads);
	if (which == "vulkan")
	{
#ifdef Y2T_VULKAN
		if (ggml_backend_vk_get_device_count() < 1)
		{
			die("no Vulkan device");
		}
		main = ggml_backend_vk_init(0);
		if (main == nullptr)
		{
			die("cannot init Vulkan device 0");
		}
		is_vk = true;
#else
		die("built without Vulkan (GGML_VULKAN=OFF); use --backend cpu");
#endif
	} else if (which == "cpu") {
		main = cpu;
	} else {
		die("unknown backend %s", which.c_str());
	}
}

std::vector<ggml_backend_t> Backends::list() const
{
	if (is_vk)
	{
		return {main, cpu};
	}
	return {cpu};
}

void Backends::free_all()
{
	if (is_vk && main)
	{
		ggml_backend_free(main);
	}
	if (cpu)
	{
		ggml_backend_free(cpu);
	}
	main = cpu = nullptr;
}

std::vector<double> sum_sq(ggml_backend_t backend, const std::vector<ggml_tensor *> & ts)
{
	ggml_init_params ip = {ggml_tensor_overhead() * (ts.size() * 3 + 8) + ggml_graph_overhead(), nullptr, true};
	ggml_context * ctx = ggml_init(ip);
	ggml_cgraph * gf = ggml_new_graph(ctx);
	std::vector<ggml_tensor *> outs;
	for (ggml_tensor * t : ts)
	{
		ggml_tensor * s = ggml_sum(ctx, ggml_sqr(ctx, t));
		ggml_set_output(s);
		ggml_build_forward_expand(gf, s);
		outs.push_back(s);
	}
	ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
	if (!ggml_gallocr_alloc_graph(ga, gf))
	{
		die("sum_sq: alloc failed");
	}
	ggml_backend_graph_compute(backend, gf);
	std::vector<double> r;
	for (ggml_tensor * s : outs)
	{
		float v = 0.0f;
		ggml_backend_tensor_get(s, &v, 0, 4);
		r.push_back(v);
	}
	ggml_gallocr_free(ga);
	ggml_free(ctx);
	return r;
}

// --------------------------------------------------------------- inputs ---

void Inputs::alloc(const Hparams & hp, int n_, int K_, bool raw, ggml_backend_t backend)
{
	n = n_;
	K = K_;
	raw_labels = raw;
	ggml_init_params ip = {ggml_tensor_overhead() * 64, nullptr, true};
	ctx = ggml_init(ip);
	auto mk = [&](ggml_type type, int64_t a, int64_t b, int64_t c, const char * name) -> ggml_tensor *
	{
		ggml_tensor * t = ggml_new_tensor_3d(ctx, type, a, b, c);
		ggml_set_name(t, name);
		ggml_set_input(t);
		return t;
	};
	feat    = mk(GGML_TYPE_F32, hp.n_feat, n, 1, "in_feat");
	tok     = mk(GGML_TYPE_I32, n, K, 1, "in_tok");
	pos     = mk(GGML_TYPE_I32, n, K, 1, "in_pos");
	rmask   = mk(GGML_TYPE_F32, 1, n, K, "in_rmask");
	w       = mk(GGML_TYPE_F32, K, 1, 1, "in_w");
	lab_idx = mk(GGML_TYPE_I32, (int64_t) n * L, K, 1, "in_lab_idx");
	lab_val = mk(GGML_TYPE_F32, 1, (int64_t) n * L, K, "in_lab_val");
	top_idx = mk(GGML_TYPE_I32, n, K, 1, "in_top_idx");
	if (raw)
	{
		hid = mk(GGML_TYPE_F32, hp.n_embd, n, K, "in_hid");
	}
	ce_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
	ggml_set_name(ce_out, "ce_out");
	for (int k = 0; k < K; k++)
	{
		char name[32];
		snprintf(name, sizeof(name), "kq_mask_%d", k);
		kq_mask.push_back(mk(GGML_TYPE_F32, (int64_t) (k + 1) * n, n, 1, name));
	}
	buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
	if (buf == nullptr)
	{
		die("cannot allocate inputs");
	}
	ggml_backend_buffer_clear(buf, 0);
	set_masks();
}

void Inputs::set_masks()
{
	// block 0: depth-0 rows, causal (j <= i); block b >= 1: depth-b row of the same origin
	for (int k = 0; k < K; k++)
	{
		const int T = (k + 1) * n;
		std::vector<float> m((size_t) T * n, -INFINITY);
		for (int i = 0; i < n; i++)
		{
			float * row = m.data() + (size_t) i * T;
			for (int j = 0; j <= i; j++)
			{
				row[j] = 0.0f;
			}
			for (int b = 1; b <= k; b++)
			{
				row[(size_t) b * n + i] = 0.0f;
			}
		}
		ggml_backend_tensor_set(kq_mask[k], m.data(), 0, m.size() * 4);
	}
}

void Inputs::free_all()
{
	if (buf)
	{
		ggml_backend_buffer_free(buf);
	}
	if (ctx)
	{
		ggml_free(ctx);
	}
	buf = nullptr;
	ctx = nullptr;
	kq_mask.clear();
}

void batch_upload(const Batch & b, Inputs & in)
{
	if (b.n != in.n || b.K != in.K)
	{
		die("batch shape %dx%d does not match inputs %dx%d", b.n, b.K, in.n, in.K);
	}
	ggml_backend_tensor_set(in.feat, b.feat.data(), 0, b.feat.size() * 4);
	ggml_backend_tensor_set(in.tok, b.tok.data(), 0, b.tok.size() * 4);
	ggml_backend_tensor_set(in.pos, b.pos.data(), 0, b.pos.size() * 4);
	ggml_backend_tensor_set(in.rmask, b.rmask.data(), 0, b.rmask.size() * 4);
	ggml_backend_tensor_set(in.w, b.w.data(), 0, b.w.size() * 4);
	ggml_backend_tensor_set(in.lab_idx, b.lab_idx.data(), 0, b.lab_idx.size() * 4);
	ggml_backend_tensor_set(in.lab_val, b.lab_val.data(), 0, b.lab_val.size() * 4);
	ggml_backend_tensor_set(in.top_idx, b.top_idx.data(), 0, b.top_idx.size() * 4);
	if (in.hid)
	{
		ggml_backend_tensor_set(in.hid, b.hid.data(), 0, b.hid.size() * 4);
	}
}

// ---------------------------------------------------------------- graph ---

GraphOut build_forward(ggml_context * ctx, const Model & m, const Inputs & in, bool with_loss, bool with_ce_out)
{
	const Hparams & h = m.hp;
	const int n = in.n, K = in.K, V = h.n_vocab, L = in.L;
	const int hd = h.head_dim, nh = h.n_head, nkv = h.n_head_kv;
	const float kq_scale = 1.0f / sqrtf((float) hd);

	GraphOut go;

	// step-0 g = fc · concat(feat_low, feat_mid, feat_high)   (norm_before_fc = false)
	ggml_tensor * g = ggml_mul_mat(ctx, m.fc, in.feat);
	ggml_set_name(g, "g0");

	ggml_tensor * Kc = nullptr;
	ggml_tensor * Vc = nullptr;
	for (int k = 0; k < K; k++)
	{
		ggml_tensor * tok_k   = ggml_view_1d(ctx, in.tok, n, (size_t) k * in.tok->nb[1]);
		ggml_tensor * pos_k   = ggml_view_1d(ctx, in.pos, n, (size_t) k * in.pos->nb[1]);
		ggml_tensor * rmask_k = ggml_view_2d(ctx, in.rmask, 1, n, in.rmask->nb[1], (size_t) k * in.rmask->nb[2]);

		// e_n = rms_norm(e) * attn_norm, g_n = rms_norm(g) * attn_norm_2, x = concat(e_n, g_n)
		ggml_tensor * e  = ggml_get_rows(ctx, m.tok_embd, tok_k);
		ggml_tensor * en = ggml_mul(ctx, ggml_rms_norm(ctx, e, h.eps), m.attn_norm);
		ggml_tensor * gn = ggml_mul(ctx, ggml_rms_norm(ctx, g, h.eps), m.attn_norm_2);
		ggml_tensor * x  = ggml_concat(ctx, en, gn, 0);

		ggml_tensor * q  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.wq, x), hd, nh, n);
		ggml_tensor * kk = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.wk, x), hd, nkv, n);
		ggml_tensor * v  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.wv, x), hd, nkv, n);

		// RoPE NORM mode, theta 1e6, no YaRN
		q  = ggml_rope_ext(ctx, q, pos_k, nullptr, hd, GGML_ROPE_TYPE_NORMAL, h.n_ctx_orig, h.theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
		kk = ggml_rope_ext(ctx, kk, pos_k, nullptr, hd, GGML_ROPE_TYPE_NORMAL, h.n_ctx_orig, h.theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

		// keys/values: depth-0 rows, then the depth-1..k rows (mask selects the diagonal)
		Kc = k == 0 ? kk : ggml_concat(ctx, Kc, kk, 2);
		Vc = k == 0 ? v : ggml_concat(ctx, Vc, v, 2);

		ggml_tensor * Qp  = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));     // [hd, n, nh]
		ggml_tensor * Kp  = ggml_cont(ctx, ggml_permute(ctx, Kc, 0, 2, 1, 3));    // [hd, T, nkv]
		ggml_tensor * kq  = ggml_mul_mat(ctx, Kp, Qp);                            // [T, n, nh]
		kq = ggml_soft_max_ext(ctx, kq, in.kq_mask[k], kq_scale, 0.0f);
		ggml_tensor * Vt  = ggml_cont(ctx, ggml_permute(ctx, Vc, 1, 2, 0, 3));    // [T, hd, nkv]
		ggml_tensor * kqv = ggml_mul_mat(ctx, Vt, kq);                            // [hd, n, nh]
		ggml_tensor * o   = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));   // [hd, nh, n]
		o = ggml_reshape_2d(ctx, o, hd * nh, n);

		// residual = raw g (norm_before_residual = false)
		ggml_tensor * h_in = ggml_add(ctx, ggml_mul_mat(ctx, m.wo, o), g);
		ggml_tensor * hn   = ggml_mul(ctx, ggml_rms_norm(ctx, h_in, h.eps), m.ffn_norm);
		ggml_tensor * f    = ggml_mul_mat(ctx, m.ffn_down,
			ggml_swiglu_split(ctx, ggml_mul_mat(ctx, m.ffn_gate, hn), ggml_mul_mat(ctx, m.ffn_up, hn)));
		ggml_tensor * hh   = ggml_add(ctx, f, h_in);    // eagle3_prenorm -> next depth's g
		g = hh;

		// head; pad rows are zeroed before the head so they carry no gradient
		ggml_tensor * ho = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, hh, h.eps), m.output_norm), rmask_k);
		ggml_tensor * logits = ggml_mul_mat(ctx, m.output, ho);                  // [V, n]
		char name[32];
		snprintf(name, sizeof(name), "logits_%d", k);
		ggml_set_name(logits, name);
		go.logits.push_back(logits);

		if (!with_loss)
		{
			continue;
		}
		ggml_tensor * labels;
		if (in.raw_labels)
		{
			ggml_tensor * hid_k = ggml_view_2d(ctx, in.hid, h.n_embd, n, in.hid->nb[1], (size_t) k * in.hid->nb[2]);
			labels = ggml_soft_max(ctx, ggml_mul_mat(ctx, m.out_tgt, hid_k));
			labels = ggml_mul(ctx, labels, rmask_k);
		} else {
			ggml_tensor * idx_k = ggml_view_1d(ctx, in.lab_idx, (int64_t) n * L, (size_t) k * in.lab_idx->nb[1]);
			ggml_tensor * val_k = ggml_view_2d(ctx, in.lab_val, 1, (int64_t) n * L, in.lab_val->nb[1], (size_t) k * in.lab_val->nb[2]);
			ggml_tensor * dense = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, (int64_t) V * n), 0.0f);
			labels = ggml_reshape_2d(ctx, ggml_set_rows(ctx, dense, val_k, idx_k), V, n);
		}
		ggml_tensor * ce = ggml_cross_entropy_loss(ctx, logits, labels);
		if (with_ce_out)
		{
			ce = ggml_cpy(ctx, ce, ggml_view_1d(ctx, in.ce_out, 1, (size_t) k * 4));
		}
		go.ce.push_back(ce);
		ggml_tensor * lk = ggml_mul(ctx, ce, ggml_view_1d(ctx, in.w, 1, (size_t) k * 4));
		go.loss = go.loss ? ggml_add(ctx, go.loss, lk) : lk;
	}
	if (go.loss)
	{
		ggml_set_name(go.loss, "loss");
	}
	return go;
}

EvalOut build_eval(ggml_context * ctx, const Model & m, const Inputs & in, const GraphOut & go)
{
	const int n = in.n, V = m.hp.n_vocab, L = in.L;
	EvalOut eo;
	for (int k = 0; k < in.K; k++)
	{
		ggml_tensor * lg   = go.logits[k];
		ggml_tensor * flat = ggml_reshape_2d(ctx, lg, 1, (int64_t) V * n);
		ggml_tensor * tidx = ggml_view_1d(ctx, in.top_idx, n, (size_t) k * in.top_idx->nb[1]);
		ggml_tensor * sidx = ggml_view_1d(ctx, in.lab_idx, (int64_t) n * L, (size_t) k * in.lab_idx->nb[1]);
		// lse with the logit at argmax p as the shift (never far below the max)
		ggml_tensor * top  = ggml_reshape_2d(ctx, ggml_get_rows(ctx, flat, tidx), 1, n);
		ggml_tensor * lse  = ggml_add(ctx, ggml_log(ctx, ggml_sum_rows(ctx, ggml_exp(ctx, ggml_sub(ctx, lg, top)))), top);
		ggml_tensor * am   = ggml_argmax(ctx, lg);
		ggml_tensor * lsup = ggml_get_rows(ctx, flat, sidx);
		for (ggml_tensor * t : {lse, am, lsup})
		{
			ggml_set_output(t);
		}
		eo.am.push_back(am);
		eo.lse.push_back(lse);
		eo.lsup.push_back(lsup);
	}
	return eo;
}

// ------------------------------------------------------------- metrics ---

void EvalAcc::init(int K_)
{
	K = K_;
	for (auto * v : {&n, &kl, &top1, &accA, &accB, &accBr, &ce, &ent})
	{
		v->assign(K, 0.0);
	}
}

void EvalAcc::add(const Batch & b, int V, const std::vector<std::vector<int32_t>> & am,
                  const std::vector<std::vector<float>> & lse, const std::vector<std::vector<float>> & lsup)
{
	(void) V;
	const int L = b.L;
	for (int k = 0; k < K; k++)
	{
		for (int i = 0; i < b.n; i++)
		{
			if (b.rmask[(size_t) k * b.n + i] == 0.0f)
			{
				continue;
			}
			const int32_t * id = b.sp_id.data() + ((size_t) k * b.n + i) * L;
			const float   * p  = b.sp_p.data() + ((size_t) k * b.n + i) * L;
			const float   * ls = lsup[k].data() + (size_t) i * L;
			const double  z    = lse[k][i];
			if (!std::isfinite(z))
			{
				static int warned = 0;
				if (warned++ < 5)
				{
					fprintf(stderr, "eval: non-finite lse %g at %s depth %d row %d\n", z, b.tag.c_str(), k, i);
				}
			}
			double c = 0, e = 0, sB = 0, qs = 0, pa = 0;
			for (int j = 0; j < L && id[j] >= 0; j++)
			{
				const double lq = ls[j] - z;
				c  -= p[j] * lq;
				e  -= p[j] > 0 ? p[j] * log((double) p[j]) : 0.0;
				sB += std::min((double) p[j], exp(lq));
				qs += exp(lq);
				if (id[j] == am[k][i])
				{
					pa = p[j];
				}
			}
			double sBr = 0;
			for (int j = 0; j < L && id[j] >= 0 && qs > 0; j++)
			{
				sBr += std::min((double) p[j], exp(ls[j] - z) / qs);
			}
			n[k]     += 1;
			ce[k]    += c;
			ent[k]   += e;
			kl[k]    += c - e;
			top1[k]  += am[k][i] == id[0] ? 1 : 0;
			accA[k]  += pa;
			accB[k]  += sB;
			accBr[k] += sBr;
		}
	}
}

std::string EvalAcc::report() const
{
	std::string s;
	char line[256];
	double prod = 1.0, len = 0.0;
	for (int k = 0; k < K; k++)
	{
		const double c = std::max(1.0, n[k]);
		snprintf(line, sizeof(line), "  d%d: n %.0f  ce %.4f  H(p) %.4f  KL %.4f  top1 %.4f  A %.4f  B %.4f  B|supp %.4f\n",
		         k, n[k], ce[k] / c, ent[k] / c, kl[k] / c, top1[k] / c, accA[k] / c, accB[k] / c, accBr[k] / c);
		s += line;
	}
	s += "  E[accepted] (scheme A, product rule):";
	for (int k = 0; k < K; k++)
	{
		prod *= accA[k] / std::max(1.0, n[k]);
		len += prod;
		snprintf(line, sizeof(line), " K=%d %.3f", k + 1, len);
		s += line;
	}
	s += "\n  E[accepted] (scheme B, product rule):";
	prod = 1.0;
	len = 0.0;
	for (int k = 0; k < K; k++)
	{
		prod *= accB[k] / std::max(1.0, n[k]);
		len += prod;
		snprintf(line, sizeof(line), " K=%d %.3f", k + 1, len);
		s += line;
	}
	s += "\n";
	return s;
}
