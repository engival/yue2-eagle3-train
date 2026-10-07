// Model tensors, init from the target, GGUF checkpoint I/O.
#include "y2t.h"

#include "gguf.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <chrono>

void die(const char * fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "error: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(1);
}

double now_s()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

Hparams Hparams::tiny()
{
	Hparams h;
	h.n_embd    = 64;
	h.n_head    = 2;
	h.n_head_kv = 1;
	h.head_dim  = 32;
	h.n_ff      = 128;
	h.n_vocab   = 128;
	h.n_feat    = 3 * 64;
	return h;
}

// ----------------------------------------------------------------- model ---

std::vector<ggml_tensor *> Model::trainable() const
{
	return {fc, output_norm, output, attn_norm, attn_norm_2, wq, wk, wv, wo, ffn_norm, ffn_gate, ffn_up, ffn_down};
}

void Model::alloc(const Hparams & h, ggml_backend_t backend, bool with_out_tgt)
{
	hp = h;
	ggml_init_params ip = {ggml_tensor_overhead() * 32, nullptr, true};
	ctx = ggml_init(ip);
	const int E = h.n_embd;
	auto f32 = [&](const char * name, int64_t a, int64_t b) -> ggml_tensor *
	{
		ggml_tensor * t = b > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a);
		ggml_set_name(t, name);
		return t;
	};
	tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, h.n_vocab);
	ggml_set_name(tok_embd, "tok_embd_slice");
	if (with_out_tgt)
	{
		out_tgt = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, E, h.n_vocab);
		ggml_set_name(out_tgt, "output_tgt_slice");
	}
	fc          = f32("fc.weight", h.n_feat, E);
	output_norm = f32("output_norm.weight", E, 0);
	output      = f32("output.weight", E, h.n_vocab);
	attn_norm   = f32("blk.0.attn_norm.weight", E, 0);
	attn_norm_2 = f32("blk.0.attn_norm_2.weight", E, 0);
	wq          = f32("blk.0.attn_q.weight", 2 * E, h.n_head * h.head_dim);
	wk          = f32("blk.0.attn_k.weight", 2 * E, h.n_head_kv * h.head_dim);
	wv          = f32("blk.0.attn_v.weight", 2 * E, h.n_head_kv * h.head_dim);
	wo          = f32("blk.0.attn_output.weight", h.n_head * h.head_dim, E);
	ffn_norm    = f32("blk.0.ffn_norm.weight", E, 0);
	ffn_gate    = f32("blk.0.ffn_gate.weight", E, h.n_ff);
	ffn_up      = f32("blk.0.ffn_up.weight", E, h.n_ff);
	ffn_down    = f32("blk.0.ffn_down.weight", h.n_ff, E);
	buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
	if (buf == nullptr)
	{
		die("cannot allocate model weights");
	}
	ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}

void Model::free_all()
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
}

void model_copy_weights(const Model & src, Model & dst)
{
	std::vector<ggml_tensor *> a = src.trainable(), b = dst.trainable();
	a.push_back(src.tok_embd);
	b.push_back(dst.tok_embd);
	if (src.out_tgt && dst.out_tgt)
	{
		a.push_back(src.out_tgt);
		b.push_back(dst.out_tgt);
	}
	std::vector<uint8_t> tmp;
	for (size_t i = 0; i < a.size(); i++)
	{
		tmp.resize(ggml_nbytes(a[i]));
		ggml_backend_tensor_get(a[i], tmp.data(), 0, tmp.size());
		ggml_backend_tensor_set(b[i], tmp.data(), 0, tmp.size());
	}
}

// ---------------------------------------------------------------- target ---

void TargetGguf::open(const std::string & p)
{
	path = p;
	gguf_init_params ip = {true, &meta};
	gg = gguf_init_from_file(p.c_str(), ip);
	if (gg == nullptr)
	{
		die("cannot read %s", p.c_str());
	}
	data_off = gguf_get_data_offset(gg);
}

void TargetGguf::close()
{
	if (gg)
	{
		gguf_free(gg);
	}
	if (meta)
	{
		ggml_free(meta);
	}
	gg = nullptr;
	meta = nullptr;
}

ggml_tensor * TargetGguf::tensor(const char * name) const
{
	ggml_tensor * t = ggml_get_tensor(meta, name);
	if (t == nullptr)
	{
		die("target has no tensor %s", name);
	}
	return t;
}

std::vector<uint8_t> TargetGguf::read_raw(const char * name, size_t row_first, size_t n_rows) const
{
	ggml_tensor * t = tensor(name);
	const int64_t id = gguf_find_tensor(gg, name);
	const size_t rs = ggml_row_size(t->type, t->ne[0]);
	if (row_first + n_rows > (size_t) ggml_nrows(t))
	{
		die("%s: rows %zu..%zu out of range", name, row_first, row_first + n_rows);
	}
	std::vector<uint8_t> out(rs * n_rows);
	FILE * f = fopen(path.c_str(), "rb");
	if (f == nullptr)
	{
		die("cannot open %s", path.c_str());
	}
	const off_t at = (off_t) (data_off + gguf_get_tensor_offset(gg, id) + rs * row_first);
	if (fseeko(f, at, SEEK_SET) != 0 || fread(out.data(), 1, out.size(), f) != out.size())
	{
		die("short read of %s", name);
	}
	fclose(f);
	return out;
}

std::vector<float> TargetGguf::read_f32(const char * name, size_t row_first, size_t n_rows) const
{
	ggml_tensor * t = tensor(name);
	if (n_rows == 0)
	{
		n_rows = ggml_nrows(t) - row_first;
	}
	std::vector<uint8_t> raw = read_raw(name, row_first, n_rows);
	std::vector<float> out((size_t) t->ne[0] * n_rows);
	if (t->type == GGML_TYPE_F32)
	{
		memcpy(out.data(), raw.data(), out.size() * 4);
		return out;
	}
	const ggml_type_traits * tt = ggml_get_type_traits(t->type);
	if (tt->to_float == nullptr)
	{
		die("%s: type %s cannot be dequantised", name, ggml_type_name(t->type));
	}
	const size_t rs = ggml_row_size(t->type, t->ne[0]);
	for (size_t r = 0; r < n_rows; r++)
	{
		tt->to_float(raw.data() + rs * r, out.data() + r * t->ne[0], t->ne[0]);
	}
	return out;
}

// ------------------------------------------------------------------ init ---

static void set_f32(ggml_tensor * t, const std::vector<float> & v)
{
	if (v.size() != (size_t) ggml_nelements(t))
	{
		die("init %s: %zu values for %lld elements", t->name, v.size(), (long long) ggml_nelements(t));
	}
	ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
}

static std::vector<float> normal(size_t n, float sd, std::mt19937_64 & rng)
{
	std::normal_distribution<float> d(0.0f, sd);
	std::vector<float> v(n);
	for (auto & x : v)
	{
		x = d(rng);
	}
	return v;
}

static void set_normal(ggml_tensor * t, float sd, std::mt19937_64 & rng)
{
	set_f32(t, normal(ggml_nelements(t), sd, rng));
}

static void set_const(ggml_tensor * t, float c)
{
	set_f32(t, std::vector<float>(ggml_nelements(t), c));
}

void model_init_random(Model & m, uint64_t seed, float sd)
{
	std::mt19937_64 rng(seed);
	for (ggml_tensor * t : m.trainable())
	{
		if (ggml_n_dims(t) == 1)
		{
			std::vector<float> v = normal(ggml_nelements(t), 0.1f, rng);
			for (auto & x : v)
			{
				x += 1.0f;
			}
			set_f32(t, v);
		} else {
			set_normal(t, sd, rng);
		}
	}
	// a random Q8_0 token table
	std::vector<float> e = normal((size_t) m.hp.n_embd * m.hp.n_vocab, 1.0f, rng);
	std::vector<uint8_t> q(ggml_nbytes(m.tok_embd));
	ggml_quantize_chunk(GGML_TYPE_Q8_0, e.data(), q.data(), 0, m.hp.n_vocab, m.hp.n_embd, nullptr);
	ggml_backend_tensor_set(m.tok_embd, q.data(), 0, q.size());
	if (m.out_tgt)
	{
		std::vector<float> o = normal((size_t) m.hp.n_embd * m.hp.n_vocab, 0.05f, rng);
		ggml_quantize_chunk(GGML_TYPE_Q8_0, o.data(), q.data(), 0, m.hp.n_vocab, m.hp.n_embd, nullptr);
		ggml_backend_tensor_set(m.out_tgt, q.data(), 0, q.size());
	}
}

void model_init_from_target(Model & m, const TargetGguf & tg, const std::string & init, uint64_t seed)
{
	const Hparams & h = m.hp;
	std::mt19937_64 rng(seed);

	{
		ggml_tensor * te = tg.tensor("token_embd.weight");
		if (te->type != GGML_TYPE_Q8_0 || te->ne[0] != h.n_embd)
		{
			die("target token_embd.weight must be Q8_0 [%d, *]", h.n_embd);
		}
		std::vector<uint8_t> raw = tg.read_raw("token_embd.weight", D2T_BASE, h.n_vocab);
		ggml_backend_tensor_set(m.tok_embd, raw.data(), 0, raw.size());
	}
	if (m.out_tgt)
	{
		std::vector<uint8_t> raw = tg.read_raw("output.weight", D2T_BASE, h.n_vocab);
		if (raw.size() != ggml_nbytes(m.out_tgt))
		{
			die("output.weight slice size mismatch (not Q8_0?)");
		}
		ggml_backend_tensor_set(m.out_tgt, raw.data(), 0, raw.size());
	}

	// head + output_norm from the target in both inits
	set_f32(m.output, tg.read_f32("output.weight", D2T_BASE, h.n_vocab));
	set_f32(m.output_norm, tg.read_f32("output_norm.weight", 0, 0));

	if (init == "struct")
	{
		set_f32(m.ffn_gate, tg.read_f32("blk.27.ffn_gate.weight", 0, 0));
		set_f32(m.ffn_up, tg.read_f32("blk.27.ffn_up.weight", 0, 0));
		set_f32(m.ffn_down, tg.read_f32("blk.27.ffn_down.weight", 0, 0));
		set_f32(m.ffn_norm, tg.read_f32("blk.27.ffn_norm.weight", 0, 0));
		set_f32(m.attn_norm_2, tg.read_f32("blk.27.attn_norm.weight", 0, 0));
		set_const(m.attn_norm, 1.0f);
		// fc = [0 | 0 | I]: g = the high-layer feature
		std::vector<float> fc((size_t) h.n_feat * h.n_embd, 0.0f);
		const int off = h.n_feat - h.n_embd;
		for (int j = 0; j < h.n_embd; j++)
		{
			fc[(size_t) j * h.n_feat + off + j] = 1.0f;
		}
		set_f32(m.fc, fc);
		set_normal(m.wq, 0.02f, rng);
		set_normal(m.wk, 0.02f, rng);
		set_normal(m.wv, 0.02f, rng);
		set_normal(m.wo, 0.02f, rng);
	} else if (init == "random") {
		for (ggml_tensor * t : {m.fc, m.wq, m.wk, m.wv, m.wo, m.ffn_gate, m.ffn_up, m.ffn_down})
		{
			set_normal(t, 0.02f, rng);
		}
		for (ggml_tensor * t : {m.attn_norm, m.attn_norm_2, m.ffn_norm})
		{
			set_const(t, 1.0f);
		}
	} else {
		die("unknown --init %s", init.c_str());
	}
}

void model_load_checkpoint(Model & m, const std::string & path)
{
	ggml_context * cctx = nullptr;
	gguf_init_params ip = {false, &cctx};
	gguf_context * gg = gguf_init_from_file(path.c_str(), ip);
	if (gg == nullptr)
	{
		die("cannot read checkpoint %s", path.c_str());
	}
	for (ggml_tensor * t : m.trainable())
	{
		ggml_tensor * s = ggml_get_tensor(cctx, t->name);
		if (s == nullptr || s->type != GGML_TYPE_F32 || !ggml_are_same_shape(s, t))
		{
			die("checkpoint %s: tensor %s missing or not F32 of the right shape", path.c_str(), t->name);
		}
		ggml_backend_tensor_set(t, s->data, 0, ggml_nbytes(t));
	}
	gguf_free(gg);
	ggml_free(cctx);
}

void model_write_gguf(const Model & m, const TargetGguf & tg, const std::string & path, const std::string & name)
{
	const Hparams & h = m.hp;
	gguf_context * out = gguf_init_empty();
	gguf_set_kv(out, tg.gg);
	gguf_set_val_str(out, "general.architecture", "eagle3");
	gguf_set_val_str(out, "general.name", name.c_str());
	gguf_set_val_u32(out, "general.file_type", 0);   // all F32
	gguf_set_val_u32(out, "eagle3.context_length", h.n_ctx_orig);
	gguf_set_val_u32(out, "eagle3.embedding_length", h.n_embd);
	gguf_set_val_u32(out, "eagle3.block_count", 1);
	gguf_set_val_u32(out, "eagle3.feed_forward_length", h.n_ff);
	gguf_set_val_u32(out, "eagle3.attention.head_count", h.n_head);
	gguf_set_val_u32(out, "eagle3.attention.head_count_kv", h.n_head_kv);
	gguf_set_val_u32(out, "eagle3.attention.key_length", h.head_dim);
	gguf_set_val_u32(out, "eagle3.attention.value_length", h.head_dim);
	gguf_set_val_f32(out, "eagle3.attention.layer_norm_rms_epsilon", h.eps);
	gguf_set_val_f32(out, "eagle3.rope.freq_base", h.theta);
	gguf_set_val_u32(out, "eagle3.rope.dimension_count", h.head_dim);
	std::vector<int32_t> layers(h.layers.begin(), h.layers.end());
	gguf_set_arr_data(out, "eagle3.target_layers", GGUF_TYPE_INT32, layers.data(), layers.size());
	gguf_set_val_u32(out, "eagle3.target_hidden_size", h.n_feat / (int) layers.size());
	gguf_set_val_bool(out, "eagle3.norm_before_residual", false);
	gguf_set_val_bool(out, "eagle3.norm_before_fc", false);

	std::vector<ggml_tensor *> ts = m.trainable();
	size_t bytes = ggml_tensor_overhead() * (ts.size() + 2) + (size_t) h.n_vocab * 8 + 64;
	for (ggml_tensor * t : ts)
	{
		bytes += ggml_nbytes(t) + 64;
	}
	ggml_init_params ip = {bytes, nullptr, false};
	ggml_context * hctx = ggml_init(ip);
	for (ggml_tensor * t : ts)
	{
		ggml_tensor * c = ggml_dup_tensor(hctx, t);
		ggml_set_name(c, t->name);
		ggml_backend_tensor_get(t, c->data, 0, ggml_nbytes(t));
		gguf_add_tensor(out, c);
	}
	ggml_tensor * d2t = ggml_new_tensor_1d(hctx, GGML_TYPE_I64, h.n_vocab);
	ggml_set_name(d2t, "d2t");
	for (int i = 0; i < h.n_vocab; i++)
	{
		((int64_t *) d2t->data)[i] = D2T_BASE + i;
	}
	gguf_add_tensor(out, d2t);

	const std::string tmp = path + ".tmp";
	if (!gguf_write_to_file(out, tmp.c_str(), false) || rename(tmp.c_str(), path.c_str()) != 0)
	{
		die("cannot write %s", path.c_str());
	}
	gguf_free(out);
	ggml_free(hctx);
}
