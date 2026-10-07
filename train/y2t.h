// EAGLE-3 draft-head trainer: shared declarations.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

// ------------------------------------------------------------- constants ---

constexpr int32_t MUSIC_END   = 151852;   // draft row 0
constexpr int32_t CODEC_FIRST = 151853;   // draft rows 1..32768
constexpr int32_t D2T_BASE    = 151852;   // d2t[i] = D2T_BASE + i
constexpr int     N_LAB       = 100;      // label slots per cache row

[[noreturn]] void die(const char * fmt, ...);
double now_s();

// --------------------------------------------------------------- hparams ---

struct Hparams
{
	int   n_embd     = 2048;
	int   n_head     = 16;
	int   n_head_kv  = 8;
	int   head_dim   = 128;
	int   n_ff       = 6144;
	int   n_vocab    = 32769;   // draft rows (= rows of the tok_embd slice)
	int   n_feat     = 6144;    // 3 target layers x n_embd_tgt
	float eps        = 1e-6f;
	float theta      = 1e6f;
	int   n_ctx_orig = 24576;
	std::vector<int> layers = {2, 14, 25};

	static Hparams tiny();       // gradient-check config (yue2mtp-check gradcheck)
};

// ----------------------------------------------------------------- model ---

struct Model
{
	Hparams hp;
	ggml_context *        ctx = nullptr;
	ggml_backend_buffer_t buf = nullptr;

	// frozen
	ggml_tensor * tok_embd = nullptr;   // Q8_0 [n_embd, n_vocab]: target token_embd rows D2T_BASE..
	ggml_tensor * out_tgt  = nullptr;   // Q8_0 [n_embd, n_vocab]: target output rows (raw labels only)

	// trainable (F32)
	ggml_tensor * fc          = nullptr; // [n_feat, n_embd]
	ggml_tensor * attn_norm   = nullptr; // [n_embd]
	ggml_tensor * attn_norm_2 = nullptr; // [n_embd]
	ggml_tensor * wq          = nullptr; // [2 n_embd, n_head hd]
	ggml_tensor * wk          = nullptr; // [2 n_embd, n_kv hd]
	ggml_tensor * wv          = nullptr; // [2 n_embd, n_kv hd]
	ggml_tensor * wo          = nullptr; // [n_head hd, n_embd]
	ggml_tensor * ffn_norm    = nullptr; // [n_embd]
	ggml_tensor * ffn_gate    = nullptr; // [n_embd, n_ff]
	ggml_tensor * ffn_up      = nullptr; // [n_embd, n_ff]
	ggml_tensor * ffn_down    = nullptr; // [n_ff, n_embd]
	ggml_tensor * output_norm = nullptr; // [n_embd]
	ggml_tensor * output      = nullptr; // [n_embd, n_vocab]

	std::vector<ggml_tensor *> trainable() const;   // in GGUF order
	void alloc(const Hparams & hp, ggml_backend_t backend, bool with_out_tgt);
	void free_all();
};

// host-side copy of all tensors, keyed by GGUF name
struct HostTensor
{
	std::string          name;
	ggml_type            type;
	std::vector<int64_t> ne;
	std::vector<uint8_t> data;
};

// target GGUF access (read-only)
struct TargetGguf
{
	std::string path;
	struct gguf_context * gg = nullptr;
	ggml_context *        meta = nullptr;
	size_t                data_off = 0;

	void open(const std::string & path);
	void close();
	ggml_tensor * tensor(const char * name) const;
	std::vector<uint8_t> read_raw(const char * name, size_t row_first, size_t n_rows) const;
	std::vector<float>   read_f32(const char * name, size_t row_first, size_t n_rows) const;
};

// init: frozen tensors + trainable init (struct|random) from the target
void model_init_from_target(Model & m, const TargetGguf & tg, const std::string & init, uint64_t seed);
// random everything (tiny gradcheck / synthetic runs)
void model_init_random(Model & m, uint64_t seed, float sd);
// copy trainable weights from a checkpoint GGUF (weights only)
void model_load_checkpoint(Model & m, const std::string & path);
// write an eagle3 draft GGUF (KVs copied from the target)
void model_write_gguf(const Model & m, const TargetGguf & tg, const std::string & path, const std::string & name);

// copy every tensor of src to dst (same shapes, possibly different backends)
void model_copy_weights(const Model & src, Model & dst);

// --------------------------------------------------------------- inputs ---

struct Inputs
{
	int n = 0;   // rows per chunk
	int K = 0;   // unroll depth
	int L = N_LAB;
	bool raw_labels = false;

	ggml_context *        ctx = nullptr;
	ggml_backend_buffer_t buf = nullptr;

	ggml_tensor * feat    = nullptr;   // F32 [n_feat, n]
	ggml_tensor * tok     = nullptr;   // I32 [n, K]       draft rows
	ggml_tensor * pos     = nullptr;   // I32 [n, K]       RoPE positions
	ggml_tensor * rmask   = nullptr;   // F32 [1, n, K]    1 = real row
	ggml_tensor * w       = nullptr;   // F32 [K]          per-depth loss weight
	ggml_tensor * lab_idx = nullptr;   // I32 [n L, K]     row*V + draft id
	ggml_tensor * lab_val = nullptr;   // F32 [1, n L, K]
	ggml_tensor * top_idx = nullptr;   // I32 [n, K]       row*V + argmax p (eval)
	ggml_tensor * hid     = nullptr;   // F32 [n_embd, n, K] (raw labels)
	ggml_tensor * ce_out  = nullptr;   // F32 [K]          per-depth CE (written by the graph)
	std::vector<ggml_tensor *> kq_mask; // F32 [(k+1) n, n]

	void alloc(const Hparams & hp, int n, int K, bool raw_labels, ggml_backend_t backend);
	void set_masks();
	void free_all();
};

// one chunk worth of host data
struct Batch
{
	int n = 0, K = 0, L = N_LAB;
	std::vector<float>   feat;      // [n][n_feat]
	std::vector<int32_t> tok;       // [K][n]
	std::vector<int32_t> pos;       // [K][n]
	std::vector<float>   rmask;     // [K][n]
	std::vector<float>   w;         // [K]
	std::vector<int32_t> lab_idx;   // [K][n*L]
	std::vector<float>   lab_val;   // [K][n*L]
	std::vector<int32_t> top_idx;   // [K][n]
	std::vector<float>   hid;       // [K][n][n_embd] (raw labels)
	std::vector<int>     n_real;    // [K]
	// sparse labels for host metrics: [K][n][L] draft ids (-1 pad) and p
	std::vector<int32_t> sp_id;
	std::vector<float>   sp_p;
	std::string          tag;       // song:offset
};

void batch_upload(const Batch & b, Inputs & in);

// ---------------------------------------------------------------- graph ---

struct GraphOut
{
	std::vector<ggml_tensor *> logits;   // [V, n] per depth
	std::vector<ggml_tensor *> ce;       // scalar per depth (raw mean over all n rows)
	ggml_tensor * loss = nullptr;        // Σ_k w_k ce_k
};

// builds forward + loss. with_ce_out: copy each ce_k into in.ce_out (static) on the way
GraphOut build_forward(ggml_context * ctx, const Model & m, const Inputs & in, bool with_loss, bool with_ce_out);

struct EvalOut
{
	std::vector<ggml_tensor *> am;     // I32 [n]     argmax q
	std::vector<ggml_tensor *> lse;    // F32 [1, n]  log Σ exp(logits)
	std::vector<ggml_tensor *> lsup;   // F32 [1, nL] logits at the label support
};
EvalOut build_eval(ggml_context * ctx, const Model & m, const Inputs & in, const GraphOut & go);

// ---------------------------------------------------------------- cache ---

struct CacheSong
{
	std::string name;     // index song
	std::string file;
	std::string split;
	int n_pos = 0;
	int S = 0;
	// mmap
	int      fd = -1;
	uint8_t * map = nullptr;
	size_t   map_size = 0;
	const int32_t  * tok = nullptr;
	const int32_t  * lab_id = nullptr;
	const float    * lab_p = nullptr;
	const uint16_t * hid = nullptr;
	const uint16_t * feat = nullptr;
	int n_layers = 0, n_embd = 0, n_lab = 0;
};

struct Cache
{
	std::string dir;
	std::vector<CacheSong> songs;
	std::vector<int> layers;

	void open(const std::string & dir, const std::string & split, int max_songs = 0);
	void close();
	int64_t total_pos() const;
};

struct ChunkRef
{
	int song;
	int start;   // first cache row
	int n_real;  // rows that are origins (<= n)
};

std::vector<ChunkRef> make_chunks(const Cache & c, int n, int min_rows, bool random_offset, std::mt19937_64 & rng);
void build_batch(const Cache & c, const ChunkRef & cr, const Hparams & hp, int n, int K, bool raw_labels, Batch & b);

// synthetic cache in the dumper's format (development / tests)
void write_synthetic_cache(const std::string & dir, const Hparams & hp, int n_songs_train, int n_songs_heldout,
                           int min_len, int max_len, uint64_t seed);

// ------------------------------------------------------------- backends ---

struct Backends
{
	ggml_backend_t main = nullptr;   // vulkan or cpu
	ggml_backend_t cpu  = nullptr;
	bool is_vk = false;
	std::vector<ggml_backend_t> list() const;
	void init(const std::string & which, int threads);
	void free_all();
};

// Σ x² per tensor, computed on the tensor's backend (small separate graph)
std::vector<double> sum_sq(ggml_backend_t backend, const std::vector<ggml_tensor *> & ts);

// host metrics for one evaluated batch
struct EvalAcc
{
	int K = 0;
	std::vector<double> n, kl, top1, accA, accB, accBr, ce, ent;
	void init(int K);
	void add(const Batch & b, int V, const std::vector<std::vector<int32_t>> & am,
	         const std::vector<std::vector<float>> & lse, const std::vector<std::vector<float>> & lsup);
	std::string report() const;
};
