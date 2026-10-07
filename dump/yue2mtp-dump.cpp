// yue2mtp-dump — teacher-forced feature dumper + M0 ceiling measurement
// (README "Dumping features"). One llama_context on the visible Vulkan device, one song at a
// time: decode prefix ++ semantic ++ [MUSIC_END] in 512-token ubatches, and for
// every semantic-phase position P = S-1 .. N-2 keep the target's layer inputs,
// final normed hidden, the recorded next token and the sampler-shaped label
// distribution. Cache layout: README "Cache format".

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "gguf.h"

#include "hash/hash.h"
#include "nlohmann/json.hpp"

#include "ref_sample_step.h"
#include "shaped_dist.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

namespace
{

const int32_t ABC_END     = 151848;
const int32_t MUSIC_START = shaped::MUSIC_START;
const int32_t MUSIC_END   = shaped::MUSIC_END;
const int32_t CODEC_OFF   = shaped::CODEC_OFFSET;
const int32_t N_LAB       = 100;
const int32_t SEM_ROWS    = 1 + shaped::CODEC_SIZE;   // MUSIC_END + codec block, contiguous

[[noreturn]] void die(const char * fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "yue2mtp-dump: ");
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	exit(1);
}

double now_s()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------ npy ---

std::vector<int32_t> load_npy_i32(const std::string & path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f)
	{
		die("cannot open %s", path.c_str());
	}
	std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (all.size() < 10 || all.compare(0, 6, "\x93NUMPY") != 0)
	{
		die("%s: not a .npy file", path.c_str());
	}
	const int major = (unsigned char) all[6];
	size_t    hlen, hoff;
	if (major == 1)
	{
		hlen = (unsigned char) all[8] | ((unsigned char) all[9] << 8);
		hoff = 10;
	} else {
		hlen = (unsigned char) all[8] | ((unsigned char) all[9] << 8) | ((size_t) (unsigned char) all[10] << 16) |
		       ((size_t) (unsigned char) all[11] << 24);
		hoff = 12;
	}
	const std::string hdr = all.substr(hoff, hlen);
	if (hdr.find("'<i4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos)
	{
		die("%s: expected little-endian int32 C-order, header %s", path.c_str(), hdr.c_str());
	}
	const size_t off = hoff + hlen;
	const size_t n   = (all.size() - off) / 4;
	std::vector<int32_t> v(n);
	memcpy(v.data(), all.data() + off, n * 4);
	return v;
}

json load_json(const std::string & path)
{
	std::ifstream f(path);
	if (!f)
	{
		die("cannot open %s", path.c_str());
	}
	return json::parse(f);
}

bool file_exists(const std::string & p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0;
}

int64_t file_size(const std::string & p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0 ? (int64_t) st.st_size : -1;
}

void mkdirs(const std::string & p)
{
	std::string cur;
	std::stringstream ss(p);
	std::string part;
	if (!p.empty() && p[0] == '/')
	{
		cur = "/";
	}
	while (std::getline(ss, part, '/'))
	{
		if (part.empty())
		{
			continue;
		}
		cur += part + "/";
		mkdir(cur.c_str(), 0755);
	}
}

std::string sha256_file(const std::string & path)
{
	const int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0)
	{
		die("cannot open %s", path.c_str());
	}
	const size_t n = (size_t) file_size(path);
	void * m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
	if (m == MAP_FAILED)
	{
		die("mmap %s failed", path.c_str());
	}
	std::string h = hash_sha256_hex(m, n);
	munmap(m, n);
	close(fd);
	return h;
}

// ----------------------------------------------------- output.weight rows ---

// MUSIC_END + codec rows of output.weight, dequantised to f32 (check 2).
std::vector<float> load_head_rows(const std::string & gguf_path, int n_embd)
{
	gguf_init_params gp = {true, nullptr};
	gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
	if (g == nullptr)
	{
		die("gguf_init_from_file(%s) failed", gguf_path.c_str());
	}
	const int64_t ti = gguf_find_tensor(g, "output.weight");
	if (ti < 0)
	{
		die("no output.weight in %s", gguf_path.c_str());
	}
	const ggml_type type = gguf_get_tensor_type(g, ti);
	const size_t    rs   = ggml_row_size(type, n_embd);
	const size_t    off  = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti) + rs * (size_t) MUSIC_END;
	gguf_free(g);

	std::vector<uint8_t> raw(rs * SEM_ROWS);
	FILE * f = fopen(gguf_path.c_str(), "rb");
	if (f == nullptr || fseeko(f, (off_t) off, SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f) != raw.size())
	{
		die("reading output.weight rows failed");
	}
	fclose(f);
	std::vector<float> w((size_t) SEM_ROWS * n_embd);
	const ggml_type_traits * tt = ggml_get_type_traits(type);
	if (tt->to_float == nullptr)
	{
		die("output.weight type %s has no to_float", ggml_type_name(type));
	}
	for (int r = 0; r < SEM_ROWS; r++)
	{
		tt->to_float(raw.data() + rs * r, w.data() + (size_t) r * n_embd, n_embd);
	}
	return w;
}

// ------------------------------------------------------------- cache file ---

// Little-endian, 256-byte header then six arrays at the recorded offsets (each
// 4096-aligned). See README "Cache format".
struct CacheHeader
{
	char     magic[8];      // "Y2MTPC01"
	uint32_t version;       // 1
	uint32_t n_pos;
	uint32_t S;
	uint32_t N;
	uint32_t n_embd;
	uint32_t n_layers;      // feature layers per row
	int32_t  layers[8];     // first n_layers used
	uint32_t n_lab;         // label slots per row (100)
	uint32_t reserved0;
	uint64_t off_tok;       // int32  [n_pos]
	uint64_t off_pmax_raw;  // f32    [n_pos]
	uint64_t off_lab_id;    // int32  [n_pos][n_lab], -1 padded
	uint64_t off_lab_p;     // f32    [n_pos][n_lab], 0 padded
	uint64_t off_hid;       // f16    [n_pos][n_embd]
	uint64_t off_feat;      // f16    [n_pos][n_layers][n_embd]
	uint64_t file_size;
	uint8_t  pad[256 - 8 - 4 * 6 - 4 * 8 - 8 - 8 * 7];
};
static_assert(sizeof(CacheHeader) == 256, "header must be 256 bytes");

uint64_t align4k(uint64_t x)
{
	return (x + 4095) & ~(uint64_t) 4095;
}

CacheHeader make_header(uint32_t n_pos, uint32_t S, uint32_t N, uint32_t n_embd, const std::vector<int> & layers)
{
	CacheHeader h;
	memset(&h, 0, sizeof(h));
	memcpy(h.magic, "Y2MTPC01", 8);
	h.version  = 1;
	h.n_pos    = n_pos;
	h.S        = S;
	h.N        = N;
	h.n_embd   = n_embd;
	h.n_layers = (uint32_t) layers.size();
	for (size_t i = 0; i < layers.size(); i++)
	{
		h.layers[i] = layers[i];
	}
	h.n_lab = N_LAB;
	uint64_t o      = 4096;
	h.off_tok       = o; o = align4k(o + 4ull * n_pos);
	h.off_pmax_raw  = o; o = align4k(o + 4ull * n_pos);
	h.off_lab_id    = o; o = align4k(o + 4ull * n_pos * N_LAB);
	h.off_lab_p     = o; o = align4k(o + 4ull * n_pos * N_LAB);
	h.off_hid       = o; o = align4k(o + 2ull * n_pos * n_embd);
	h.off_feat      = o; o = o + 2ull * n_pos * layers.size() * n_embd;
	h.file_size     = o;
	return h;
}

// ------------------------------------------------------------- M0 records ---

enum
{
	F_IN_SUPPORT = 1 << 0,   // recorded token inside the shaped support
	F_ARGMAX     = 1 << 1,   // recorded token == shaped argmax
	F_NG_HIT     = 1 << 2,   // bits 2..4: n-gram n=2,3,4 predicted the recorded token
	F_NG_HAVE    = 1 << 5,   // bits 5..7: n-gram n=2,3,4 had an earlier occurrence
	F_RAW_ARGMAX = 1 << 8,   // recorded token == raw full-vocab argmax
	F_SUPP_CUT   = 1 << 9,   // shaped support > 100 (tail not stored)
};

struct M0Rec
{
	uint32_t song;
	int32_t  step;          // P - (S-1)
	int32_t  n_pos;
	uint32_t flags;
	float    pmax;          // shaped
	float    ent;           // shaped, nats
	float    p_rec;         // shaped prob of the recorded token (0 if outside)
	float    pmax_raw;      // raw softmax over the full vocab
	float    ent_raw;       // nats, full vocab
	float    out_mass;      // raw mass outside {MUSIC_END} ∪ codec
	int32_t  supp;          // shaped support size
};

// ---------------------------------------------------------------- options ---

struct Opts
{
	std::string              model;
	std::vector<std::pair<std::string, std::string>> lists;   // (file, split)
	std::string              out = "data/cache";
	std::vector<int>         layers = {2, 14, 25};
	bool                     measure_only = false;
	std::string              report;
	int                      n_ctx = 16384;
	int                      threads = 8;
	int                      check_sampler_every = 97;   // rows; 0 = off
	int                      check_logits = 4;           // rows per song; 0 = off
	int                      max_songs = 0;
	bool                     resume = true;
	std::string              root;   // songs root: <root>/<name>/... (default: a "/songs/" path component)
};

void usage()
{
	fprintf(stderr,
		"usage: yue2mtp-dump -m MODEL.gguf --songs LIST[:SPLIT] [--songs LIST2:SPLIT2 ...]\n"
		"                    [--out data/cache] [--layers 2,14,25] [--measure-only]\n"
		"                    [--report FILE.md] [--n-ctx 16384] [--threads 8]\n"
		"                    [--check-sampler EVERY_N_ROWS (97, 0=off)] [--check-logits ROWS_PER_SONG (4)]\n"
		"                    [--max-songs N] [--no-resume] [--root SONGS_DIR]\n");
	exit(2);
}

Opts parse(int argc, char ** argv)
{
	Opts o;
	for (int i = 1; i < argc; i++)
	{
		std::string a = argv[i];
		auto next = [&]() -> std::string
		{
			if (i + 1 >= argc)
			{
				usage();
			}
			return argv[++i];
		};
		if (a == "-m")
		{
			o.model = next();
		} else if (a == "--songs") {
			std::string v = next();
			std::string split = "train";
			const size_t c = v.rfind(':');
			if (c != std::string::npos)
			{
				split = v.substr(c + 1);
				v     = v.substr(0, c);
			}
			o.lists.push_back({v, split});
		} else if (a == "--out") {
			o.out = next();
		} else if (a == "--layers") {
			o.layers.clear();
			std::stringstream ss(next());
			std::string t;
			while (std::getline(ss, t, ','))
			{
				o.layers.push_back(atoi(t.c_str()));
			}
		} else if (a == "--measure-only") {
			o.measure_only = true;
		} else if (a == "--report") {
			o.report = next();
		} else if (a == "--n-ctx") {
			o.n_ctx = atoi(next().c_str());
		} else if (a == "--threads") {
			o.threads = atoi(next().c_str());
		} else if (a == "--check-sampler") {
			o.check_sampler_every = atoi(next().c_str());
		} else if (a == "--check-logits") {
			o.check_logits = atoi(next().c_str());
		} else if (a == "--max-songs") {
			o.max_songs = atoi(next().c_str());
		} else if (a == "--no-resume") {
			o.resume = false;
		} else if (a == "--root") {
			o.root = next();
			while (o.root.size() > 1 && o.root.back() == '/')
			{
				o.root.pop_back();
			}
		} else {
			usage();
		}
	}
	if (o.model.empty() || o.lists.empty() || o.layers.empty() || o.layers.size() > 8)
	{
		usage();
	}
	return o;
}

struct Song
{
	std::string dir;
	std::string split;
	std::string name;       // <root>/<name>/...
	std::string take;       // rest of the path, '/' -> '_'
};

std::string song_key(const Song & s)
{
	return s.name + "__" + s.take;
}

// "<root>/<name>/<kind>/<take>" → name, kind_take. Without --root, <root> is the
// first ".../songs" component of the path; failing that, the whole path.
void split_song_path(Song & s, const std::string & root)
{
	std::string d = s.dir;
	while (!d.empty() && d.back() == '/')
	{
		d.pop_back();
	}
	std::string rel;
	if (!root.empty() && d.compare(0, root.size() + 1, root + "/") == 0)
	{
		rel = d.substr(root.size() + 1);
	} else {
		const size_t k = d.find("/songs/");
		rel = k == std::string::npos ? d : d.substr(k + 7);
	}
	while (!rel.empty() && rel[0] == '/')
	{
		rel.erase(0, 1);
	}
	const size_t sl = rel.find('/');
	s.name = sl == std::string::npos ? rel : rel.substr(0, sl);
	s.take = sl == std::string::npos ? "take" : rel.substr(sl + 1);
	for (auto & ch : s.take)
	{
		if (ch == '/')
		{
			ch = '_';
		}
	}
}

// Running stats for the report.
struct Checks
{
	int64_t pos = 0, in_support = 0;
	int64_t sampler_rows = 0, sampler_fail = 0;
	double  sampler_maxdp = 0;
	int64_t logit_rows = 0;
	double  logit_maxdiff_f32 = 0, logit_maxdiff_f16 = 0, logit_maxdiff_q8 = 0, logit_store = 0, logit_maxabs = 0;
	int64_t feat_stale = 0, feat_same_pos = 0;
	double  rms_first_sum = 0, rms_last_sum = 0;
	std::vector<double> rms_layer_sum;
	int64_t rms_n = 0;
	int64_t supp_cut = 0;
	std::vector<std::pair<std::string, double>> rejected;
	std::vector<std::pair<std::string, double>> song_support;
};

}

int main(int argc, char ** argv)
{
	const Opts o = parse(argc, argv);

	std::vector<Song> songs;
	for (const auto & l : o.lists)
	{
		std::ifstream f(l.first);
		if (!f)
		{
			die("cannot open %s", l.first.c_str());
		}
		std::string line;
		while (std::getline(f, line))
		{
			if (line.empty() || line[0] == '#')
			{
				continue;
			}
			Song s;
			s.dir   = line;
			s.split = l.second;
			split_song_path(s, o.root);
			songs.push_back(s);
		}
	}
	if (o.max_songs > 0 && (int) songs.size() > o.max_songs)
	{
		songs.resize(o.max_songs);
	}
	fprintf(stderr, "yue2mtp-dump: %zu songs, layers", songs.size());
	for (int L : o.layers)
	{
		fprintf(stderr, " %d", L);
	}
	fprintf(stderr, ", %s\n", o.measure_only ? "measure only" : "writing cache");

	mkdirs(o.out);
	const double t_hash = now_s();
	const std::string gguf_sha = sha256_file(o.model);
	fprintf(stderr, "yue2mtp-dump: model sha256 %s (%.1f s)\n", gguf_sha.c_str(), now_s() - t_hash);

	llama_backend_init();
	llama_model_params mp = llama_model_default_params();
	mp.n_gpu_layers = 999;
	llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
	if (model == nullptr)
	{
		die("cannot load %s", o.model.c_str());
	}
	const int n_embd  = llama_model_n_embd(model);
	const int n_layer = llama_model_n_layer(model);
	const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
	for (int L : o.layers)
	{
		if (L < 0 || L >= n_layer)
		{
			die("layer %d out of range [0, %d)", L, n_layer);
		}
	}
	if (n_vocab < CODEC_OFF + shaped::CODEC_SIZE)
	{
		die("vocab %d too small", n_vocab);
	}

	const int n_batch = 512;
	llama_context_params cp = llama_context_default_params();
	cp.n_ctx           = (uint32_t) o.n_ctx;
	cp.n_batch         = n_batch;
	cp.n_ubatch        = n_batch;
	cp.n_seq_max       = 1;
	cp.embeddings      = true;
	cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
	cp.no_perf         = true;
	cp.n_threads       = o.threads;
	cp.n_threads_batch = o.threads;
	llama_context * ctx = llama_init_from_model(model, cp);
	if (ctx == nullptr)
	{
		die("cannot create context (n_ctx %d)", o.n_ctx);
	}
	for (int L : o.layers)
	{
		llama_set_embeddings_layer_inp(ctx, (uint32_t) L, true);
	}
	llama_memory_t mem = llama_get_memory(ctx);

	std::vector<float> head_rows;
	if (o.check_logits > 0)
	{
		head_rows = load_head_rows(o.model, n_embd);
	}

	llama_batch batch = llama_batch_init(n_batch, 0, 1);
	const int   NL    = (int) o.layers.size();

	Checks ck;
	ck.rms_layer_sum.assign(NL, 0.0);
	json index = json::array();
	std::string index_path = o.out + "/index.json";
	if (o.resume && !o.measure_only && file_exists(index_path))
	{
		index = load_json(index_path);
	}
	std::map<std::string, size_t> index_at;
	for (size_t i = 0; i < index.size(); i++)
	{
		index_at[index[i]["split"].get<std::string>() + "/" + index[i]["file"].get<std::string>()] = i;
	}

	const std::string m0_path = o.out + (o.measure_only ? "/m0_positions.measure.bin" : "/m0_positions.bin");
	FILE * m0f = fopen(m0_path.c_str(), "wb");
	if (m0f == nullptr)
	{
		die("cannot write %s", m0_path.c_str());
	}
	json m0_songs = json::array();

	double  t_decode = 0, t_cpu = 0, t_write = 0;
	int64_t tok_decoded = 0, pos_done = 0;
	int64_t bytes_written = 0;
	const double t_start = now_s();

	for (size_t si = 0; si < songs.size(); si++)
	{
		const Song & sg = songs[si];
		const std::string fname = song_key(sg) + ".bin";
		const std::string fpath = o.out + "/" + sg.split + "/" + fname;

		const json res = load_json(sg.dir + "/result.json");
		std::vector<int32_t> seq = load_npy_i32(sg.dir + "/prefix.npy");
		const std::vector<int32_t> sem = load_npy_i32(sg.dir + "/semantic.npy");
		if (seq.size() < 2 || seq[seq.size() - 2] != ABC_END || seq.back() != MUSIC_START)
		{
			fprintf(stderr, "SKIP %s: prefix does not end in ABC_END MUSIC_START\n", sg.dir.c_str());
			continue;
		}
		if ((int64_t) sem.size() + 1 != res["timing"]["semantic"]["output_tokens"].get<int64_t>() ||
		    res["truncated"]["semantic"].get<bool>())
		{
			fprintf(stderr, "SKIP %s: len(semantic)+1 != output_tokens or truncated\n", sg.dir.c_str());
			continue;
		}
		const int S = (int) seq.size();
		for (int32_t c : sem)
		{
			if (c < 0 || c >= shaped::CODEC_SIZE)
			{
				die("%s: codec id %d out of range", sg.dir.c_str(), c);
			}
			seq.push_back(c + CODEC_OFF);
		}
		seq.push_back(MUSIC_END);
		const int N     = (int) seq.size();
		const int n_pos = N - S;
		if (N - 1 > o.n_ctx)
		{
			fprintf(stderr, "SKIP %s: N %d > n_ctx %d (rerun with a larger --n-ctx)\n", sg.dir.c_str(), N, o.n_ctx);
			continue;
		}

		const bool have = !o.measure_only && o.resume && index_at.count(sg.split + "/" + fname) &&
		                  file_size(fpath) == (int64_t) make_header(n_pos, S, N, n_embd, o.layers).file_size;
		if (have)
		{
			fprintf(stderr, "[%zu/%zu] %s: cached, skipping\n", si + 1, songs.size(), fname.c_str());
			continue;
		}

		// per-song outputs
		std::vector<int32_t>     tok(n_pos);
		std::vector<float>       pmax_raw(n_pos);
		std::vector<int32_t>     lab_id((size_t) n_pos * N_LAB, -1);
		std::vector<float>       lab_p((size_t) n_pos * N_LAB, 0.0f);
		std::vector<ggml_fp16_t> hid, feat;
		if (!o.measure_only)
		{
			hid.resize((size_t) n_pos * n_embd);
			feat.resize((size_t) n_pos * NL * n_embd);
		}
		std::vector<M0Rec> recs(n_pos);
		for (int r = 0; r < n_pos; r++)
		{
			tok[r] = seq[S + r];
		}

		// rows for check 2: evenly spaced
		std::vector<int> logit_rows;
		for (int k = 0; k < o.check_logits; k++)
		{
			logit_rows.push_back((int) ((int64_t) (2 * k + 1) * n_pos / (2 * o.check_logits)));
		}
		// feature RMS / stale checks keep two f32 rows per layer
		std::vector<float> feat_prev_batch;   // row i of the previous batch, layer 0
		int64_t            stale = 0;

		llama_memory_clear(mem, true);

		std::atomic<int64_t> s_in{0}, s_fail{0}, s_rows{0}, s_cut{0};
		double s_maxdp = 0;
		double s_ldiff32 = 0, s_ldiff16 = 0, s_ldiffq8 = 0, s_lstore = 0, s_lmax = 0;
		int    s_lrows = 0;
		std::vector<double> s_rms(NL, 0.0);
		int64_t s_rms_n = 0;

		const double t_song = now_s();
		for (int off = 0; off < N - 1; off += n_batch)
		{
			const int nt = std::min(n_batch, N - 1 - off);
			batch.n_tokens = 0;
			for (int i = 0; i < nt; i++)
			{
				batch.token[i]     = seq[off + i];
				batch.pos[i]       = off + i;
				batch.n_seq_id[i]  = 1;
				batch.seq_id[i][0] = 0;
				batch.logits[i]    = true;   // embeddings=true outputs every row anyway
			}
			batch.n_tokens = nt;
			double t0 = now_s();
			if (llama_decode(ctx, batch) != 0)
			{
				die("llama_decode failed (%s, off %d)", sg.dir.c_str(), off);
			}
			const int r0 = std::max(0, S - 1 - off);   // first batch row with P >= S-1
			std::vector<const float *> linp(NL);
			for (int l = 0; l < NL; l++)
			{
				linp[l] = llama_get_embeddings_layer_inp(ctx, (uint32_t) o.layers[l]);
			}
			t_decode += now_s() - t0;
			tok_decoded += nt;

			// stale-buffer check: layer-0 input of row 0 vs the previous batch's row 0
			if (!feat_prev_batch.empty() && memcmp(feat_prev_batch.data(), linp[0], n_embd * sizeof(float)) == 0)
			{
				stale++;
			}
			feat_prev_batch.assign(linp[0], linp[0] + n_embd);

			if (r0 >= nt)
			{
				continue;
			}
			t0 = now_s();

			// copy features / hidden (f16) and RMS
			for (int i = r0; i < nt; i++)
			{
				const int r = off + i - (S - 1);
				const float * h = llama_get_embeddings_ith(ctx, i);
				if (h == nullptr)
				{
					die("no embeddings for row %d", i);
				}
				if (!o.measure_only)
				{
					ggml_fp32_to_fp16_row(h, &hid[(size_t) r * n_embd], n_embd);
					for (int l = 0; l < NL; l++)
					{
						ggml_fp32_to_fp16_row(linp[l] + (size_t) i * n_embd, &feat[((size_t) r * NL + l) * n_embd], n_embd);
					}
				}
				for (int l = 0; l < NL; l++)
				{
					double ss = 0;
					const float * x = linp[l] + (size_t) i * n_embd;
					for (int e = 0; e < n_embd; e++)
					{
						ss += (double) x[e] * x[e];
					}
					s_rms[l] += std::sqrt(ss / n_embd);
				}
				s_rms_n++;

				// check 2: hid · output rows vs llama's logits
				if (std::find(logit_rows.begin(), logit_rows.end(), r) != logit_rows.end() && !head_rows.empty())
				{
					const float * lg = llama_get_logits_ith(ctx, i);
					std::vector<float> h16(n_embd);
					std::vector<ggml_fp16_t> tmp(n_embd);
					ggml_fp32_to_fp16_row(h, tmp.data(), n_embd);
					ggml_fp16_to_fp32_row(tmp.data(), h16.data(), n_embd);
					// llama's Vulkan matmul quantizes the activation to q8_1 blocks
					// before the integer dot; q8_0 round-trip = the same values
					std::vector<float>   hq(n_embd);
					std::vector<uint8_t> qb(ggml_row_size(GGML_TYPE_Q8_0, n_embd));
					const ggml_type_traits * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
					q8->from_float_ref(h, qb.data(), n_embd);
					q8->to_float(qb.data(), hq.data(), n_embd);
					for (int v = 0; v < SEM_ROWS; v++)
					{
						const float * w = &head_rows[(size_t) v * n_embd];
						double a = 0, b = 0, c = 0;
						for (int e = 0; e < n_embd; e++)
						{
							a += (double) w[e] * h[e];
							b += (double) w[e] * h16[e];
							c += (double) w[e] * hq[e];
						}
						const double ref = lg[MUSIC_END + v];
						s_ldiff32 = std::max(s_ldiff32, std::fabs(a - ref));
						s_ldiff16 = std::max(s_ldiff16, std::fabs(b - ref));
						s_ldiffq8 = std::max(s_ldiffq8, std::fabs(c - ref));
						s_lstore  = std::max(s_lstore, std::fabs(b - a));
						s_lmax    = std::max(s_lmax, std::fabs(ref));
					}
					s_lrows++;
				}
			}

			// labels + M0 per row, in parallel (row pointers fetched here: the
			// llama_get_* accessors synchronize and are not for worker threads)
			std::vector<const float *> lgp(nt, nullptr);
			for (int i = r0; i < nt; i++)
			{
				lgp[i] = llama_get_logits_ith(ctx, i);
				if (lgp[i] == nullptr)
				{
					die("no logits for row %d", i);
				}
			}
			std::vector<std::thread> pool;
			std::vector<double>      maxdp(o.threads, 0.0);
			for (int th = 0; th < o.threads; th++)
			{
				pool.emplace_back([&, th]()
				{
					shaped::Scratch sc;
					std::vector<int32_t> id(40000);
					std::vector<double>  p(40000);
					std::vector<int32_t> rid;
					std::vector<double>  rp;
					for (int i = r0 + th; i < nt; i += o.threads)
					{
						const int     P  = off + i;
						const int     r  = P - (S - 1);
						const float * lg = lgp[i];
						const int n = shaped::dist(lg, seq.data() + S, (size_t) r, r, id.data(), p.data(),
						                           (int) id.size(), sc);
						M0Rec & m = recs[r];
						m.song  = (uint32_t) si;
						m.step  = r;
						m.n_pos = n_pos;
						m.flags = 0;
						m.supp  = n;
						m.pmax  = (float) p[0];
						double ent = 0;
						m.p_rec = 0;
						for (int k = 0; k < n; k++)
						{
							ent -= p[k] * std::log(p[k]);
							if (id[k] == tok[r])
							{
								m.p_rec = (float) p[k];
								m.flags |= F_IN_SUPPORT;
							}
						}
						m.ent = (float) ent;
						if (id[0] == tok[r])
						{
							m.flags |= F_ARGMAX;
						}
						if (n > N_LAB)
						{
							m.flags |= F_SUPP_CUT;
							s_cut++;
						}
						for (int k = 0; k < std::min(n, N_LAB); k++)
						{
							lab_id[(size_t) r * N_LAB + k] = id[k];
							lab_p[(size_t) r * N_LAB + k]  = (float) p[k];
						}
						if (m.flags & F_IN_SUPPORT)
						{
							s_in++;
						}

						// raw softmax over the full vocab
						float mx = lg[0];
						int   am = 0;
						for (int v = 1; v < n_vocab; v++)
						{
							if (lg[v] > mx)
							{
								mx = lg[v];
								am = v;
							}
						}
						double z = 0, zl = 0, zin = 0;
						for (int v = 0; v < n_vocab; v++)
						{
							const double d = (double) lg[v] - mx;
							const double e = std::exp(d);
							z  += e;
							zl += e * d;
							if (v >= MUSIC_END && v < CODEC_OFF + shaped::CODEC_SIZE)
							{
								zin += e;
							}
						}
						m.pmax_raw  = (float) (1.0 / z);
						m.ent_raw   = (float) (std::log(z) - zl / z);
						m.out_mass  = (float) (1.0 - zin / z);
						pmax_raw[r] = m.pmax_raw;
						if (am == tok[r])
						{
							m.flags |= F_RAW_ARGMAX;
						}

						// unit test on real rows: shaped::dist vs yue2.cpp's sample_step
						if (o.check_sampler_every > 0 && (r % o.check_sampler_every == 0 || r == n_pos - 1 || r == 199 || r == 200))
						{
							std::vector<int32_t> hist(seq.begin() + S, seq.begin() + S + r);
							ref_semantic_dist(lg, n_vocab, hist, r, 1, rid, rp);
							bool ok = (int) rid.size() == n;
							for (int k = 0; ok && k < n; k++)
							{
								ok = rid[k] == id[k];
								maxdp[th] = std::max(maxdp[th], std::fabs(rp[k] - p[k]));
							}
							s_rows++;
							if (!ok)
							{
								s_fail++;
							}
						}
					}
				});
			}
			for (auto & t : pool)
			{
				t.join();
			}
			for (double d : maxdp)
			{
				s_maxdp = std::max(s_maxdp, d);
			}
			t_cpu += now_s() - t0;
		}
		const double song_s = now_s() - t_song;

		// n-gram baseline over this song's own semantic tokens (incl. MUSIC_END)
		for (int n = 2; n <= 4; n++)
		{
			std::unordered_map<uint64_t, int> last;
			for (int r = n; r < n_pos; r++)
			{
				uint64_t key = 0;
				for (int j = r - n; j < r; j++)
				{
					key = (key << 16) | (uint64_t) (tok[j] - MUSIC_END);
				}
				auto it = last.find(key);
				if (it != last.end())
				{
					recs[r].flags |= F_NG_HAVE << (n - 2);
					if (tok[it->second] == tok[r])
					{
						recs[r].flags |= F_NG_HIT << (n - 2);
					}
				}
				last[key] = r;
			}
		}

		// feature at two positions must differ
		int64_t same = 0;
		if (!o.measure_only && n_pos > 2)
		{
			const size_t a = 0, b = (size_t) n_pos / 2;
			same = memcmp(&feat[a * NL * n_embd], &feat[b * NL * n_embd], NL * n_embd * sizeof(ggml_fp16_t)) == 0;
		}

		const double rate = (double) s_in / n_pos;
		ck.pos            += n_pos;
		ck.in_support     += s_in;
		ck.sampler_rows   += s_rows;
		ck.sampler_fail   += s_fail;
		ck.sampler_maxdp   = std::max(ck.sampler_maxdp, s_maxdp);
		ck.logit_rows     += s_lrows;
		ck.logit_maxdiff_f32 = std::max(ck.logit_maxdiff_f32, s_ldiff32);
		ck.logit_maxdiff_f16 = std::max(ck.logit_maxdiff_f16, s_ldiff16);
		ck.logit_maxdiff_q8  = std::max(ck.logit_maxdiff_q8, s_ldiffq8);
		ck.logit_store       = std::max(ck.logit_store, s_lstore);
		ck.logit_maxabs      = std::max(ck.logit_maxabs, s_lmax);
		ck.feat_stale     += stale;
		ck.feat_same_pos  += same;
		ck.supp_cut       += s_cut;
		for (int l = 0; l < NL; l++)
		{
			ck.rms_layer_sum[l] += s_rms[l];
		}
		ck.rms_n += s_rms_n;
		ck.song_support.push_back({song_key(sg), rate});

		fwrite(recs.data(), sizeof(M0Rec), recs.size(), m0f);
		json ms = {{"song", sg.name}, {"take", sg.take}, {"dir", sg.dir}, {"split", sg.split},
		           {"n_pos", n_pos}, {"S", S}, {"N", N}, {"support_rate", rate}};
		m0_songs.push_back(ms);
		pos_done += n_pos;

		const bool reject = rate < 0.99;
		if (reject)
		{
			ck.rejected.push_back({song_key(sg), rate});
		}

		double t0 = now_s();
		if (!o.measure_only && !reject)
		{
			mkdirs(o.out + "/" + sg.split);
			const CacheHeader h = make_header(n_pos, S, N, n_embd, o.layers);
			const std::string tmp = fpath + ".tmp";
			FILE * f = fopen(tmp.c_str(), "wb");
			if (f == nullptr)
			{
				die("cannot write %s", tmp.c_str());
			}
			auto put = [&](uint64_t at, const void * p, size_t n)
			{
				if (fseeko(f, (off_t) at, SEEK_SET) != 0 || fwrite(p, 1, n, f) != n)
				{
					die("write to %s failed", tmp.c_str());
				}
			};
			put(0, &h, sizeof(h));
			put(h.off_tok, tok.data(), tok.size() * 4);
			put(h.off_pmax_raw, pmax_raw.data(), pmax_raw.size() * 4);
			put(h.off_lab_id, lab_id.data(), lab_id.size() * 4);
			put(h.off_lab_p, lab_p.data(), lab_p.size() * 4);
			put(h.off_hid, hid.data(), hid.size() * 2);
			put(h.off_feat, feat.data(), feat.size() * 2);
			if (fclose(f) != 0 || rename(tmp.c_str(), fpath.c_str()) != 0)
			{
				die("finishing %s failed", fpath.c_str());
			}
			bytes_written += (int64_t) h.file_size;

			json e = {{"song", sg.name}, {"take", sg.take}, {"dir", sg.dir}, {"split", sg.split},
			          {"file", fname}, {"n_pos", n_pos}, {"S", S}, {"N", N}, {"gguf_sha256", gguf_sha},
			          {"layers", o.layers}, {"support_rate", rate}, {"bytes", h.file_size}};
			const std::string k = sg.split + "/" + fname;
			if (index_at.count(k))
			{
				index[index_at[k]] = e;
			} else {
				index_at[k] = index.size();
				index.push_back(e);
			}
			std::ofstream(index_path + ".tmp") << index.dump(1);
			rename((index_path + ".tmp").c_str(), index_path.c_str());
		}
		t_write += now_s() - t0;

		fprintf(stderr, "[%zu/%zu] %s: N %d S %d n_pos %d, %.1f s (%.0f tok/s), support %.4f%s, sampler %lld/%lld ok, "
		        "logit diff f32 %.2e f16 %.2e q8act %.2e%s\n",
		        si + 1, songs.size(), fname.c_str(), N, S, n_pos, song_s, (N - 1) / song_s, rate,
		        reject ? " REJECTED" : "", (long long) (s_rows - s_fail), (long long) s_rows.load(), s_ldiff32, s_ldiff16, s_ldiffq8,
		        stale || same ? " FEATURE-STALE" : "");
	}
	fclose(m0f);
	{
		std::ofstream(o.out + (o.measure_only ? "/m0_songs.measure.json" : "/m0_songs.json")) << m0_songs.dump(1);
	}

	const double wall = now_s() - t_start;
	json summary = {
		{"songs", m0_songs.size()},
		{"positions", ck.pos},
		{"tokens_decoded", tok_decoded},
		{"wall_s", wall},
		{"decode_s", t_decode},
		{"cpu_s", t_cpu},
		{"write_s", t_write},
		{"decode_tok_per_s", tok_decoded / std::max(t_decode, 1e-9)},
		{"overall_tok_per_s", tok_decoded / std::max(wall, 1e-9)},
		{"positions_per_s", pos_done / std::max(wall, 1e-9)},
		{"bytes_written", bytes_written},
		{"check1_support_rate", (double) ck.in_support / std::max<int64_t>(ck.pos, 1)},
		{"check1_rejected", ck.rejected},
		{"check1_min_song_rate", ck.song_support.empty() ? 0.0 :
			std::min_element(ck.song_support.begin(), ck.song_support.end(),
				[](const auto & a, const auto & b) { return a.second < b.second; })->second},
		{"support_gt_100", ck.supp_cut},
		{"sampler_rows", ck.sampler_rows},
		{"sampler_fail", ck.sampler_fail},
		{"sampler_max_dp", ck.sampler_maxdp},
		{"check2_rows", ck.logit_rows},
		{"check2_maxdiff_f32", ck.logit_maxdiff_f32},
		{"check2_maxdiff_f16", ck.logit_maxdiff_f16},
		{"check2_maxdiff_q8act", ck.logit_maxdiff_q8},
		{"check2_f16_vs_f32_hid", ck.logit_store},
		{"check2_max_abs_logit", ck.logit_maxabs},
		{"check3_stale_batches", ck.feat_stale},
		{"check3_same_feat_positions", ck.feat_same_pos},
		{"layers", o.layers},
		{"gguf_sha256", gguf_sha},
	};
	std::vector<double> rms;
	for (int l = 0; l < NL; l++)
	{
		rms.push_back(ck.rms_layer_sum[l] / std::max<int64_t>(ck.rms_n, 1));
	}
	summary["check3_mean_rms_per_layer"] = rms;
	const std::string sp = o.out + (o.measure_only ? "/summary.measure.json" : "/summary.json");
	std::ofstream(sp) << summary.dump(1);
	fprintf(stderr, "%s\n", summary.dump(1).c_str());
	if (!o.report.empty())
	{
		const std::string cmd = std::string("python3 '") + Y2M_TOOLS_DIR + "/m0_report.py' '" + o.out + "'" +
		                        (o.measure_only ? " --measure" : "") + " > '" + o.report + "'";
		if (system(cmd.c_str()) != 0)
		{
			fprintf(stderr, "yue2mtp-dump: report command failed: %s\n", cmd.c_str());
		}
	}

	llama_batch_free(batch);
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
	return 0;
}
