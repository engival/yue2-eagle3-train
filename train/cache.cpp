// Feature cache reader (format: README "Cache format", Y2MTPC01),
// chunking and batch assembly (README "Objective"), synthetic cache writer.
#include "y2t.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

namespace
{

struct CacheHeader
{
	char     magic[8];      // "Y2MTPC01"
	uint32_t version;       // 1
	uint32_t n_pos;
	uint32_t S;
	uint32_t N;
	uint32_t n_embd;
	uint32_t n_layers;
	int32_t  layers[8];
	uint32_t n_lab;
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

} // namespace

void Cache::open(const std::string & d, const std::string & split, int max_songs)
{
	dir = d;
	std::ifstream f(d + "/index.json");
	if (!f)
	{
		die("cannot read %s/index.json", d.c_str());
	}
	json idx = json::parse(f);
	for (const auto & e : idx)
	{
		if (e.value("split", std::string()) != split)
		{
			continue;
		}
		if (max_songs > 0 && (int) songs.size() >= max_songs)
		{
			break;
		}
		CacheSong s;
		s.name  = e.value("song", std::string()) + "/" + e.value("take", std::string());
		s.file  = e["file"].get<std::string>();
		s.split = split;
		const std::string path = d + "/" + split + "/" + s.file;
		s.fd = ::open(path.c_str(), O_RDONLY);
		if (s.fd < 0)
		{
			die("cannot open %s", path.c_str());
		}
		struct stat st;
		fstat(s.fd, &st);
		s.map_size = st.st_size;
		void * p = mmap(nullptr, s.map_size, PROT_READ, MAP_SHARED, s.fd, 0);
		if (p == MAP_FAILED)
		{
			die("cannot mmap %s", path.c_str());
		}
		s.map = (uint8_t *) p;
		CacheHeader h;
		memcpy(&h, s.map, sizeof(h));
		if (memcmp(h.magic, "Y2MTPC01", 8) != 0 || h.version != 1)
		{
			die("%s: not a Y2MTPC01 v1 cache file", path.c_str());
		}
		if (h.file_size != s.map_size)
		{
			die("%s: size %zu, header says %llu", path.c_str(), s.map_size, (unsigned long long) h.file_size);
		}
		s.n_pos    = h.n_pos;
		s.S        = h.S;
		s.n_layers = h.n_layers;
		s.n_embd   = h.n_embd;
		s.n_lab    = h.n_lab;
		if (s.n_lab != N_LAB)
		{
			die("%s: n_lab %d, expected %d", path.c_str(), s.n_lab, N_LAB);
		}
		std::vector<int> ly(h.layers, h.layers + h.n_layers);
		if (layers.empty())
		{
			layers = ly;
		} else if (layers != ly) {
			die("%s: layers differ from the other cache files", path.c_str());
		}
		s.tok    = (const int32_t *) (s.map + h.off_tok);
		s.lab_id = (const int32_t *) (s.map + h.off_lab_id);
		s.lab_p  = (const float *) (s.map + h.off_lab_p);
		s.hid    = (const uint16_t *) (s.map + h.off_hid);
		s.feat   = (const uint16_t *) (s.map + h.off_feat);
		madvise(s.map, s.map_size, MADV_SEQUENTIAL);
		songs.push_back(s);
	}
}

void Cache::close()
{
	for (auto & s : songs)
	{
		if (s.map)
		{
			munmap(s.map, s.map_size);
		}
		if (s.fd >= 0)
		{
			::close(s.fd);
		}
	}
	songs.clear();
}

int64_t Cache::total_pos() const
{
	int64_t t = 0;
	for (const auto & s : songs)
	{
		t += s.n_pos;
	}
	return t;
}

// Origins of a song are cache rows 0 .. n_pos-2 (the last row has no depth-0 label).
std::vector<ChunkRef> make_chunks(const Cache & c, int n, int min_rows, bool random_offset, std::mt19937_64 & rng)
{
	std::vector<ChunkRef> out;
	for (int si = 0; si < (int) c.songs.size(); si++)
	{
		const int n_orig = c.songs[si].n_pos - 1;
		const int off = random_offset ? (int) (rng() % (uint64_t) n) : 0;
		if (off >= min_rows && off < n_orig)
		{
			out.push_back({si, 0, off});
		}
		for (int s = off; s < n_orig; s += n)
		{
			const int r = std::min(n, n_orig - s);
			if (r >= min_rows)
			{
				out.push_back({si, s, r});
			}
		}
	}
	return out;
}

void build_batch(const Cache & c, const ChunkRef & cr, const Hparams & hp, int n, int K, bool raw_labels, Batch & b)
{
	const CacheSong & s = c.songs[cr.song];
	const int V = hp.n_vocab, L = N_LAB, F = hp.n_feat, E = hp.n_embd;
	if (s.n_layers * s.n_embd != F || s.n_embd != E)
	{
		die("cache feature size %dx%d does not match the model (%d)", s.n_layers, s.n_embd, F);
	}
	b.n = n;
	b.K = K;
	b.L = L;
	b.tag = s.name + ":" + std::to_string(cr.start);
	b.feat.assign((size_t) n * F, 0.0f);
	b.tok.assign((size_t) K * n, 0);
	b.pos.assign((size_t) K * n, 0);
	b.rmask.assign((size_t) K * n, 0.0f);
	b.w.assign(K, 0.0f);
	b.lab_idx.assign((size_t) K * n * L, 0);
	b.lab_val.assign((size_t) K * n * L, 0.0f);
	b.top_idx.assign((size_t) K * n, 0);
	b.sp_id.assign((size_t) K * n * L, -1);
	b.sp_p.assign((size_t) K * n * L, 0.0f);
	b.n_real.assign(K, 0);
	if (raw_labels)
	{
		b.hid.assign((size_t) K * n * E, 0.0f);
	}

	for (int i = 0; i < cr.n_real; i++)
	{
		const int P = cr.start + i;
		ggml_fp16_to_fp32_row((const ggml_fp16_t *) s.feat + (size_t) P * F, b.feat.data() + (size_t) i * F, F);
	}
	for (int k = 0; k < K; k++)
	{
		for (int i = 0; i < n; i++)
		{
			const size_t r = (size_t) k * n + i;
			b.lab_idx[r * L] = i * V;   // default for pad rows: (row, id 0) with value 0
			for (int j = 1; j < L; j++)
			{
				b.lab_idx[r * L + j] = i * V;
			}
			b.top_idx[r] = i * V;
			const int P = cr.start + i;
			b.pos[r] = (s.S - 1) + P + k;
			if (i >= cr.n_real || P + 1 + k >= s.n_pos)
			{
				continue;
			}
			const int32_t t = s.tok[P + k];
			if (t < D2T_BASE || t >= D2T_BASE + V)
			{
				die("%s row %d: token %d outside the draft vocab", s.name.c_str(), P + k, t);
			}
			b.tok[r] = t - D2T_BASE;
			b.rmask[r] = 1.0f;
			b.n_real[k]++;

			const int32_t * id = s.lab_id + (size_t) (P + 1 + k) * L;
			const float   * p  = s.lab_p + (size_t) (P + 1 + k) * L;
			double sum = 0;
			int m = 0;
			for (; m < L && id[m] >= 0; m++)
			{
				sum += p[m];
			}
			if (m == 0 || sum <= 0)
			{
				die("%s row %d: empty label", s.name.c_str(), P + 1 + k);
			}
			for (int j = 0; j < L; j++)
			{
				const int jj = j < m ? j : 0;   // pad slots repeat entry 0 (same index, same value)
				const int32_t d = id[jj] - D2T_BASE;
				if (d < 0 || d >= V)
				{
					die("%s row %d: label id %d outside the draft vocab", s.name.c_str(), P + 1 + k, id[jj]);
				}
				const float pv = (float) (p[jj] / sum);
				b.lab_idx[r * L + j] = i * V + d;
				b.lab_val[r * L + j] = pv;
				if (j < m)
				{
					b.sp_id[r * L + j] = d;
					b.sp_p[r * L + j] = pv;
				}
			}
			b.top_idx[r] = i * V + (id[0] - D2T_BASE);
			if (raw_labels)
			{
				ggml_fp16_to_fp32_row((const ggml_fp16_t *) s.hid + (size_t) (P + 1 + k) * E, b.hid.data() + r * E, E);
			}
		}
		b.w[k] = b.n_real[k] > 0 ? (float) n / (float) (b.n_real[k] * K) : 0.0f;
	}
}

// ------------------------------------------------------------ synthetic ---

void write_synthetic_cache(const std::string & dir, const Hparams & hp, int n_train, int n_heldout,
                           int min_len, int max_len, uint64_t seed)
{
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	const int E = hp.n_embd, nl = (int) hp.layers.size();
	json idx = json::array();
	mkdir(dir.c_str(), 0755);
	for (int which = 0; which < 2; which++)
	{
		const std::string split = which == 0 ? "train" : "heldout";
		mkdir((dir + "/" + split).c_str(), 0755);
		const int ns = which == 0 ? n_train : n_heldout;
		for (int si = 0; si < ns; si++)
		{
			const int n_pos = min_len + (int) (rng() % (uint64_t) (max_len - min_len + 1));
			const int S = 500;
			const CacheHeader h = make_header(n_pos, S, S + n_pos, E, hp.layers);
			std::vector<int32_t> tok(n_pos), lab_id((size_t) n_pos * N_LAB, -1);
			std::vector<float> pmax(n_pos, 0.5f), lab_p((size_t) n_pos * N_LAB, 0.0f);
			std::vector<ggml_fp16_t> hid((size_t) n_pos * E), feat((size_t) n_pos * nl * E);
			for (int P = 0; P < n_pos; P++)
			{
				// label at P: a few ids, the recorded token tok[P] is among them
				const int m = 1 + (int) (rng() % 20);
				std::vector<int32_t> ids;
				while ((int) ids.size() < m)
				{
					const int32_t d = (int32_t) (rng() % (uint64_t) hp.n_vocab);
					if (std::find(ids.begin(), ids.end(), d) == ids.end())
					{
						ids.push_back(d);
					}
				}
				std::vector<double> w(m);
				double z = 0;
				for (int j = 0; j < m; j++)
				{
					w[j] = exp(2.0 * nd(rng));
					z += w[j];
				}
				std::vector<int> ord(m);
				for (int j = 0; j < m; j++)
				{
					ord[j] = j;
				}
				std::sort(ord.begin(), ord.end(), [&](int a, int b2) { return w[a] > w[b2]; });
				for (int j = 0; j < m; j++)
				{
					lab_id[(size_t) P * N_LAB + j] = D2T_BASE + ids[ord[j]];
					lab_p[(size_t) P * N_LAB + j] = (float) (w[ord[j]] / z);
				}
				tok[P] = D2T_BASE + ids[ord[0]];
				for (int e = 0; e < E; e++)
				{
					hid[(size_t) P * E + e] = ggml_fp32_to_fp16(nd(rng));
				}
				for (int e = 0; e < nl * E; e++)
				{
					feat[(size_t) P * nl * E + e] = ggml_fp32_to_fp16(nd(rng) * (e < E ? 0.5f : 2.0f));
				}
			}
			char fname[64];
			snprintf(fname, sizeof(fname), "synth%03d__take.bin", si);
			const std::string path = dir + "/" + split + "/" + fname;
			FILE * f = fopen(path.c_str(), "wb");
			if (f == nullptr)
			{
				die("cannot write %s", path.c_str());
			}
			auto put = [&](uint64_t at, const void * p, size_t nb)
			{
				if (fseeko(f, (off_t) at, SEEK_SET) != 0 || fwrite(p, 1, nb, f) != nb)
				{
					die("write %s failed", path.c_str());
				}
			};
			put(0, &h, sizeof(h));
			put(h.off_tok, tok.data(), tok.size() * 4);
			put(h.off_pmax_raw, pmax.data(), pmax.size() * 4);
			put(h.off_lab_id, lab_id.data(), lab_id.size() * 4);
			put(h.off_lab_p, lab_p.data(), lab_p.size() * 4);
			put(h.off_hid, hid.data(), hid.size() * 2);
			put(h.off_feat, feat.data(), feat.size() * 2);
			fclose(f);
			idx.push_back({{"song", "synth" + std::to_string(si)}, {"take", "take"}, {"split", split},
			               {"file", fname}, {"n_pos", n_pos}, {"S", S}, {"N", S + n_pos}, {"layers", hp.layers}});
		}
	}
	std::ofstream(dir + "/index.json") << idx.dump(1);
}
