// Unit test: shaped::dist vs yue2.cpp's sample_step on synthetic semantic-phase
// logit rows that hit the edge cases (min_tokens mask, heavy penalties on both
// signs, top-k boundary ties, a dominant head that top-p cuts to 1, flat rows).
// The dumper repeats the same comparison on real recorded positions
// (--check-sampler).

#include "ref_sample_step.h"
#include "shaped_dist.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const int N_VOCAB = 184704;

static bool compare(const char * what, const float * row, const std::vector<int32_t> & hist, int step, int & worst_n)
{
	std::vector<int32_t> rid;
	std::vector<double>  rp;
	ref_semantic_dist(row, N_VOCAB, hist, step, 1, rid, rp);

	shaped::Scratch      sc;
	std::vector<int32_t> id(40000);
	std::vector<double>  p(40000);
	const int n = shaped::dist(row, hist.data(), hist.size(), step, id.data(), p.data(), 40000, sc);
	worst_n = std::max(worst_n, n);
	if ((size_t) n != rid.size())
	{
		printf("FAIL %s: support %d vs ref %zu\n", what, n, rid.size());
		return false;
	}
	double md = 0;
	for (int i = 0; i < n; i++)
	{
		if (id[i] != rid[i])
		{
			printf("FAIL %s: entry %d id %d vs ref %d\n", what, i, id[i], rid[i]);
			return false;
		}
		md = std::max(md, std::fabs(p[i] - rp[i]));
	}
	if (md > 1e-6)
	{
		printf("FAIL %s: max |dp| %.3g\n", what, md);
		return false;
	}
	return true;
}

int main()
{
	std::mt19937_64 rng(12345);
	std::vector<float> row(N_VOCAB);
	int    n_ok = 0, n_all = 0, worst = 0;
	char   name[128];

	for (int t = 0; t < 400; t++)
	{
		const int kind = t % 8;
		std::normal_distribution<float> g(0.0f, kind == 3 ? 0.05f : 3.0f);
		for (auto & v : row)
		{
			v = g(rng);
		}
		if (kind == 1)
		{
			// many exact ties around the top-k boundary
			for (int i = shaped::CODEC_OFFSET; i < shaped::CODEC_OFFSET + shaped::CODEC_SIZE; i += 97)
			{
				row[i] = 7.0f;
			}
		}
		if (kind == 2)
		{
			row[shaped::CODEC_OFFSET + 5] = 40.0f;   // top-p keeps only the head
		}
		if (kind == 4)
		{
			row[shaped::MUSIC_END] = 30.0f;          // end dominant; masked below 200
		}
		if (kind == 5)
		{
			// shift everything negative so the penalty multiplies
			for (auto & v : row)
			{
				v -= 20.0f;
			}
		}
		if (kind == 6)
		{
			for (auto & v : row)
			{
				v = 1.0f;                            // flat: >100 ties at top-k
			}
		}

		const int step = (t * 37) % 600;
		std::vector<int32_t> hist;
		std::uniform_int_distribution<int> pick(0, (t % 3 == 0) ? 20 : shaped::CODEC_SIZE - 1);
		for (int i = 0; i < step; i++)
		{
			hist.push_back(shaped::CODEC_OFFSET + pick(rng));
		}
		snprintf(name, sizeof(name), "case %d kind %d step %d", t, kind, step);
		n_all++;
		n_ok += compare(name, row.data(), hist, step, worst);
	}
	printf("shaped_dist_test: %d/%d cases match sample_step (max support seen %d)\n", n_ok, n_all, worst);
	return n_ok == n_all ? 0 : 1;
}
