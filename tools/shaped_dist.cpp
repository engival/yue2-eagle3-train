#include "shaped_dist.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace shaped
{

static bool by_score_then_id(const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b)
{
	return a.first != b.first ? a.first > b.first : a.second < b.second;
}

int dist(const float * logits, const int32_t * history, size_t n_hist, int step,
	int32_t * out_id, double * out_p, int cap, Scratch & sc)
{
	const float ninf = -std::numeric_limits<float>::infinity();

	// penalty over the last PENALTY_WIN history ids: alpha = 1.2^freq (float, as
	// sample_step computes it), v < 0 ? v*alpha : v/alpha
	sc.pen.clear();
	{
		const size_t from = n_hist > (size_t) PENALTY_WIN ? n_hist - PENALTY_WIN : 0;
		std::vector<int32_t> ids(history + from, history + n_hist);
		std::sort(ids.begin(), ids.end());
		for (size_t i = 0; i < ids.size();)
		{
			size_t j = i;
			while (j < ids.size() && ids[j] == ids[i])
			{
				j++;
			}
			const float alpha = (float) std::pow(PENALTY, (double) (j - i));
			const float v     = logits[ids[i]];
			sc.pen.push_back(std::make_pair(ids[i], v < 0 ? v * alpha : v / alpha));
			i = j;
		}
	}

	// allowed ids ascending: MUSIC_END (unless step < MIN_TOKENS), then the codec block
	sc.all.clear();
	size_t pi = 0;
	auto visit = [&](int32_t i)
	{
		while (pi < sc.pen.size() && sc.pen[pi].first < i)
		{
			pi++;
		}
		const float v = pi < sc.pen.size() && sc.pen[pi].first == i ? sc.pen[pi].second : logits[i];
		if (v > ninf)
		{
			sc.all.push_back(std::make_pair(v, i));
		}
	};
	if (step >= MIN_TOKENS)
	{
		visit(MUSIC_END);
	}
	for (int32_t i = CODEC_OFFSET; i < CODEC_OFFSET + CODEC_SIZE; i++)
	{
		visit(i);
	}
	if (sc.all.empty())
	{
		fprintf(stderr, "shaped::dist: every token masked (step %d)\n", step);
		abort();
	}

	// top-k: keep every value >= the k-th largest (ties included)
	sc.cand.clear();
	if (sc.all.size() <= (size_t) TOP_K)
	{
		sc.cand = sc.all;
	} else {
		std::vector<float> v(sc.all.size());
		for (size_t i = 0; i < sc.all.size(); i++)
		{
			v[i] = sc.all[i].first;
		}
		std::nth_element(v.begin(), v.begin() + (TOP_K - 1), v.end(), std::greater<float>());
		const float thr = v[TOP_K - 1];
		for (const auto & e : sc.all)
		{
			if (!(e.first < thr))
			{
				sc.cand.push_back(e);
			}
		}
	}
	std::sort(sc.cand.begin(), sc.cand.end(), by_score_then_id);

	// softmax (double) over the survivors
	sc.prob.assign(sc.cand.size(), 0.0);
	const double top = sc.cand[0].first;
	double       sum = 0;
	for (size_t i = 0; i < sc.cand.size(); i++)
	{
		sc.prob[i] = std::exp((double) sc.cand[i].first - top);
		sum       += sc.prob[i];
	}
	for (auto & p : sc.prob)
	{
		p /= sum;
	}

	// top-p, head always kept
	size_t n   = sc.cand.size();
	double cum = 0;
	for (size_t i = 0; i < sc.cand.size(); i++)
	{
		cum += sc.prob[i];
		if (i >= 1 && cum - sc.prob[i] > TOP_P)
		{
			n = i;
			break;
		}
	}
	sum = 0;
	for (size_t i = 0; i < n; i++)
	{
		sum += sc.prob[i];
	}
	for (size_t i = 0; i < n && (int) i < cap; i++)
	{
		out_id[i] = sc.cand[i].second;
		out_p[i]  = sc.prob[i] / sum;
	}
	return (int) n;
}

}
