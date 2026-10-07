#include "ref_sample_step.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

typedef int32_t llama_token;

static const int EOD          = 151643;
static const int ABC_END      = 151848;
static const int MUSIC_END    = 151852;
static const int CODEC_OFFSET = 151853;
static const int CODEC_SIZE   = 32768;

struct Sampling
{
	double temperature;
	double top_p;
	int    top_k;
	double repetition_penalty;
	int    penalty_window;
	int    min_tokens;
	int    max_tokens;
};

[[noreturn]] static void die(const char * fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	exit(1);
}

#include "y2_sample_step.inc"

}

int32_t ref_semantic_dist(const float * logits, int n_vocab, const std::vector<int32_t> & history,
	int step, uint64_t seed, std::vector<int32_t> & ids, std::vector<double> & probs)
{
	const Sampling  s = {1.0, 0.95, 100, 1.2, 50, 200, 9000};  // sampling_semantic()
	std::mt19937_64 rng(seed);
	SampleScratch   sc;
	const llama_token t = sample_step(logits, n_vocab, s, history, step, false, false, rng, &sc, false);
	ids.resize(sc.cand.size());
	probs.assign(sc.prob.begin(), sc.prob.end());
	for (size_t i = 0; i < sc.cand.size(); i++)
	{
		ids[i] = sc.cand[i].second;
	}
	(void) EOD;
	(void) ABC_END;
	(void) MUSIC_END;
	(void) CODEC_OFFSET;
	(void) CODEC_SIZE;
	return t;
}
