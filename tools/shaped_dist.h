// shaped_dist — YuE2's semantic-phase sampler distribution.
//
// The exact distribution `sample_step` (yue2.cpp src/stage_ar.cpp) draws from in
// the semantic phase, before the draw: allowed set {MUSIC_END} ∪ codec block →
// MUSIC_END masked below min_tokens → window penalty 1.2^freq over the last 50 →
// temperature 1 (no-op) → top-k 100 with ties → softmax (double) → top-p 0.95
// (drop from the first i >= 1 with cumsum - p_i > 0.95) → renormalise.
//
// Shared by the dumper (labels) and the trainer's eval. Header + one .cpp, no deps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace shaped
{

static const int32_t MUSIC_START   = 151851;
static const int32_t MUSIC_END     = 151852;
static const int32_t CODEC_OFFSET  = 151853;
static const int32_t CODEC_SIZE    = 32768;
static const int32_t MIN_TOKENS    = 200;
static const int32_t TOP_K         = 100;
static const double  TOP_P         = 0.95;
static const double  PENALTY       = 1.2;
static const int32_t PENALTY_WIN   = 50;

// Reusable buffers; one per thread. Never sized by the vocab.
struct Scratch
{
	std::vector<std::pair<int32_t, float>> pen;   // penalised ids, ascending by id
	std::vector<std::pair<float, int32_t>> all;   // allowed (score, id)
	std::vector<std::pair<float, int32_t>> cand;  // top-k survivors
	std::vector<double>                    prob;
};

// logits:  one full-vocab row (indexed by absolute token id, >= CODEC_OFFSET + CODEC_SIZE)
// history: the semantic ids sampled so far in this song (oldest first), absolute ids
// step:    number of semantic tokens already sampled (= history size in a plain song)
// out_id / out_p: room for `cap` entries; filled sorted by p desc (id asc on ties).
// Returns the support size (<= TOP_K plus boundary ties). Only the first `cap`
// entries are written; a return > cap means the tail was cut (probabilities still
// normalised over the whole support).
int dist(const float * logits, const int32_t * history, size_t n_hist, int step,
	int32_t * out_id, double * out_p, int cap, Scratch & sc);

}
