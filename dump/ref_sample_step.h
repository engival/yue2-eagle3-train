// yue2.cpp's own sample_step, compiled from a verbatim cut of
// $YUE2_CPP_DIR/src/stage_ar.cpp (see CMakeLists.txt), exposed for the semantic
// phase so shaped::dist can be checked against it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Runs sample_step (semantic phase, unguided, cot != off) on one full-vocab logit
// row and returns the distribution it drew from: ids and probabilities in its
// own order. Also returns the token it drew with `seed`.
int32_t ref_semantic_dist(const float * logits, int n_vocab, const std::vector<int32_t> & history,
	int step, uint64_t seed, std::vector<int32_t> & ids, std::vector<double> & probs);
