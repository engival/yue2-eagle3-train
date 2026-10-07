# Third-party notices

yue2-eagle3-train is MIT licensed (see LICENSE). It ships **no model weights**
and no training data.

## Code lineage

- **llama.cpp** (`llama.cpp/` submodule, which also carries **ggml**) —
  Copyright (c) 2023-2026 The ggml authors and the llama.cpp contributors,
  MIT. The trainer is written on `ggml` / `ggml-opt`; the dumper and the
  round-trip check link `libllama` (and `llama-common` for the latter). The
  submodule points at the fork branch `engival/llama.cpp` `feat/eagle3-train`,
  which adds two ggml backward passes (see README); those commits are under
  llama.cpp's MIT license.
- **nlohmann/json** (`llama.cpp/vendor/nlohmann`) — Copyright (c) 2013-2025
  Niels Lohmann, MIT. Cache index and dumper reports.
- **SHA-256** (`llama.cpp/vendor/hash/sha256`, Igor Pavlov, public domain) —
  the dumper records the target GGUF's hash in the cache index.
- **yue2.cpp** (github.com/engival/yue2.cpp) — Copyright (c) 2026 Richard
  Lavigne, MIT. `dump/CMakeLists.txt` cuts `sample_step` out of a yue2.cpp
  checkout's `src/stage_ar.cpp` at configure time and compiles it as the
  reference for the sampler unit test; nothing of it is copied into this
  repository. `tools/shaped_dist.cpp` is an independent port of the same
  semantic-phase sampling rule.
- **EAGLE-3** (Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large
  Language Models via Training-Time Test", 2025) — the draft architecture and
  the training-time-test unroll follow the paper and llama.cpp's `eagle3`
  inference graph (`llama.cpp/src/models/eagle3.cpp`); no code from the
  reference EAGLE / SpecForge implementations is included.

All of the above are MIT (the SHA-256 public domain); their full license texts
are included with the respective upstream projects (`llama.cpp/LICENSE`).

## Model weights (NOT included, NOT covered by this license)

- **m-a-p/YuE2-3B** — CC BY-NC 4.0. Non-commercial. The dumper reads a GGUF of
  its AR half (made with `yue2 convert`), and the trainer initialises the draft
  head from it (output rows, the last block's FFN, the token embeddings it
  reads frozen).
- Anything trained with this code from YuE2 features — including every draft
  head it writes, such as `engival/YuE2-3B-EAGLE3-draft-GGUF` — is a
  derivative of those weights and is **CC BY-NC 4.0**, non-commercial. The MIT
  license here covers the code only.
