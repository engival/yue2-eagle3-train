# yue2-eagle3-train

Trains an **EAGLE-3 draft head** for the semantic phase of
[YuE2](https://huggingface.co/m-a-p/YuE2-3B)'s AR stage, so that
[yue2.cpp](https://github.com/engival/yue2.cpp) can decode the codec tokens
speculatively. Everything runs on ggml (`ggml-opt` for the optimiser) through
the Vulkan backend; no PyTorch, no CUDA.

- Trained head: [`engival/YuE2-3B-EAGLE3-draft-GGUF`](https://huggingface.co/engival/YuE2-3B-EAGLE3-draft-GGUF)
  (CC BY-NC 4.0, like the YuE2 weights it derives from).
- Inference side: yue2.cpp branch
  [`feat/eagle3-draft`](https://github.com/engival/yue2.cpp/tree/feat/eagle3-draft)
  (`yue2 song --draft HEAD.gguf`), an experiment that is not on yue2.cpp's master.

**Provenance and warranty.** This code was written by Claude (Anthropic) working
as a coding agent, directed by a human who set the goals, ran the trainings and
listened to the results, but did not closely review the source. The trainer is
checked against a double-precision reference with finite-difference gradients,
CPU against Vulkan, and through a GGUF round trip into llama.cpp (see
[Validation](#validation)), but treat it as unreviewed software: use at your own
risk.

## Results

The released head (one EAGLE-3 layer, 4-token unroll) on held-out songs, with p =
YuE2's sampler-shaped target distribution and q = the draft's:

| draft depth | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| acceptance E[Σ min(p, q)] | 0.486 | 0.384 | 0.316 | 0.261 |

Expected accepted tokens per round for K = 1..4 drafted tokens: 0.486 / 0.673 /
0.732 / 0.747.

In yue2.cpp (`feat/eagle3-draft`) that makes the semantic phase about **1.1×**
faster in exact mode (rejection sampling: output distributed exactly as the plain
sampler's) and about **1.35×** with `--draft-lambda 3` (accepts more freely
inside the sampler's support, no audible loss in our listening), which is
**~5–10 % of a whole song**. The ceiling is the codec tokens themselves: YuE2
samples them at temperature 1.0 with top-k 100 / top-p 0.95, the shaped
distribution's top-1 probability averages ~0.26, and even a perfect greedy draft
would get ≤ 1.36 tokens per target step. n-gram / lookup drafting is useless
here (codec n-grams almost never repeat).

## Pipeline

1. **Render songs with yue2.cpp** (`yue2 song` / `yue2 batch`). Each render dir
   holds `request.json`, `config.json`, `result.json`, `plan.json`,
   `prefix.npy`, `semantic.npy`, which is all the dumper needs.
2. **Select** the renders that are plain, unguided semantic decodes and split
   them into train / held-out by song name (`tools/list_songs.py`).
3. **Dump features** (`yue2mtp-dump`): teacher-force each recorded song through
   the **Q8_0 AR GGUF** (`yue2-ar-q8_0.gguf` from `yue2 convert`) and store, per
   semantic position, the target's layer inputs at layers 2 / 14 / 25, the final
   hidden state, the recorded token and the sampler-shaped label distribution.
4. **Train** (`yue2mtp-train`) on that cache; it writes an `eagle3` GGUF that
   loads in llama.cpp as a draft model against the AR GGUF.

The head is tied to the exact target it was dumped from (the Q8_0 AR GGUF; the
cache index records its SHA-256).

## Build

Needs CMake ≥ 3.16, a C++17 compiler, and for the Vulkan backend a Vulkan driver
plus `glslc` (same as llama.cpp). The dumper also needs a **yue2.cpp source
checkout**: its sampler port is unit-tested against yue2.cpp's own
`sample_step`, cut out of `src/stage_ar.cpp` at configure time. By default it is
looked for next to this repo (`../yue2.cpp`).

```bash
git clone --recursive https://github.com/engival/yue2-eagle3-train.git
cd yue2-eagle3-train
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
      -DYUE2_CPP_DIR=/path/to/yue2.cpp
cmake --build build -j8
```

`-DYUE2_EAGLE3_BUILD_DUMP=OFF` builds the trainer without a yue2.cpp checkout;
`-DGGML_VULKAN=OFF` gives a CPU-only build (enough for the gradient check and
`--backend cpu`, far too slow for real training). Binaries land in
`build/bin/`: `yue2mtp-dump`, `shaped-dist-test`, `yue2mtp-train`,
`yue2mtp-check`, `yue2mtp-roundtrip`. The Python helpers need `numpy`.

**Tested on** one box: Linux (Slackware64-current), one AMD Radeon RX 7900 XTX
(RADV) through ggml's Vulkan backend. Nothing else has been tried.

## Usage

Paths below are placeholders. Run every GPU step on one device
(`GGML_VK_VISIBLE_DEVICES=0` or similar): the trainer always uses Vulkan device 0.

### Selecting songs

```bash
tools/list_songs.py --songs /path/to/songs --out lists
```

Scans `--songs` recursively for `semantic.npy`. The directory right below
`--songs` is the **song name**; everything under it (takes, remixes) is one
song. A render is kept only if it is a plain semantic decode of the Q8_0 target:
`config.json` `backend == "yue2.cpp"` and `ar_gguf` named `yue2-ar-q8_0.gguf`,
`cfg_scale` unset or 1.0, `cot` not `off`, no `guidance` / `sections` /
`handover` / `negative_*` / `semantic_keep` in `plan.json`, `status ==
complete`, not truncated, and `len(semantic) + 1 == output_tokens`. About 10 % of
the semantic tokens, as whole song names, go to held-out. Writes
`songs_train.txt`, `songs_heldout.txt`, `songs_all.txt` and `exclusions.json`.

### Dumping features

```bash
build/bin/shaped-dist-test        # sampler port vs yue2.cpp's sample_step, CPU only

GGML_VK_VISIBLE_DEVICES=0 build/bin/yue2mtp-dump \
    -m /path/to/yue2-ar-q8_0.gguf --root /path/to/songs \
    --songs lists/songs_train.txt:train --songs lists/songs_heldout.txt:heldout \
    --out data/cache [--report data/cache/report.md]
```

- `--songs FILE[:SPLIT]` is repeatable; SPLIT (default `train`) is the cache
  subdir. `--root` is the same directory given to `list_songs.py`; it names the
  cache files `<song>__<take>.bin`.
- Resumable: songs already in `index.json` with the right file size are
  skipped (`--no-resume` redoes them).
- `--check-sampler N` (default 97): every N-th row also runs yue2.cpp's
  `sample_step` and must match `shaped::dist` exactly. `--check-logits N`
  recomputes N rows' logits from the stored hidden state.
- `--measure-only` writes no cache, only the per-position statistics;
  `--report FILE` turns them into a markdown report (`tools/m0_report.py`):
  shaped p_max / entropy / support, the perfect-greedy ceiling and the n-gram
  baseline. Run it first to see what a draft can gain on your songs.
- Songs where < 99 % of the recorded tokens fall inside the reconstructed
  sampler support are rejected (expect ~99.87 %: prefill-vs-decode numerics flip
  the odd near-boundary token).
- Size: 17,192 bytes per semantic position (~100 MB for a 3.4-minute song);
  our 3.3 M positions took ~56 GB. Throughput on the 7900 XTX was ~2800
  tokens/s overall.

`tools/read_cache.py FILE.bin ...` verifies cache files (and `load()` gives
numpy memmaps).

### Training

```bash
GGML_VK_VISIBLE_DEVICES=0 build/bin/yue2mtp-train \
    --target /path/to/yue2-ar-q8_0.gguf --cache data/cache --out runs/first \
    --epochs 20 --lr 1e-4 --warmup 200 --unroll 4 --chunk 2048 --accum 8 \
    --init struct --eval-every 162 --eval-chunks 0 --save-every 324 --log-every 10
```

That is the recipe of the released head's first run (~5.75 h on the 7900 XTX for
~2.1 M train positions). It was then warm-started on a larger cache (3.26 M
positions, same held-out songs):

```bash
... yue2mtp-train --target ... --cache data/cache --out runs/second \
    --resume runs/first/draft.gguf --epochs 7 --lr 5e-5 --warmup 100 \
    --unroll 4 --chunk 2048 --accum 8 --eval-every 125 --eval-chunks 0 \
    --save-every 250 --log-every 10
```

About **9 GPU-hours** in total on one RX 7900 XTX, F32 precision, ~10–12 GB of
VRAM. Outputs in `--out`: `train.log` (appended), `draft-stepNNNNNN.gguf`
checkpoints, final `draft.gguf` (all F32).

- Every `--eval-every` optimiser steps the held-out split is evaluated per
  depth: CE, H(p), KL(p‖q), top-1 agreement, greedy acceptance E[p(argmax q)],
  rejection-sampling acceptance E[Σ min(p, q)] (the number that matters), and
  expected accepted length for K = 1..4. `--eval-chunks 0` = all held-out
  chunks. `--eval-only --resume CKPT` evaluates a checkpoint.
- `--resume` loads weights only: fresh AdamW moments, the lr schedule restarts
  with its warmup.
- `--precision f16` keeps ggml-vulkan's fp16 / coopmat matmuls and adds loss
  scaling (default 1024): ~1.8× faster, gradients within ~7e-3 relative of F32.
  The default `f32` is the validated path.
- lr: linear warmup, cosine to 10 %; AdamW β 0.9 / 0.95, no weight decay. The
  global gradient norm is logged every step (needs `--accum` ≥ 2); a spike
  above 10× the recent median halves the lr multiplier for the rest of the run.
- `--labels raw` trains on the target's unshaped softmax instead (ablation
  only); `--init random` starts everything but the output head from noise.

## Model

The draft is llama.cpp's `eagle3` graph (`llama.cpp/src/models/eagle3.cpp`),
one decoder layer with the target's sizes: n_embd 2048, 16 query / 8 KV heads of
128, FFN 6144, RMS eps 1e-6, RoPE NORM mode with θ 1e6, no q/k norm.

- Step 0: `g = fc · concat(feat_2, feat_14, feat_25)`; the input row is
  `concat(rms_norm(embed(tok)) * attn_norm, rms_norm(g) * attn_norm_2)`; the
  residual is the raw `g`; the layer's pre-norm output feeds the next depth.
- The vocabulary is only YuE2's semantic block: 32769 rows, `d2t[i] = 151852 +
  i` (MUSIC_END, then the 32768 codec tokens). The token embeddings are the
  target's, read frozen and never written to the GGUF.
- `--init struct` (default): output head and `output_norm` from the target's
  matching `output.weight` rows; FFN and `attn_norm_2` from the target's last
  block; `fc = [0 | 0 | I]` so step 0 starts from the layer-25 input;
  attention random. It beats random init clearly in the early steps.
- The GGUF copies every key of the target (tokenizer included) and sets the
  `eagle3.*` hyperparameters, so it loads unmodified in llama.cpp as a draft
  against the AR GGUF.

## Objective

Rows are teacher-forced along the recorded path. For cache row P the depth-0
input is `(tok[P], feat[P])` at position P and its label is the target's shaped
distribution at P+1; depth k feeds `(tok[P+k], h_{k-1})` at position P+k and is
labelled with the distribution at P+1+k. The **training-time test** unrolls 4
depths in one graph: a depth-k query attends to the depth-0 rows up to its
origin plus its own origin's depth-1..k rows, exactly the KV the draft has at
inference. Gradients flow through `h_{k-1}` into the deeper depths.

Labels are YuE2's **sampler-shaped** semantic distribution, reproduced exactly
(`tools/shaped_dist.*`): allowed ids MUSIC_END + codec block, MUSIC_END masked
for the first 200 tokens, repetition penalty 1.2^count over the last 50 tokens,
top-k 100 (ties kept), softmax, top-p 0.95. ≤ 100 nonzeros per row, stored
sparse and scattered on the GPU. The loss is the cross entropy against them,
averaged over the real rows of each depth and over the depths.

Chunks are `--chunk` consecutive rows of one song (random offset per epoch,
shuffled; shorter than 256 rows dropped), one chunk per graph, gradients
accumulated over `--accum` chunks. Padding rows are masked out before the head.

## Cache format

`<out>/<split>/<song>__<take>.bin`, little-endian: a 256-byte header, then six
arrays at the offsets it records (each 4096-aligned). Row r is semantic-phase
position P = S−1+r (row 0 = MUSIC_START, whose features predict the first codec
token).

| header field | type | |
|---|---|---|
| magic | char[8] | `Y2MTPC01` |
| version | u32 | 1 |
| n_pos | u32 | N − S |
| S | u32 | length of `prefix.npy` = index of the first semantic token |
| N | u32 | length of prefix ++ (semantic + 151853) ++ [MUSIC_END 151852] |
| n_embd | u32 | 2048 |
| n_layers | u32 | 3 |
| layers | i32[8] | first n_layers used (layer-input index: residual entering that layer) |
| n_lab | u32 | 100 |
| reserved | u32 | 0 |
| off_tok, off_pmax_raw, off_lab_id, off_lab_p, off_hid, off_feat | u64 × 6 | byte offsets |
| file_size | u64 | total bytes |

| array | dtype | shape | content |
|---|---|---|---|
| tok | int32 | [n_pos] | token sampled at P (absolute id; MUSIC_END on the last row) |
| pmax_raw | f32 | [n_pos] | raw full-vocab softmax top-1 probability at P |
| lab_id | int32 | [n_pos][100] | shaped support at P, sorted by p desc (id asc on ties), −1 padded |
| lab_p | f32 | [n_pos][100] | matching probabilities (sum 1), 0 padded |
| hid | f16 | [n_pos][2048] | final normed hidden state at P |
| feat | f16 | [n_pos][3][2048] | layer inputs at P, in `layers` order |

`<out>/index.json` lists every file with song, take, source dir, split,
n_pos, S, N, the target GGUF's SHA-256, layers and support rate.

## Validation

`yue2mtp-check` holds the trainer's checks:

- `gradcheck` (CPU, a tiny config on a synthetic cache with padded chunks):
  the graph's forward against an independent double-precision reference of the
  model and the unrolled objective (per-depth CE rel. error ≤ 6e-7), and the
  analytic gradients of all 13 trainable tensors against central differences
  of that reference (rel. error ≤ 2e-4). Covers both concats and every depth.
- `cpuvk --target AR.gguf --cache DIR`: loss and per-tensor gradients, CPU
  backend vs Vulkan, at the real size on a real chunk. F32 mode: loss
  bit-equal, worst gradient 5e-5 relative. Also lists any op of the
  forward+backward graph Vulkan cannot run (none).
- `synth DIR`: writes a synthetic cache in the dumper's format.

`yue2mtp-roundtrip --draft CKPT.gguf --target AR.gguf --cache DIR` loads a
checkpoint as an eagle3 draft in libllama, feeds cache rows the way llama.cpp's
`draft-eagle3` does, and compares the draft logits with the trainer's own
forward: max |Δ| 5.7e-6 on CPU with an F32 KV cache; the argmax agrees in every
mode. The written GGUF also loads in stock `llama-speculative-simple
--spec-type draft-eagle3`.

```bash
build/bin/yue2mtp-check gradcheck        # CPU only, seconds
```

## ggml patches (llama.cpp fork branch)

The `llama.cpp` submodule is pinned to branch
[`feat/eagle3-train`](https://github.com/engival/llama.cpp/tree/feat/eagle3-train)
of the fork `engival/llama.cpp`: two commits on top of the fork's `dev`.

1. **`GGML_OP_CONCAT` backward** (`839c9971e`). ggml had none, and the draft
   uses concat twice (the embedding ‖ hidden input row, and the K/V of the
   unrolled depths). The gradient of each source is the matching slice of the
   output gradient, made contiguous first (it arrives as a permuted view
   through the K/V concat). `test-backend-ops` grad mode now covers concat.
2. **F32 `MUL_MAT` backward via `mul_mat` of transposed copies**
   (`0e352ffe6`). The stock backward uses `OUT_PROD` with a transposed
   gradient, which ggml-vulkan rejects (non-contiguous), so every activation
   gradient fell back to the CPU, and Vulkan's `OUT_PROD` is a naive kernel
   anyway. For F32 operands it now computes `ds0 = mul_mat(cont(s1ᵀ),
   cont(dtᵀ))` and `ds1 = mul_mat(cont(s0ᵀ), dt)`; quantized / F16 `src0` (LoRA
   finetuning) keeps the `OUT_PROD` path. Correct, but it changes the backward
   for every backend, which is why it lives on a fork branch rather than in an
   upstream PR as is.

## Vulkan workarounds

Two ggml-vulkan behaviours are worked around with environment variables that
`yue2mtp-train` sets itself (unless already set):

- **fp16 coopmat backward underflow.** ggml-vulkan stages F32 matmuls through
  fp16 (cooperative matrix). The logit gradients are ~(softmax − p)/n ≈ 1e-7,
  below fp16's normal range, and 20–75 % of the weight gradients came out wrong.
  `--precision f32` (default) sets `GGML_VK_DISABLE_F16=1` and
  `GGML_VK_DISABLE_COOPMAT=1`; `--precision f16` keeps coopmat with loss
  scaling instead.
- **`graph_optimize` corrupts the eval graph.** With ggml-vulkan's node
  reordering on, the held-out evaluation graph produced NaN log-sum-exp on
  some rows and wrong argmaxes (exact again with it off; the training graph was
  unaffected). The trainer and `yue2mtp-check` set
  `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1`; no throughput cost measured.
  To reproduce, remove that `setenv` from `train/train.cpp` and run
  `yue2mtp-train --eval-only --resume CKPT ...`.

## License

Code: MIT (LICENSE). YuE2 weights and everything trained from them, including
the heads this trainer writes: CC BY-NC 4.0, non-commercial. See NOTICE.md.
