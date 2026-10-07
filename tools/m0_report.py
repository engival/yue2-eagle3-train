#!/usr/bin/env python3
# Semantic-phase ceiling report ("M0") from the dumper's per-position records.
#
#	tools/m0_report.py OUT_DIR [--measure]   > report.md
#
# Reads OUT_DIR/m0_positions[.measure].bin (M0Rec, 44 bytes, see yue2mtp-dump.cpp),
# m0_songs[.measure].json and summary[.measure].json.

import json
import os
import sys

import numpy as np

REC = np.dtype([
	("song", "<u4"), ("step", "<i4"), ("n_pos", "<i4"), ("flags", "<u4"),
	("pmax", "<f4"), ("ent", "<f4"), ("p_rec", "<f4"),
	("pmax_raw", "<f4"), ("ent_raw", "<f4"), ("out_mass", "<f4"), ("supp", "<i4"),
])
F_IN_SUPPORT, F_ARGMAX, F_NG_HIT, F_NG_HAVE, F_RAW_ARGMAX, F_SUPP_CUT = 1, 2, 4, 32, 256, 512
KMAX = 6


def scheme_a(pmax, starts_ok):
	"""E[accepted] per round for a perfect greedy draft, K=1..KMAX: sum_k prod_{j<k} pmax[t+j]."""
	out = []
	n = len(pmax)
	prod = np.ones(n)
	acc = np.zeros(n)
	for k in range(KMAX):
		sh = np.zeros(n)
		sh[: n - k] = pmax[k:]
		prod = prod * sh
		acc = acc + prod
		out.append(acc.copy())
	return out


def block(r, title):
	if len(r) == 0:
		return f"| {title} | 0 | | | | | | | | |\n"
	f = r["flags"]
	ng = [((f & (F_NG_HIT << i)) != 0).mean() for i in range(3)]
	return (f"| {title} | {len(r):,} | {r['pmax'].mean():.3f} | {r['ent'].mean():.2f} | {r['supp'].mean():.1f} | "
	        f"{((f & F_ARGMAX) != 0).mean():.3f} | {r['pmax_raw'].mean():.3f} | {r['ent_raw'].mean():.2f} | "
	        f"{ng[0]:.3f} / {ng[1]:.3f} / {ng[2]:.3f} | {r['ea'][:, 3].mean():.2f} |\n")


def hist(x, edges):
	h, _ = np.histogram(x, bins=edges)
	h = h / max(len(x), 1)
	return " | ".join(f"{v * 100:.1f}" for v in h)


def main():
	out = sys.argv[1]
	sfx = ".measure" if "--measure" in sys.argv else ""
	recs = np.fromfile(os.path.join(out, f"m0_positions{sfx}.bin"), dtype=REC)
	songs = json.load(open(os.path.join(out, f"m0_songs{sfx}.json")))
	summ = json.load(open(os.path.join(out, f"summary{sfx}.json")))

	# scheme A per song (products never cross a song boundary)
	ea = np.zeros((len(recs), KMAX))
	start = 0
	while start < len(recs):
		end = start + int(recs["n_pos"][start])
		parts = scheme_a(recs["pmax"][start:end].astype(np.float64), None)
		for k in range(KMAX):
			ea[start:end, k] = parts[k]
		start = end
	full = np.zeros(len(recs), dtype=bool)   # rounds that fit inside the song for K=KMAX
	full[:] = recs["step"] + KMAX <= recs["n_pos"]

	r = np.zeros(len(recs), dtype=REC.descr + [("ea", "<f8", (KMAX,))])
	for name in REC.names:
		r[name] = recs[name]
	r["ea"] = ea
	f = recs["flags"]

	print("## M0 — semantic-phase ceiling\n")
	print(f"{len(songs)} songs, {len(recs):,} semantic positions (P = S−1..N−2). "
	      f"Shaped = the sampler distribution (penalty 1.2/50 → top-k 100 → top-p 0.95); "
	      f"raw = softmax over the full 184704 vocab.\n")

	print("### Headline\n")
	print("| metric | value |\n|---|---|")
	print(f"| mean shaped p_max | {recs['pmax'].mean():.4f} |")
	print(f"| median shaped p_max | {np.median(recs['pmax']):.4f} |")
	print(f"| mean shaped entropy (nats) | {recs['ent'].mean():.3f} |")
	print(f"| mean shaped support size | {recs['supp'].mean():.1f} |")
	print(f"| recorded token = shaped argmax | {((f & F_ARGMAX) != 0).mean():.4f} |")
	print(f"| mean shaped p(recorded token) | {recs['p_rec'].mean():.4f} |")
	print(f"| mean raw p_max (full vocab) | {recs['pmax_raw'].mean():.4f} |")
	print(f"| mean raw entropy (nats) | {recs['ent_raw'].mean():.3f} |")
	print(f"| recorded token = raw argmax | {((f & F_RAW_ARGMAX) != 0).mean():.4f} |")
	print(f"| mean raw mass outside {{MUSIC_END}} ∪ codec | {recs['out_mass'].mean():.2e} |")
	print(f"| positions with shaped support > 100 | {int(((f & F_SUPP_CUT) != 0).sum())} |")
	print()

	print("### Scheme A ceiling (perfect greedy draft, sample-and-compare)\n")
	print("E[accepted] = Σ_{k≤K} Π_{j<k} p_max(t+j) over the recorded path, averaged over every t whose "
	      f"K={KMAX} window fits in the song ({int(full.sum()):,} rounds). Tokens per verify = 1 + E.\n")
	print("| K | " + " | ".join(str(k + 1) for k in range(KMAX)) + " |")
	print("|---|" + "---|" * KMAX)
	print("| E[accepted] | " + " | ".join(f"{ea[full, k].mean():.3f}" for k in range(KMAX)) + " |")
	print("| tokens/verify | " + " | ".join(f"{1 + ea[full, k].mean():.3f}" for k in range(KMAX)) + " |")
	print()

	print("### n-gram baseline (song's own semantic history, most recent earlier occurrence)\n")
	print("| n | has earlier occurrence | hit rate (all positions) | precision (when matched) |")
	print("|---|---|---|---|")
	for i, n in enumerate((2, 3, 4)):
		have = (f & (F_NG_HAVE << i)) != 0
		hit = (f & (F_NG_HIT << i)) != 0
		print(f"| {n} | {have.mean():.4f} | {hit.mean():.4f} | {hit[have].mean() if have.any() else 0:.4f} |")
	print()

	print("### Splits\n")
	print("| subset | positions | mean p_max | entropy | support | argmax=rec | raw p_max | raw ent | n-gram hit 2/3/4 | E[acc] K=4 |")
	print("|---|---|---|---|---|---|---|---|---|---|")
	frac = recs["step"] / np.maximum(recs["n_pos"], 1)
	sl = (recs["step"] + KMAX <= recs["n_pos"])
	print(block(r[sl], "all").rstrip())
	print(block(r[sl & (frac < 0.1)], "first 10 % of song").rstrip())
	print(block(r[sl & (frac >= 0.1)], "rest").rstrip())
	print(block(r[sl & (recs["step"] < 200)], "step < 200").rstrip())
	print(block(r[sl & (recs["step"] >= 200)], "step ≥ 200").rstrip())
	splits = sorted(set(s["split"] for s in songs))
	if len(splits) > 1:
		sidx = np.array([s["split"] for s in songs])[recs["song"]]
		for sp in splits:
			print(block(r[sl & (sidx == sp)], f"split {sp}").rstrip())
	print()

	print("### Histograms (% of positions)\n")
	e = np.linspace(0, 1, 11)
	print("| p_max bin | " + " | ".join(f"{e[i]:.1f}–{e[i + 1]:.1f}" for i in range(10)) + " |")
	print("|---|" + "---|" * 10)
	print("| shaped | " + hist(recs["pmax"], e) + " |")
	print("| raw | " + hist(recs["pmax_raw"], e) + " |")
	ee = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 20]
	print()
	print("| entropy bin (nats) | " + " | ".join(f"{ee[i]}–{ee[i + 1]}" for i in range(10)) + " |")
	print("|---|" + "---|" * 10)
	print("| shaped | " + hist(recs["ent"], ee) + " |")
	print("| raw | " + hist(recs["ent_raw"], ee) + " |")
	se = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 1e9]
	print()
	print("| support bin | " + " | ".join(f"{se[i]}–{int(se[i + 1]) - 1 if se[i + 1] < 1e8 else '∞'}" for i in range(10)) + " |")
	print("|---|" + "---|" * 10)
	print("| shaped | " + hist(recs["supp"], se) + " |")
	print()

	print("### Checks\n")
	print(f"- check 1 (recorded token in shaped support): {summ['check1_support_rate']:.5f} overall, "
	      f"min per song {summ['check1_min_song_rate']:.4f}, rejected (< 99 %): {len(summ['check1_rejected'])}")
	print(f"- sample_step equivalence on real rows: {summ['sampler_rows'] - summ['sampler_fail']}/{summ['sampler_rows']} "
	      f"identical support and order, max |Δp| {summ['sampler_max_dp']:.2e}")
	print(f"- check 2 (hid · output.weight vs llama logits, {summ['check2_rows']} rows × 32769): max |Δ| "
	      f"f32 hid {summ['check2_maxdiff_f32']:.3e}, fp16 hid {summ['check2_maxdiff_f16']:.3e}, "
	      f"q8-rounded hid (emulating Vulkan's q8_1 activation) {summ['check2_maxdiff_q8act']:.3e}; "
	      f"fp16 storage alone (f16 vs f32 hid, same CPU dot) {summ['check2_f16_vs_f32_hid']:.3e} "
	      f"(max |logit| {summ['check2_max_abs_logit']:.1f})")
	print(f"- check 3: mean RMS per feature layer {['%.2f' % x for x in summ['check3_mean_rms_per_layer']]} "
	      f"(layers {summ['layers']}); stale batches {summ['check3_stale_batches']}, "
	      f"identical feature rows at two positions {summ['check3_same_feat_positions']}")
	print(f"- throughput: {summ['tokens_decoded']:,} tokens decoded in {summ['wall_s']:.0f} s wall "
	      f"({summ['overall_tok_per_s']:.0f} tok/s overall, {summ['decode_tok_per_s']:.0f} tok/s in llama_decode; "
	      f"cpu {summ['cpu_s']:.0f} s, write {summ['write_s']:.0f} s); {summ['positions_per_s']:.0f} semantic positions/s")
	if summ.get("bytes_written"):
		print(f"- cache written: {summ['bytes_written'] / 1e9:.2f} GB")


if __name__ == "__main__":
	main()
