#!/usr/bin/env python3
# Pick the YuE2 renders the dumper may teacher-force (README "Selecting songs").
#
# Scans SONGS recursively for semantic.npy, classifies each artifact dir by its
# own fields (never by path or lyrics), writes songs_all.txt / songs_train.txt /
# songs_heldout.txt / exclusions.json into OUT. Held-out is chosen by song name
# (SONGS/<name>/...), so every take and remix of one song is on the same side.
#
#	tools/list_songs.py --songs SONGS [--out lists] [--heldout-frac 0.10]

import argparse
import hashlib
import json
import os
import sys

import numpy as np

MUSIC_START = 151851
ABC_END     = 151848
TARGET_GGUF = "yue2-ar-q8_0.gguf"
PLAN_KEYS   = ("guidance", "sections", "handover", "negative_style", "negative_lyrics", "semantic_keep")
BELT_FILES  = ("negative_prefix.npy", "guidance.json", "sections.json", "handover.json")
NEEDED      = ("request.json", "config.json", "result.json", "plan.json", "prefix.npy", "semantic.npy")


def load_json(path):
	with open(path) as f:
		return json.load(f)


def classify(d):
	"""Returns (reason or None, n_semantic)."""
	for f in NEEDED:
		if not os.path.exists(os.path.join(d, f)):
			return "missing_" + f, 0
	cfg = load_json(os.path.join(d, "config.json"))
	req = load_json(os.path.join(d, "request.json"))
	res = load_json(os.path.join(d, "result.json"))
	plan = load_json(os.path.join(d, "plan.json"))
	if cfg.get("backend") != "yue2.cpp":
		return "backend_not_yue2cpp", 0
	if os.path.basename(str(cfg.get("ar_gguf"))) != TARGET_GGUF:
		return "ar_gguf_not_q8_0", 0
	if req.get("cot") == "off":
		return "cot_off", 0
	if cfg.get("cfg_scale") not in (None, 1.0):
		return "cfg_scale_guided", 0
	for k in PLAN_KEYS:
		if k in plan:
			return "plan_" + k, 0
	trunc = res.get("truncated", {})
	if res.get("status") != "complete":
		return "status_not_complete", 0
	if trunc.get("semantic", True) is not False:
		return "truncated_semantic", 0
	for f in BELT_FILES:
		if os.path.exists(os.path.join(d, f)):
			return "file_" + f, 0
	sem = np.load(os.path.join(d, "semantic.npy"))
	pre = np.load(os.path.join(d, "prefix.npy"))
	if len(pre) < 2 or int(pre[-2]) != ABC_END or int(pre[-1]) != MUSIC_START:
		return "prefix_tail_not_abcend_musicstart", 0
	if len(sem) + 1 != res["timing"]["semantic"]["output_tokens"]:
		return "semantic_len_mismatch", 0
	return None, len(sem)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--songs", required=True, help="root dir of yue2.cpp render dirs, one subdir per song name")
	ap.add_argument("--out", default="lists")
	ap.add_argument("--heldout-frac", type=float, default=0.10)
	a = ap.parse_args()

	songs = os.path.abspath(a.songs)
	dirs = []
	for root, _, files in os.walk(songs):
		if "semantic.npy" in files:
			dirs.append(root)
	dirs.sort()

	reasons = {}
	keep = []
	for d in dirs:
		why, n = classify(d)
		if why is None:
			keep.append((d, n))
		else:
			reasons[why] = reasons.get(why, 0) + 1

	# held-out: whole song names, in sha1(name) order, added while the held-out
	# token share stays <= frac (a name that would overshoot is skipped, so the two
	# giant songs never land there by accident of the hash).
	by_name = {}
	for d, n in keep:
		name = os.path.relpath(d, songs).split(os.sep)[0]
		by_name.setdefault(name, []).append((d, n))
	total = sum(n for _, n in keep)
	held = set()
	acc = 0
	for name in sorted(by_name, key=lambda s: hashlib.sha1(s.encode()).hexdigest()):
		t = sum(n for _, n in by_name[name])
		if acc + t <= a.heldout_frac * total:
			held.add(name)
			acc += t

	os.makedirs(a.out, exist_ok=True)
	train = [d for name in sorted(by_name) if name not in held for d, _ in by_name[name]]
	hold = [d for name in sorted(by_name) if name in held for d, _ in by_name[name]]
	for fn, lst in (("songs_all.txt", [d for d, _ in keep]), ("songs_train.txt", train), ("songs_heldout.txt", hold)):
		with open(os.path.join(a.out, fn), "w") as f:
			f.write("".join(x + "\n" for x in lst))

	summary = {
		"scanned": len(dirs),
		"kept": len(keep),
		"kept_semantic_tokens": total,
		"excluded": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
		"per_song": {name: {"takes": len(v), "tokens": sum(n for _, n in v), "split": "heldout" if name in held else "train"}
		             for name, v in sorted(by_name.items())},
		"train_takes": len(train),
		"heldout_takes": len(hold),
		"heldout_tokens": acc,
		"heldout_names": sorted(held),
	}
	with open(os.path.join(a.out, "exclusions.json"), "w") as f:
		json.dump(summary, f, indent=1)
	json.dump({k: v for k, v in summary.items() if k != "per_song"}, sys.stdout, indent=1)
	print()


if __name__ == "__main__":
	main()
