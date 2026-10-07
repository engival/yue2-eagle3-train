#!/usr/bin/env python3
# Reader / verifier for the yue2mtp-dump cache files (README "Cache format").
#
#	tools/read_cache.py data/cache/train/<song>__<take>.bin [...]   # verify files
#	from read_cache import load; c = load(path)                     # numpy memmaps

import struct
import sys

import numpy as np

HDR = struct.Struct("<8sIIIIII8iII7Q")


def load(path):
	with open(path, "rb") as f:
		raw = f.read(HDR.size)
	v = HDR.unpack(raw)
	magic, version, n_pos, S, N, n_embd, n_layers = v[:7]
	layers = list(v[7:7 + n_layers])
	n_lab = v[15]
	off_tok, off_pmax, off_lab_id, off_lab_p, off_hid, off_feat, size = v[17:24]
	if magic != b"Y2MTPC01" or version != 1:
		raise ValueError(f"{path}: bad magic/version {magic} {version}")
	m = np.memmap(path, mode="r", dtype=np.uint8)
	if len(m) != size:
		raise ValueError(f"{path}: size {len(m)} != header {size}")

	def arr(off, dt, shape):
		n = int(np.prod(shape)) * np.dtype(dt).itemsize
		return m[off:off + n].view(dt).reshape(shape)

	return {
		"n_pos": n_pos, "S": S, "N": N, "n_embd": n_embd, "layers": layers, "n_lab": n_lab,
		"tok": arr(off_tok, "<i4", (n_pos,)),
		"pmax_raw": arr(off_pmax, "<f4", (n_pos,)),
		"lab_id": arr(off_lab_id, "<i4", (n_pos, n_lab)),
		"lab_p": arr(off_lab_p, "<f4", (n_pos, n_lab)),
		"hid": arr(off_hid, "<f2", (n_pos, n_embd)),
		"feat": arr(off_feat, "<f2", (n_pos, n_layers, n_embd)),
	}


def verify(path):
	c = load(path)
	tok, li, lp = c["tok"], c["lab_id"], c["lab_p"]
	ok = True
	s = lp.sum(1)
	if np.abs(s - 1).max() > 1e-5:
		print(f"  label sums off: max |1-sum| {np.abs(s - 1).max():.2e}")
		ok = False
	if (li[:, 0] < 0).any():
		print("  empty label row")
		ok = False
	pad = li < 0
	if (lp[pad] != 0).any():
		print("  nonzero p on padding")
		ok = False
	if not ((tok >= 151852) & (tok < 151853 + 32768)).all() or tok[-1] != 151852:
		print("  tok outside the semantic set or last token != MUSIC_END")
		ok = False
	insup = (li == tok[:, None]).any(1).mean()
	f = np.asarray(c["feat"][:: max(1, c["n_pos"] // 64)], dtype=np.float32)
	h = np.asarray(c["hid"][:: max(1, c["n_pos"] // 64)], dtype=np.float32)
	if not (np.isfinite(f).all() and np.isfinite(h).all()):
		print("  non-finite features")
		ok = False
	rms = np.sqrt((f ** 2).mean(axis=(0, 2)))
	print(f"{path}: n_pos {c['n_pos']} S {c['S']} N {c['N']} layers {c['layers']} support {insup:.4f} "
	      f"feat rms {np.round(rms, 2).tolist()} hid rms {np.sqrt((h ** 2).mean()):.2f} -> {'OK' if ok else 'FAIL'}")
	return ok


if __name__ == "__main__":
	bad = sum(0 if verify(p) else 1 for p in sys.argv[1:])
	sys.exit(1 if bad else 0)
