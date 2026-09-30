# -*- coding: utf-8 -*-
"""Holm-Bonferroni correction for the McNemar tests reported in the thesis for experiments that are NOT
re-run in the revision (preliminary condition, Tables 4-5; two-stage system, Tables 7 and 9).
Exact two-sided McNemar (binomial on discordant pairs); samples missing from a result file count as wrong.
Families = the set of tests reported for one table. Run from the repo root: python results/rev_holm_existing.py"""
import os, math
from scipy.stats import binomtest
R = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def load(prefix):
    ok = {}
    for y in (2014, 2016, 2019):
        with open(os.path.join(R, f'{prefix}_{y}_results.txt'), encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 2 and p[1] in ('CORRECT', 'WRONG'):
                    ok[(y, p[0])] = p[1] == 'CORRECT'
    return ok
def mcnemar(a, b):
    keys = set(a) | set(b)
    x = sum(1 for k in keys if a.get(k, False) and not b.get(k, False))
    y = sum(1 for k in keys if b.get(k, False) and not a.get(k, False))
    p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p
def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0] * m; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
FAM = {
 'Table 4-5 (preliminary, ground-truth SRT)': [
   ('abl_dual_shared', 'abl_concat'), ('abl_dual_shared', 'abl_cascaded'), ('abl_concat', 'abl_cascaded'),
   ('abl_dual_shared', 'abl_online'), ('abl_dual_shared', 'abl_uni')],
 'Table 7 (two-stage, four recognizers)': [
   (a, b) for i, a in enumerate(['traj_abl_srtoof_thayckpt', 'traj_abl_srtoof_thayv4full', 'traj_abl_srtoof', 'traj_abl_srtoof_better'])
   for b in ['traj_abl_srtoof_thayckpt', 'traj_abl_srtoof_thayv4full', 'traj_abl_srtoof', 'traj_abl_srtoof_better'][i + 1:]],
 'Table 9 (decoder grid, predicted SRT)': [
   ('traj_abl_srtoof_thayv4_concat', 'traj_abl_offline_lrmax'), ('traj_abl_srtoof_thayv4_cascaded', 'traj_abl_offline_lrmax'),
   ('traj_abl_srtoof_thayv4full', 'traj_abl_offline_lrmax'), ('traj_abl_srtoof_thayv4_cascaded', 'traj_abl_srtoof_thayv4full'),
   ('traj_abl_srtoof_thayv4_cascaded', 'traj_abl_srtoof_thayv4_concat'), ('traj_abl_srtoof_thayv4full', 'traj_abl_srtoof_thayv4_concat'),
   ('traj_abl_srtoof_thayv4full', 'traj_abl_srtoof_thayv4_online'), ('traj_abl_srtoof_thayv4_cascaded', 'traj_abl_srtoof_thayv4_online')],
}
cache = {}
out = []
for fam, pairs in FAM.items():
    rows = []
    for a, b in pairs:
        for n in (a, b):
            if n not in cache: cache[n] = load(n)
        rows.append((a, b) + mcnemar(cache[a], cache[b]))
    adj = holm([r[4] for r in rows])
    out.append(f'== {fam}  (m = {len(rows)}) ==')
    for r, h in zip(rows, adj):
        out.append(f'{r[0]:34s} vs {r[1]:34s} b={r[2]:4d} c={r[3]:4d}  p={r[4]:.3g}  p_Holm={h:.3g}  {"*" if h < 0.05 else ""}')
    out.append('')
print('\n'.join(out))
open(os.path.join(R, 'rev_holm_existing.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
