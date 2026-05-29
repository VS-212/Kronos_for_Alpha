"""
Deep exploration of belief state: confidence, entropy, top3_mass, entropy_ratio.
What do they tell us about prediction quality and model certainty?
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.evaluation.metrics import direction_accuracy, ic_rank, return_correlation

DATA_DIR = "data/v3/predictions/10min_sber_mini"
RAW_DIR = "data/tickers/SBER"
LK = 500; PL = 12

preds = np.load(f"{DATA_DIR}/SBER_preds_pl12_sc5.npy")
belief = np.load(f"{DATA_DIR}/SBER_belief_pl12_sc5.npy")
raw = np.load(f"{RAW_DIR}/feats_test_raw.npy")

N = preds.shape[0]
assert belief.shape == (N, 5, PL, 4), f"belief shape mismatch: {belief.shape}"

# ── Per-window returns ──────────────────────────────────────────────────────
entry_prices = np.zeros(N)
pred_ret = np.zeros(N)
actual_ret = np.zeros(N)
for i in range(N):
    ep = raw[i + LK - 1, 3]
    entry_prices[i] = ep
    pred_ret[i] = (preds[i, :, PL - 1, 3].mean() - ep) / max(ep, 1e-8)
    ci = min(i + LK + PL - 1, len(raw) - 1)
    actual_ret[i] = (raw[ci, 3] - ep) / max(ep, 1e-8)

# ── Belief channels ─────────────────────────────────────────────────────────
# shape: (N, 5, 12, 4) → reduce over MC samples (axis=1) and/or steps (axis=2)
conf = belief[:, :, :, 0]  # (N, 5, 12)
ent = belief[:, :, :, 1]   # (N, 5, 12)
top3 = belief[:, :, :, 2]  # (N, 5, 12)
ent_r = belief[:, :, :, 3] # (N, 5, 12)

# ── 1. COVERS ALL 6428 WINDOWS ──────────────────────────────────────────────
print("=" * 70)
print("  BELIEF DISTRIBUTION (all 6428 windows × 5 MC × 12 steps)")
print("=" * 70)

labels = ["confidence", "entropy_s1", "top3_mass", "entropy_ratio"]
for idx, ch in enumerate([conf, ent, top3, ent_r]):
    d = ch.flatten()
    p5, p25, p50, p75, p95 = np.percentile(d, [5, 25, 50, 75, 95])
    print(f"  {labels[idx]:<15s}  mean={d.mean():>8.4f}  std={d.std():>8.4f}  "
          f"p5={p5:>8.4f}  p25={p25:>8.4f}  p50={p50:>8.4f}  p75={p75:>8.4f}  p95={p95:>8.4f}  "
          f"[{d.min():.4f}, {d.max():.4f}]")

# ── 2. BELIEF CORRELATION MATRIX (window-averaged) ─────────────────────────
print(f"\n{'─'*70}")
print(f"  BELIEF CHANNEL CORRELATIONS (mean over MC×steps per window)")
print(f"{'─'*70}")

b_mean = np.stack([conf.mean(axis=(1, 2)), ent.mean(axis=(1, 2)),
                    top3.mean(axis=(1, 2)), ent_r.mean(axis=(1, 2))], axis=1)
corr = np.corrcoef(b_mean.T)
print(f"  {'':>15s} {'confidence':>10s} {'entropy':>10s} {'top3_mass':>10s} {'ent_ratio':>10s}")
for i, l in enumerate(labels):
    print(f"  {l:>15s} {corr[i,0]:>10.4f} {corr[i,1]:>10.4f} {corr[i,2]:>10.4f} {corr[i,3]:>10.4f}")

# ── 3. BELIEF vs PREDICTION QUALITY ────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  CONFIDENCE → PREDICTION QUALITY (IC / dir_acc)")
print(f"{'─'*70}")
print(f"  {'threshold':>10s} {'windows':>8s} {'IC':>8s} {'dir_acc':>8s} {'mean_ret':>10s} {'std_ret':>10s}")

for th in np.arange(0.1, 0.66, 0.025):
    m = conf.mean(axis=(1, 2)) >= th
    if m.sum() < 30: continue
    print(f"  {'conf>=' + f'{th:.3f}':>10s} {m.sum():>8d} "
          f"{ic_rank(actual_ret[m], pred_ret[m]):>8.4f} "
          f"{direction_accuracy(actual_ret[m], pred_ret[m]):>8.2%} "
          f"{actual_ret[m].mean():>10.6f} {actual_ret[m].std():>10.6f}")

# ── 4. ENTROPY → PREDICTION QUALITY ───────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  ENTROPY (inverse = certainty) → PREDICTION QUALITY")
print(f"{'─'*70}")
print(f"  {'threshold':>10s} {'windows':>8s} {'IC':>8s} {'dir_acc':>8s} {'mean_ret':>10s}")
for th in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]:
    m = ent.mean(axis=(1, 2)) < th  # LOW entropy = certain
    if m.sum() < 30: continue
    print(f"  {'ent<' + str(th):>10s} {m.sum():>8d} "
          f"{ic_rank(actual_ret[m], pred_ret[m]):>8.4f} "
          f"{direction_accuracy(actual_ret[m], pred_ret[m]):>8.2%} "
          f"{actual_ret[m].mean():>10.6f}")

# ── 5. TOP3_MASS → PREDICTION QUALITY ─────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  TOP3_MASS → PREDICTION QUALITY")
print(f"{'─'*70}")
print(f"  {'threshold':>10s} {'windows':>8s} {'IC':>8s} {'dir_acc':>8s}")
for th in np.arange(0.3, 0.91, 0.05):
    m = top3.mean(axis=(1, 2)) >= th
    if m.sum() < 30: continue
    print(f"  {'top3>=' + f'{th:.2f}':>10s} {m.sum():>8d} "
          f"{ic_rank(actual_ret[m], pred_ret[m]):>8.4f} "
          f"{direction_accuracy(actual_ret[m], pred_ret[m]):>8.2%}")

# ── 6. CONFIDENCE EVOLUTION ACROSS STEPS ────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  CONFIDENCE BY PREDICTION STEP (mean over N×MC)")
print(f"{'─'*70}")
for step in range(PL):
    c = conf[:, :, step].mean()
    e = ent[:, :, step].mean()
    t = top3[:, :, step].mean()
    print(f"  step {step:>2d}:  conf={c:.4f}  ent={e:.4f}  top3={t:.4f}")

# ── 7. MC SAMPLE AGREEMENT vs CONFIDENCE ────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  MC SAMPLE AGREEMENT")
print(f"{'─'*70}")

# For each window: MC samples predict the same direction? Fraction of agreement
pred_dirs = np.sign(preds[:, :, PL-1, 3])  # (N, 5): sign of each MC sample's final close
mc_agree = np.mean(pred_dirs == np.sign(pred_ret[:, None]), axis=1)  # fraction agreeing with mean direction
conf_mean = conf.mean(axis=(1, 2))

print(f"  MC agreement distribution:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    p{p}: {np.percentile(mc_agree, p):.1%}")

print(f"\n  Confident windows (conf>0.5) have MC agreement:")
m_hc = conf_mean > 0.5
print(f"    n={m_hc.sum()}: mean MC agreement = {mc_agree[m_hc].mean():.2%}")

print(f"  Low-confidence windows (conf<0.2) have MC agreement:")
m_lc = conf_mean < 0.2
print(f"    n={m_lc.sum()}: mean MC agreement = {mc_agree[m_lc].mean():.2%}")

# ── 8. CONFIDENCE × PREDICTED RETURN → RANKED BUCKETS ──────────────────────
print(f"\n{'─'*70}")
print(f"  JOINT: confidence × predicted_return — decile buckets")
print(f"{'─'*70}")

c_dec = np.digitize(conf_mean, np.percentile(conf_mean, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
r_dec = np.digitize(pred_ret, np.percentile(pred_ret, [10, 20, 30, 40, 50, 60, 70, 80, 90]))

# Top confidence + top predicted return
m_top = (c_dec >= 9) & (r_dec >= 9)
m_bot = (c_dec >= 9) & (r_dec <= 2)
m_cntr = (r_dec >= 9) & ~(c_dec >= 9)
print(f"  High conf + High pred return:  n={m_top.sum():>5d}  actual_ret={actual_ret[m_top].mean():>.4%}  IC={ic_rank(actual_ret[m_top], pred_ret[m_top]):.4f}")
print(f"  High conf + Low pred return:   n={m_bot.sum():>5d}  actual_ret={actual_ret[m_bot].mean():>.4%}  IC={ic_rank(actual_ret[m_bot], pred_ret[m_bot]):.4f}")
print(f"  Low conf + High pred return:   n={m_cntr.sum():>5d}  actual_ret={actual_ret[m_cntr].mean():>.4%}  IC={ic_rank(actual_ret[m_cntr], pred_ret[m_cntr]):.4f}")

# ── 9. CAN CONFIDENCE PREDICT DIRECTION ACCURACY? (binary) ───────────────────
print(f"\n{'─'*70}")
print(f"  CONFIDENCE AS A PREDICTOR OF DIRECTION ACCURACY")
print(f"{'─'*70}")

correct = np.sign(pred_ret) == np.sign(actual_ret)
for th in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    m_h = conf_mean >= th
    m_l = conf_mean < th
    if m_h.sum() < 30 or m_l.sum() < 30:
        continue
    acc_h = correct[m_h].mean()
    acc_l = correct[m_l].mean()
    print(f"  conf>={th:.2f}:  acc_high={acc_h:.2%}  acc_low={acc_l:.2%}  diff={acc_h-acc_l:+.2%}")

# ── 10. VERIFY ENTROPY_RATIO BUG ────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  ENTROPY_RATIO ANALYSIS (channel 3)")
print(f"{'─'*70}")
print(f"  Range: [{ent_r.min():.1f}, {ent_r.max():.1f}]")
print(f"  Mean: {ent_r.mean():.1f}  Median: {np.median(ent_r):.1f}")
print(f"  Note: ch3 = sum(-log2(p_s2)) / entropy_s1")
print(f"  For uniform dist over 1024: sum(-log2(p)) = 10240, H = 10")
print(f"  So ratio for uniform ~ 1024. Observed mean ~ {ent_r.mean():.0f} suggests")
print(f"  distribution is somewhat peaked.")

# ── SAVE ────────────────────────────────────────────────────────────────────
os.makedirs("data/v3/results", exist_ok=True)
print(f"\nDone. Data in memory.")
