"""
================================================================================
  ORACLE SUBMISSION BUILDER — Pipeline Testing Tool
================================================================================
  Purpose:
    Build a submission.parquet that mirrors R_train as predictions + weights.
    Use this to:
      ✅ Validate your submission format is correct
      ✅ Verify your scorer / metrics pipeline works
      ✅ Establish the theoretical upper bound on train performance
      ✅ Debug evaluation code before running expensive model training

  What it does:
    • Train rows : pred_* = actual return, weight_* = sign(return) / n_assets
    • Test  rows : pred_* = 0,             weight_* = 0   (no labels available)

  ⚠️  This is a LOCAL DEBUG TOOL only.
      It will score perfectly on train and ~0 on test (no R_test available).
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path


def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "SECTION": "\n" + "═"*60 + "\n  "}
    print(f"{icons.get(level,'')} {msg}")


def sharpe(returns, eps=1e-8):
    if len(returns) < 2: return 0.0
    return float(np.mean(returns) / (np.std(returns, ddof=1) + eps) * np.sqrt(252))


def compute_metrics(returns):
    if len(returns) < 2:
        return dict(sharpe=0., cumulative_return=0., volatility=0., max_drawdown=0.)
    cum  = float(np.sum(returns))
    vol  = float(np.std(returns, ddof=1) * np.sqrt(252))
    sh   = sharpe(returns)
    curve = np.cumsum(returns)
    dd   = float(np.max(np.maximum.accumulate(curve) - curve))
    return dict(sharpe=sh, cumulative_return=cum, volatility=vol, max_drawdown=dd)


# ─────────────────────────────────────────────────────────────────────────────

def build_oracle_submission(
    data_dir   : str   = "data",
    output_path: str   = "submission.parquet",
    weight_mode: str   = "proportional",   # 'proportional' | 'equal_sign' | 'normalized'
    verbose    : bool  = True,
) -> pd.DataFrame:
    """
    Build an oracle submission from R_train for local pipeline testing.

    Parameters
    ----------
    data_dir     : directory containing X_train.parquet, R_train.parquet, X_test.parquet
    output_path  : where to write submission.parquet
    weight_mode  :
        'proportional'  → weight_i = return_i / sum(|return_j|)   (L1-normalised)
        'equal_sign'    → weight_i = sign(return_i) / n_assets     (equal allocation per direction)
        'normalized'    → weight_i = return_i / max(|return_i|)    (scale to [-1, 1])
    verbose      : print detailed logs

    Returns
    -------
    submission DataFrame
    """
    data_dir = Path(data_dir)
    log("ORACLE SUBMISSION BUILDER", "SECTION")
    log("Loading data...")

    x_train = pd.read_parquet(data_dir / "X_train.parquet")
    r_train = pd.read_parquet(data_dir / "R_train.parquet")
    x_test  = pd.read_parquet(data_dir / "X_test.parquet")

    asset_cols  = [c for c in r_train.columns if c not in ("id", "date", "split")]
    pred_cols   = [f"pred_{c}"   for c in asset_cols]
    weight_cols = [f"weight_{c}" for c in asset_cols]
    n_assets    = len(asset_cols)

    log(f"Train rows : {len(x_train)}", "OK")
    log(f"Test  rows : {len(x_test)}",  "OK")
    log(f"Assets     : {n_assets} → {asset_cols}", "OK")
    log(f"Weight mode: {weight_mode}", "OK")

    # ── Sort everything by date
    x_train = x_train.sort_values("date").reset_index(drop=True)
    r_train = r_train.sort_values("date").reset_index(drop=True)
    x_test  = x_test.sort_values("date").reset_index(drop=True)

    # ── Align x_train and r_train on common ids
    common_ids  = set(x_train["id"]).intersection(r_train["id"])
    x_tr_aligned = x_train[x_train["id"].isin(common_ids)].sort_values("date").reset_index(drop=True)
    r_tr_aligned = r_train[r_train["id"].isin(common_ids)].sort_values("date").reset_index(drop=True)

    n_train = len(x_tr_aligned)
    log(f"Aligned train rows: {n_train}", "OK")

    # ── Build TRAIN portion of submission
    R_vals = r_tr_aligned[asset_cols].values.astype(float)   # (n_train, n_assets)

    # Predictions = actual returns (oracle)
    pred_train = pd.DataFrame(R_vals, columns=pred_cols)

    # Weights from actual returns
    W_train = np.zeros_like(R_vals)
    for t in range(n_train):
        r = R_vals[t]
        if weight_mode == "proportional":
            # w_i = r_i / sum(|r_j|)   → L1 = 1 always (unless all zero)
            denom = np.abs(r).sum()
            W_train[t] = r / (denom + 1e-10)

        elif weight_mode == "equal_sign":
            # +1/K for positive, -1/K for negative
            W_train[t] = np.sign(r) / n_assets

        elif weight_mode == "normalized":
            # Scale to [-1, 1] by max absolute value, then L1-normalise
            max_abs = np.abs(r).max()
            w = r / (max_abs + 1e-10)
            # L1 normalise
            l1 = np.abs(w).sum()
            W_train[t] = w / (l1 + 1e-10) if l1 > 1.0 else w

        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

    weights_train = pd.DataFrame(W_train, columns=weight_cols)

    # ── Build TEST portion (zeros — no R_test available)
    n_test = len(x_test)
    pred_test   = pd.DataFrame(np.zeros((n_test, n_assets)), columns=pred_cols)
    weights_test = pd.DataFrame(np.zeros((n_test, n_assets)), columns=weight_cols)

    # ── Assemble full submission
    index_train = x_tr_aligned[["id", "date"]].reset_index(drop=True)
    index_test  = x_test[["id", "date"]].reset_index(drop=True)

    train_block = pd.concat([index_train, pred_train, weights_train], axis=1)
    test_block  = pd.concat([index_test,  pred_test,  weights_test],  axis=1)
    submission  = pd.concat([train_block, test_block], axis=0).reset_index(drop=True)

    # ── Sanity checks
    log("VALIDATION", "SECTION")
    assert not submission.isnull().any().any(), "❌ NaN values found"
    assert not submission["id"].duplicated().any(), "❌ Duplicate IDs found"

    l1_max = submission[weight_cols].abs().sum(axis=1).max()
    log(f"L1 constraint  : max={l1_max:.6f}  (must be ≤ 1.0) {'✅' if l1_max <= 1.0+1e-6 else '❌'}", "OK")
    log(f"Weight range   : [{submission[weight_cols].values.min():.4f}, {submission[weight_cols].values.max():.4f}]", "OK")
    log(f"Pred range     : [{submission[pred_cols].values.min():.6f}, {submission[pred_cols].values.max():.6f}]", "OK")

    # ── Performance on train (theoretical max)
    if verbose:
        log("ORACLE PERFORMANCE ON TRAIN", "SECTION")
        W = train_block[weight_cols].values
        R = r_tr_aligned[asset_cols].values
        port_ret = (W * R).sum(axis=1)
        m = compute_metrics(port_ret)
        log(f"  Sharpe            = {m['sharpe']:.4f}  ← theoretical max for this weight_mode")
        log(f"  Cumulative Return = {m['cumulative_return']:.4f}")
        log(f"  Volatility (ann)  = {m['volatility']:.4f}")
        log(f"  Max Drawdown      = {m['max_drawdown']:.4f}")

        log("\n  Per-asset oracle Sharpe:")
        for a, w in zip(asset_cols, weight_cols):
            pnl_a = train_block[w].values * r_tr_aligned[a].values
            log(f"    {a:12s}: Sharpe={sharpe(pnl_a):.3f}")

    # ── Save
    out = Path(output_path)
    submission.to_parquet(out, index=False)
    log(f"\nSaved → {out}", "OK")
    log(f"  Shape   : {submission.shape}")
    log(f"  Columns : {list(submission.columns[:6])} ... ({len(submission.columns)} total)")
    log(f"  Train   : {n_train} rows | Test: {n_test} rows")

    return submission


# ─────────────────────────────────────────────────────────────────────────────

def compare_weight_modes(data_dir="data"):
    """
    Run all 3 weight modes and compare oracle train performance.
    Useful to understand the ceiling before model training.
    """
    log("COMPARING ALL WEIGHT MODES", "SECTION")

    data_dir = Path(data_dir)
    r_train  = pd.read_parquet(data_dir / "R_train.parquet")
    asset_cols = [c for c in r_train.columns if c not in ("id","date","split")]
    R = r_train[asset_cols].values.astype(float)
    n_assets = len(asset_cols)

    results = {}
    for mode in ["proportional", "equal_sign", "normalized"]:
        W = np.zeros_like(R)
        for t in range(len(R)):
            r = R[t]
            if mode == "proportional":
                W[t] = r / (np.abs(r).sum() + 1e-10)
            elif mode == "equal_sign":
                W[t] = np.sign(r) / n_assets
            elif mode == "normalized":
                w = r / (np.abs(r).max() + 1e-10)
                l1 = np.abs(w).sum()
                W[t] = w / l1 if l1 > 1.0 else w

        port_ret = (W * R).sum(axis=1)
        m = compute_metrics(port_ret)
        results[mode] = m
        log(f"\n  [{mode:14s}]  "
            f"Sharpe={m['sharpe']:.3f}  "
            f"CumRet={m['cumulative_return']:.4f}  "
            f"MaxDD={m['max_drawdown']:.4f}", "OK")

    return results


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oracle submission builder for pipeline testing")
    parser.add_argument("--data_dir",    default="data",               help="Data directory")
    parser.add_argument("--output",      default="submission.parquet", help="Output path")
    parser.add_argument("--mode",        default="proportional",
                        choices=["proportional","equal_sign","normalized"],
                        help="Weight construction mode")
    parser.add_argument("--compare",     action="store_true",
                        help="Compare all weight modes before building")
    args = parser.parse_args()

    if args.compare:
        compare_weight_modes(args.data_dir)

    build_oracle_submission(
        data_dir    = args.data_dir,
        output_path = args.output,
        weight_mode = args.mode,
        verbose     = True,
    )
