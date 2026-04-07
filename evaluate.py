"""
evaluate.py — Évaluateur local de soumissions parquet.

Usage :
    python evaluate.py              # évalue tous les .parquet dans prk/
    python evaluate.py prk/foo.parquet prk/bar.parquet   # fichiers spécifiques

Métriques calculées sur la période TRAIN (seule période où on a les vrais retours) :
  - Sharpe annualisé (données horaires → annualisation sqrt(8760))
  - Retour cumulatif
  - MSE des prédictions vs retours réels
  - Gross exposure moyen (sum|w|)

Le classement final est trié par Sharpe train décroissant.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data")
PRK_DIR    = Path("prk")
ANNUALIZE  = np.sqrt(8760)  # données horaires

ASSET_COLS  = [f"asset_{i}"  for i in range(1, 13)]
PRED_COLS   = [f"pred_{c}"   for c in ASSET_COLS]
WEIGHT_COLS = [f"weight_{c}" for c in ASSET_COLS]


def load_data():
    r_train = pd.read_parquet(DATA_DIR / "R_train.parquet")
    x_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
    return r_train, x_test


def validate(sub: pd.DataFrame, path: str) -> list[str]:
    errors = []
    required = ["id", "date"] + PRED_COLS + WEIGHT_COLS
    missing = [c for c in required if c not in sub.columns]
    if missing:
        errors.append(f"colonnes manquantes : {missing}")
    if sub["id"].duplicated().any():
        errors.append("ids dupliqués")
    if sub[required].isnull().any().any():
        errors.append("valeurs manquantes (NaN)")
    w = sub[WEIGHT_COLS].values
    if np.any(np.abs(w) > 1 + 1e-6):
        errors.append("poids hors [-1, 1]")
    gross = np.abs(w).sum(axis=1)
    if np.any(gross > 1 + 1e-6):
        errors.append(f"gross exposure > 1 sur {(gross > 1 + 1e-6).sum()} lignes")
    return errors


def score(sub: pd.DataFrame, r_train: pd.DataFrame) -> dict:
    r = r_train[["id", "date"] + ASSET_COLS].copy()
    s = sub[["id", "date"] + PRED_COLS + WEIGHT_COLS].copy()
    r["date"] = pd.to_datetime(r["date"])
    s["date"] = pd.to_datetime(s["date"])
    merged = r.merge(s, on=["id", "date"], how="inner")
    if merged.empty:
        return {"error": "aucune ligne commune avec R_train"}

    R = merged[ASSET_COLS].to_numpy(dtype=float)
    W = merged[WEIGHT_COLS].to_numpy(dtype=float)
    P = merged[PRED_COLS].to_numpy(dtype=float)

    pnl = (W * R).sum(axis=1)

    sharpe   = pnl.mean() / (pnl.std() + 1e-12) * ANNUALIZE
    cum_ret  = float(np.prod(1 + pnl) - 1)
    mse      = float(np.mean((R - P) ** 2))
    gross    = float(np.abs(W).sum(axis=1).mean())
    n_rows   = int(len(merged))

    return {
        "sharpe_train":  round(float(sharpe), 4),
        "cum_ret_train": round(cum_ret * 100, 2),
        "mse_pred":      round(mse, 8),
        "gross_exp":     round(gross, 4),
        "n_rows":        n_rows,
    }


def evaluate_file(path: Path, r_train: pd.DataFrame) -> dict:
    result = {"file": path.name}
    try:
        sub = pd.read_parquet(path)
    except Exception as e:
        result["error"] = f"lecture impossible : {e}"
        return result

    errors = validate(sub, str(path))
    if errors:
        result["warnings"] = errors

    metrics = score(sub, r_train)
    result.update(metrics)
    return result


def print_table(rows: list[dict]):
    sortable = [r for r in rows if "sharpe_train" in r]
    broken   = [r for r in rows if "sharpe_train" not in r]

    sortable.sort(key=lambda r: r["sharpe_train"], reverse=True)

    header = f"{'Rang':<5} {'Fichier':<30} {'Sharpe':>8} {'CumRet%':>9} {'MSE':>12} {'Gross':>7} {'N':>7}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for rank, r in enumerate(sortable, 1):
        warn = " !" if r.get("warnings") else ""
        print(f"{rank:<5} {r['file'][:30]:<30} {r['sharpe_train']:>8.4f} "
              f"{r['cum_ret_train']:>8.2f}% {r['mse_pred']:>12.2e} "
              f"{r['gross_exp']:>7.4f} {r['n_rows']:>7}{warn}")
    if broken:
        print("\nFichiers en erreur :")
        for r in broken:
            print(f"  {r['file']} → {r.get('error', r.get('warnings', '?'))}")
    print(sep)
    if sortable:
        best = sortable[0]
        print(f"\nMeilleur : {best['file']}  "
              f"Sharpe={best['sharpe_train']}  "
              f"CumRet={best['cum_ret_train']}%")
    print()


def main():
    r_train, _ = load_data()

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(PRK_DIR.glob("*.parquet"))

    if not paths:
        print(f"Aucun fichier .parquet trouvé dans {PRK_DIR}/")
        return

    print(f"Évaluation de {len(paths)} fichier(s)…")
    rows = [evaluate_file(p, r_train) for p in paths]
    print_table(rows)


if __name__ == "__main__":
    main()
