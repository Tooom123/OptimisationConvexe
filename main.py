"""
main.py — Exécute tous les modèles non-vides et évalue leurs soumissions.

Pour chaque fichier model_*.py non vide :
  - importe sa classe PortfolioChallengeModel
  - entraîne et génère prk/{nom}.parquet
Puis appelle evaluate.py sur tous les parquets générés.

Les fichiers vides sont ignorés et signalés.
"""

import importlib.util
import importlib
import traceback
from pathlib import Path

DATA_DIR   = "data"
PRK_DIR    = Path("prk")
MODELS_DIR = Path("models")
PRK_DIR.mkdir(exist_ok=True)

MODEL_FILES = [
    "model_Tom",
    "model_Aymeric",
    "model_Artus",
    "model_Estebane",
]


def load_model_class(module_name: str):
    path = MODELS_DIR / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"models/{module_name}.py introuvable")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PortfolioChallengeModel"):
        raise AttributeError(f"PortfolioChallengeModel absent de models/{module_name}.py")
    return mod.PortfolioChallengeModel


def run_model(name: str) -> Path | None:
    out_path = PRK_DIR / f"{name}.parquet"

    print(f"\n{'='*60}")
    print(f"  Modèle : {name}")
    print(f"{'='*60}")

    try:
        ModelClass = load_model_class(name)
    except (FileNotFoundError, AttributeError) as e:
        print(f"  [IGNORÉ] {e}")
        return None

    try:
        model = ModelClass()
        model.fit_predict_save(data_dir=DATA_DIR, output_path=str(out_path))
        print(f"  Sauvegardé : {out_path}")
        return out_path
    except Exception:
        print(f"  [ERREUR] {name} a échoué :")
        traceback.print_exc()
        return None


def main():
    skipped   = []
    generated = []

    for name in MODEL_FILES:
        result = run_model(name)
        if result is None:
            skipped.append(name)
        else:
            generated.append(result)

    print(f"\n{'='*60}")
    if skipped:
        print(f"  Ignorés (vides/absents) : {', '.join(skipped)}")
    print(f"  {len(generated)}/{len(MODEL_FILES)} modèle(s) exécuté(s)")
    print(f"{'='*60}\n")

    if not generated:
        print("Aucun parquet généré, évaluation annulée.")
        return

    import evaluate
    importlib.reload(evaluate)

    r_train, _ = evaluate.load_data()
    rows = [evaluate.evaluate_file(p, r_train) for p in sorted(generated)]
    evaluate.print_table(rows)


if __name__ == "__main__":
    main()
