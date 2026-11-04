#!/usr/bin/env python3
"""
Activity prediction from fingerprint features.

Just run:  python predict_activity.py
and answer the questions. Your model files should be in a folder called 'models'
one level above this script's folder; pick one from the list. Advanced users can also pass options, e.g.

  python predict_activity.py --model_path xgb.json --features_csv features.csv --output_results predictions.csv --n_jobs 4
  
Optional: --n_jobs (CPU cores; default half your cores, auto-skipped for single-threaded models like SVM), --verbose.
  
Supported models:
  .json / .ubj              XGBoost model saved with save_model()
  .joblib / .pkl / .pickle  scikit-learn model (SVM, etc.) saved with joblib

Needs: numpy, pandas, joblib, scikit-learn (and xgboost only for XGBoost models)
Install with:  pip install numpy pandas joblib scikit-learn xgboost
"""

import os
import sys
import json
import logging
import argparse
import inspect

import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
MODEL_EXTENSIONS = (".json", ".ubj", ".joblib", ".pkl", ".pickle")


def clean_path(text):
    return text.strip().strip('"').strip("'")

def ask_existing_file(question):
    while True:
        path = clean_path(input(f"\n{question}\n> "))
        if os.path.isfile(path):
            return path
        print("  File not found. Check the path and try again.")

def list_models():
    if not os.path.isdir(MODEL_FOLDER):
        return []
    return sorted(f for f in os.listdir(MODEL_FOLDER) if f.lower().endswith(MODEL_EXTENSIONS))

def ask_model():
    models = list_models()
    if not models:
        print(f"\nNo models found in: {MODEL_FOLDER}")
        print("Put your model files in that folder and run the script again,")
        print("or type the path to a model file now.")
        return ask_existing_file("Path to the model file:")

    print(f"\nModels found in {MODEL_FOLDER}:")
    for i, name in enumerate(models, 1):
        print(f"  {i}) {name}")
    other = len(models) + 1
    print(f"  {other}) Use a model from somewhere else")

    while True:
        answer = input("Type a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= other:
            n = int(answer)
            if n == other:
                return ask_existing_file("Path to the model file:")
            return os.path.join(MODEL_FOLDER, models[n - 1])
        print("  Please type one of the numbers above.")

def ask_output_path(default):
    answer = clean_path(input(f"\nWhere should the predictions be saved? [press Enter for '{default}']\n> "))
    return answer or default

def interactive_setup(args):
    print("=" * 50)
    print("  Activity prediction")
    print("=" * 50)

    if not args.model_path:
        args.model_path = ask_model()
    if not args.features_csv:
        args.features_csv = ask_existing_file("Path to the features CSV (made by fingerprints.py):")
    if not args.output_results:
        args.output_results = ask_output_path("predictions.csv")
    return args


def load_xgboost(path):
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("This is an XGBoost model but xgboost is not installed. Run: pip install xgboost")

    objective = ""
    if path.lower().endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                objective = json.load(f)["learner"]["objective"]["name"]
        except Exception:
            pass

    model = xgb.XGBRegressor() if objective.startswith("reg:") else xgb.XGBClassifier()
    model.load_model(path)
    return model

def load_model(path):
    logging.info(f"Loading model from {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext in (".json", ".ubj"):
        model = load_xgboost(path)
    else:
        model = joblib.load(path)

    if not hasattr(model, "predict"):
        raise ValueError("This file does not contain a usable model (no predict function).")
    logging.info(f"Model type: {type(model).__name__}")
    return model

def load_features(path):
    logging.info(f"Loading features from {path}")
    df = pd.read_csv(path)
    if "compound_name" not in df.columns:
        raise ValueError("The features CSV needs a 'compound_name' column.")
    names = df["compound_name"].tolist()
    return df.drop(columns=["compound_name"]), names

def match_feature_names(features, model):
    expected = None
    if hasattr(model, "get_booster"):
        try:
            expected = model.get_booster().feature_names
        except Exception:
            pass
    if expected is None:
        expected = getattr(model, "feature_names_in_", None)

    if expected is not None:
        if features.shape[1] != len(expected):
            raise ValueError(
                f"The CSV has {features.shape[1]} features but the model expects {len(expected)}. "
                "Make sure the features were made with the same fingerprint type the model was trained on."
            )
        features = features.copy()
        features.columns = expected
        return features
    return features.values


def _supports_n_jobs_kwargs(func):
    try:
        return "n_jobs" in inspect.signature(func).parameters
    except (ValueError, TypeError):
        return False

def predict(model, features, names, n_jobs):
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and features.shape[1] != n_expected:
        raise ValueError(
            f"The CSV has {features.shape[1]} features but the model expects {n_expected}. "
            "Make sure the features were made with the same fingerprint type the model was trained on."
        )

    supports_n_jobs_attr = hasattr(model, "n_jobs")
    supports_n_jobs_kwargs = _supports_n_jobs_kwargs(model.predict)
    is_multithreaded = supports_n_jobs_attr or supports_n_jobs_kwargs
    
    if is_multithreaded:
        if supports_n_jobs_attr:
            model.n_jobs = n_jobs
        if hasattr(model, "nthread"):
            model.nthread = n_jobs
        logging.info(f"Predicting {len(names)} molecules using {n_jobs} CPU core(s) (Multi-threaded model)")
    else:
        logging.info(f"Predicting {len(names)} molecules using 1 CPU core (Single-threaded model like SVM)")
    
    predict_kwargs = {}
    if supports_n_jobs_kwargs:
        predict_kwargs["n_jobs"] = n_jobs

    predicted_classes = model.predict(features, **predict_kwargs)
    results = pd.DataFrame({"compound_name": names, "Predicted_Class": predicted_classes})

    if hasattr(model, "predict_proba"):
        try:
            proba_kwargs = {}
            if _supports_n_jobs_kwargs(model.predict_proba):
                proba_kwargs["n_jobs"] = n_jobs
            
            probs = model.predict_proba(features, **proba_kwargs)
            for i, cls in enumerate(model.classes_):
                results[f"{cls}_Probability"] = probs[:, i]
        except Exception as e:
            logging.warning(f"Could not calculate probabilities: {e}")

    has_probs = any(c.endswith("_Probability") for c in results.columns)
    if not has_probs and hasattr(model, "decision_function"):
        try:
            results["Decision_Score"] = model.decision_function(features)
        except Exception as e:
            logging.warning(f"Could not calculate decision scores: {e}")

    return results


def save(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    results.to_csv(path, index=False)
    logging.info(f"Saved to {os.path.abspath(path)}")


def parse_args():
    default_n_jobs = max(1, (os.cpu_count() or 2) // 2)
    p = argparse.ArgumentParser(description="Predict activity from fingerprint features.")
    p.add_argument("--model_path", help="Trained model file (.json / .ubj for XGBoost, .joblib / .pkl for sklearn models)")
    p.add_argument("--features_csv", help="Features CSV with a 'compound_name' column")
    p.add_argument("--output_results", help="Where to save the predictions CSV")
    p.add_argument("--n_jobs", type=int, default=default_n_jobs, 
                   help=f"CPU cores for prediction. Defaults to half your cores ({default_n_jobs}). Automatically ignored for single-threaded models (e.g., SVM).")
    p.add_argument("--verbose", action="store_true", help="Show more detail")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    try:
        args = interactive_setup(args)
        for path in (args.model_path, args.features_csv):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")

        print()
        model = load_model(args.model_path)
        features, names = load_features(args.features_csv)
        features = match_feature_names(features, model)
        results = predict(model, features, names, args.n_jobs)
        save(results, args.output_results)

        print("\nPredicted class counts:")
        print(results["Predicted_Class"].value_counts().to_string())
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()