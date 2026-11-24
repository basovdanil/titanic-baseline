#!/usr/bin/env python3
import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TITLES = ("Mr", "Mrs", "Miss", "Master", "Rare")
FEATURES = [
    "pclass",
    "sex",
    "age",
    "age_missing",
    "sibsp",
    "parch",
    "family",
    "alone",
    "fare",
    "embarked_C",
    "embarked_Q",
    "embarked_S",
] + [f"title_{t}" for t in TITLES]
REQUIRED = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
MODELS = ("logreg", "forest", "boost")
DEFAULT_CSV = ROOT / "data" / "train.csv"
DEFAULT_MODEL = ROOT / "work" / "model.pkl"
DEFAULT_METRICS = ROOT / "work" / "metrics.json"
DEFAULT_SUBMISSION = ROOT / "work" / "submission.csv"


def add_features(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("нет колонок: " + ", ".join(missing))
    out = pd.DataFrame(index=df.index)
    out["pclass"] = df["Pclass"].astype(int)
    out["sex"] = (df["Sex"].astype(str).str.lower() == "female").astype(int)
    out["age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["age_missing"] = out["age"].isna().astype(int)
    out["sibsp"] = df["SibSp"].astype(int)
    out["parch"] = df["Parch"].astype(int)
    out["family"] = out["sibsp"] + out["parch"] + 1
    out["alone"] = (out["family"] == 1).astype(int)
    out["fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    embarked = df["Embarked"].fillna("S").astype(str).str.upper()
    for code in ("C", "Q", "S"):
        out[f"embarked_{code}"] = (embarked == code).astype(int)
    title = df["Name"].astype(str).str.extract(r",\s*([^.]+)\.")[0].str.strip()
    title = title.where(title.isin(TITLES[:-1]), "Rare")
    for name in TITLES:
        out[f"title_{name}"] = (title == name).astype(int)
    return out[FEATURES]


def make_model(kind):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    imputer = SimpleImputer(strategy="median")
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        return Pipeline([("imp", imputer), ("sc", StandardScaler()), ("clf", clf)])
    if kind == "forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        return Pipeline([("imp", imputer), ("clf", clf)])
    if kind == "boost":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.06)
        return Pipeline([("imp", imputer), ("clf", clf)])
    raise ValueError(f"неизвестная модель: {kind}")


def cv_scores(X, y, kinds, folds):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = {}
    for kind in kinds:
        acc = cross_val_score(make_model(kind), X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        auc = cross_val_score(make_model(kind), X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        rows[kind] = {
            "accuracy": float(acc.mean()),
            "accuracy_std": float(acc.std()),
            "roc_auc": float(auc.mean()),
        }
    return rows


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла: {path}")
    return pd.read_csv(path)


def write_submission(frame, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def read_submission(path):
    frame = pd.read_csv(path)
    problems = []
    if list(frame.columns) != ["PassengerId", "Survived"]:
        problems.append("колонки не PassengerId,Survived")
    elif frame.empty:
        problems.append("нет строк")
    else:
        if frame["PassengerId"].duplicated().any():
            problems.append("повтор PassengerId")
        if not frame["PassengerId"].astype(str).str.fullmatch(r"\d+").all():
            problems.append("PassengerId не целые")
        if not set(frame["Survived"].unique()) <= {0, 1}:
            problems.append("Survived вне 0/1")
    return frame, problems


def cmd_fetch(args):
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url, timeout=60) as response:
        payload = response.read()
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    rows = len(pd.read_csv(target))
    print(f"file={target} rows={rows} bytes={len(payload)} sha256={digest}")
    return 0


def cmd_train(args):
    df = load_csv(args.data)
    if "Survived" not in df.columns:
        raise ValueError("в обучающем файле нет колонки Survived")
    X = add_features(df)
    y = df["Survived"].astype(int)
    rows = cv_scores(X, y, MODELS, args.folds)
    best = max(rows, key=lambda name: rows[name]["accuracy"])
    model = make_model(best).fit(X, y)
    payload = {
        "model": model,
        "kind": best,
        "features": FEATURES,
        "metrics": rows,
        "rows": int(len(df)),
        "folds": args.folds,
        "seed": SEED,
        "source": str(args.data),
    }
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    metrics_path = Path(args.metrics)
    metrics_path.write_text(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "rows": payload["rows"],
        "folds": args.folds,
        "best": best,
        "accuracy": rows[best]["accuracy"],
        "roc_auc": rows[best]["roc_auc"],
        "models": rows,
        "model": str(model_path),
    }
    print(f"rows={payload['rows']} folds={args.folds}")
    for name, scores in rows.items():
        print(
            f"{name:8s} acc={scores['accuracy']:.4f} ±{scores['accuracy_std']:.4f} "
            f"auc={scores['roc_auc']:.4f}"
        )
    print(f"BEST={best} ACC={rows[best]['accuracy']:.4f} AUC={rows[best]['roc_auc']:.4f}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_predict(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"нет модели: {model_path} (сначала train)")
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    kind = payload["kind"] if args.kind in (None, "", "best") else args.kind
    from sklearn.metrics import accuracy_score

    if args.test:
        frame = load_csv(args.test)
        if "Survived" in frame.columns:
            raise ValueError("в тестовом файле не должно быть Survived")
        model = payload["model"]
        pred = model.predict(add_features(frame)).astype(int)
        submission = pd.DataFrame({"PassengerId": frame["PassengerId"].astype(int), "Survived": pred})
        accuracy = None
    else:
        frame = load_csv(args.data or DEFAULT_CSV)
        if "Survived" not in frame.columns:
            raise ValueError("для holdout-режима нужен Survived")
        from sklearn.model_selection import train_test_split

        train_df, holdout = train_test_split(
            frame, test_size=args.holdout, stratify=frame["Survived"], random_state=SEED
        )
        model = make_model(kind).fit(add_features(train_df), train_df["Survived"].astype(int))
        pred = model.predict(add_features(holdout)).astype(int)
        accuracy = float(accuracy_score(holdout["Survived"].astype(int), pred))
        submission = pd.DataFrame({"PassengerId": holdout["PassengerId"].astype(int), "Survived": pred})
    out = write_submission(submission, args.out)
    _, problems = read_submission(out)
    report = {
        "model": str(model_path),
        "kind": kind,
        "mode": "test" if args.test else "holdout",
        "rows": int(len(submission)),
        "survived": int(submission["Survived"].sum()),
        "accuracy": accuracy,
        "out": str(out),
        "problems": problems,
    }
    print(f"mode={report['mode']} rows={report['rows']} survived={report['survived']} out={out}")
    if accuracy is not None:
        print(f"ACC={accuracy:.4f}")
    if problems:
        print("FORMAT_FAIL " + "; ".join(problems))
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 4
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_submit(args):
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"нет файла: {path}")
    _, problems = read_submission(path)
    if problems:
        print("FORMAT_FAIL " + "; ".join(problems))
        return 4
    command = ["kaggle", "competitions", "submit", "-c", args.competition, "-f", str(path), "-m", args.message]
    if args.dry_run:
        print(" ".join(command))
        return 0
    if shutil.which("kaggle") is None:
        print("kaggle CLI не установлен: нужен токен ~/.kaggle/kaggle.json (Settings -> Create New Token)")
        return 3
    return subprocess.call(command)


def cmd_check(args):
    checks = []
    df = load_csv(args.data)
    X = add_features(df)
    checks.append(("features", list(X.columns) == FEATURES))
    checks.append(("no_target_leak", "Survived" not in X.columns and len(X) == len(df)))
    metrics_path = Path(args.metrics)
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = max(metrics, key=lambda name: metrics[name]["accuracy"]) if metrics else None
    checks.append(("metrics", bool(metrics)))
    checks.append(("accuracy", bool(best) and metrics[best]["accuracy"] >= args.min_accuracy))
    submission = Path(args.submission)
    problems = ["нет файла"]
    if submission.exists():
        _, problems = read_submission(submission)
        checks.append(("submission", not problems))
    else:
        checks.append(("submission", False))
    model_path = Path(args.model)
    checks.append(("model", model_path.exists()))
    for name, ok in checks:
        print(f"CHECK {name}={'OK' if ok else 'FAIL'}")
    failed = [name for name, ok in checks if not ok]
    print(f"CHECKED={len(checks)} FAILED={len(failed)}")
    if failed:
        print("failed: " + ", ".join(failed))
        if problems and problems != ["нет файла"]:
            print("submission: " + "; ".join(problems))
        return 4
    print(f"best={best} accuracy={metrics[best]['accuracy']:.4f} rows={len(df)} submission={submission}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="titanic.py",
        description="Воспроизводимый пайплайн Titanic: CV трёх моделей, генератор submission, проверка формата.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="скачать публичную копию обучающей таблицы")
    fetch.add_argument("--url", default=SOURCE_URL)
    fetch.add_argument("--out", default=str(DEFAULT_CSV))
    fetch.set_defaults(func=cmd_fetch)

    train = sub.add_parser("train", help="кросс-валидация моделей и обучение лучшей")
    train.add_argument("--data", default=str(DEFAULT_CSV))
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--model", default=str(DEFAULT_MODEL))
    train.add_argument("--metrics", default=str(DEFAULT_METRICS))
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="сгенерировать submission.csv")
    predict.add_argument("--test", help="файл без Survived: предсказать все строки")
    predict.add_argument("--data", default=str(DEFAULT_CSV))
    predict.add_argument("--holdout", type=float, default=0.2)
    predict.add_argument("--kind", default="best", choices=("best",) + MODELS)
    predict.add_argument("--model", default=str(DEFAULT_MODEL))
    predict.add_argument("--out", default=str(DEFAULT_SUBMISSION))
    predict.add_argument("--json", action="store_true")
    predict.set_defaults(func=cmd_predict)

    submit = sub.add_parser("submit", help="отправить submission через kaggle CLI")
    submit.add_argument("--file", default=str(DEFAULT_SUBMISSION))
    submit.add_argument("--competition", default="titanic")
    submit.add_argument("--message", default="titanic-baseline")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    check = sub.add_parser("check", help="проверить артефакты и формат submission")
    check.add_argument("--data", default=str(DEFAULT_CSV))
    check.add_argument("--metrics", default=str(DEFAULT_METRICS))
    check.add_argument("--model", default=str(DEFAULT_MODEL))
    check.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    check.add_argument("--min-accuracy", type=float, default=0.78)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 3
    except ValueError as error:
        print(f"error: {error}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
