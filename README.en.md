# titanic-baseline

[![tests](https://github.com/basovdanil/titanic-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/basovdanil/titanic-baseline/actions/workflows/tests.yml)

Cross-validation of three models on the Titanic data with 17 engineered features, a holdout check
and a submission generator. Random forest wins: **accuracy 0,8316 ± 0,0120**, AUC 0,8827.

Not a leaderboard chase: the repo shows the whole path — data check, features, stratified 5-fold CV,
holdout, artifact verification — with metrics written to `work/metrics.json`.

## Quick start

```
python titanic.py fetch                  # public copy of train.csv into data/
python titanic.py train                  # CV of three models, work/metrics.json + model.pkl
python titanic.py predict --holdout 0.2  # holdout check
python titanic.py submit --dry-run       # show the submission command
python titanic.py check --min-accuracy 0.78
```

## Models and results

| Model | Accuracy | AUC |
| --- | --- | --- |
| LogisticRegression + StandardScaler | 0,8283 ± 0,0105 | 0,8719 |
| RandomForest (300 trees) | **0,8316 ± 0,0120** | **0,8827** |
| HistGradientBoosting | 0,8215 ± 0,0224 | 0,8742 |

## Exit codes

`0` done, `2` bad arguments, `3` file missing, `4` data or artifact error.

## Tests

```
python -m unittest discover -s tests
```

28 tests: feature extraction, fold determinism, metric arithmetic, artifact and submission format.

## Scope

A textbook dataset, used as a discipline exercise: fixed seeds, explicit holdout, verified artifacts.

---

More projects: [github.com/basovdanil](https://github.com/basovdanil) · write-ups: [portfolio](https://basovdanil.github.io/portfolio/)
