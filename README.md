# titanic-baseline — воспроизводимый пайплайн с кросс-валидацией

[![tests](https://github.com/basovdanil/titanic-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/basovdanil/titanic-baseline/actions/workflows/tests.yml)

**[English](README.en.md)** · [Русский](README.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/basovdanil/basovdanil/main/assets/tools.png"
       alt="Восемь инструментов с тестами и кодами возврата" width="100%">
</p>

Предсказание выживания на Titanic как рабочий каркас: извлечение признаков, кросс-валидация трёх
моделей, обучение, генерация `submission.csv` и проверка артефактов перед отправкой.
Данные реальные и публичные, все цифры воспроизводятся одной командой с фиксированным `SEED = 42`.

Это не соревновательный ноутбук на максимальный скор. Это минимальный скелет ML-задачи, в котором
видно, откуда взялось каждое число и что нужно проверить перед отправкой.

## Пайплайн

```text
data/train.csv ──▶ 17 признаков ──▶ CV, 5 фолдов ──▶ лучшая модель ──▶ work/model.pkl
   891 × 12         add_features        3 модели         metrics.json
                                                              │
                            work/submission.csv ◀─────────────┘ predict
                                      │
                                      ├──▶ check: 6 проверок артефактов, exit 0 или 4
                                      └──▶ submit: нужен Kaggle CLI и токен
```

## Установка и запуск

Нужен Python 3.9+ и `pandas` + `scikit-learn` (проверено на 3.11, pandas 2.2.3, scikit-learn 1.9.1).

```bash
python titanic.py fetch                        # скачать публичную копию train.csv в data/
python titanic.py train                        # CV трёх моделей, work/metrics.json + work/model.pkl
python titanic.py predict --holdout 0.2        # проверка на отложенной выборке
python titanic.py predict --test data/test.csv --out work/submission.csv
python titanic.py check --min-accuracy 0.78    # проверить артефакты и формат submission
python titanic.py submit --dry-run             # показать команду отправки
```

`train`, `predict`, `check` принимают `--json`. Пути по умолчанию: `data/train.csv`, `work/model.pkl`,
`work/metrics.json`, `work/submission.csv`.

## Данные

`data/train.csv` скачан из публичного зеркала `datasciencedojo/datasets`, 891 строка, 12 колонок,
60 302 байта, sha256 `4a437fde…4af4bd7`. Доля выживших 0,3838.
Пропуски: `Age` 177, `Cabin` 687, `Embarked` 2. Колонка `Cabin` в признаки не идёт: пропусков 77 %,
а выигрыш от неё на 891 строке не проверяем. Тестовая половина соревнования не выкладывается:
публичная копия содержит только `train.csv`.

## Признаки (17)

```text
pclass, sex, age, age_missing, sibsp, parch, family, alone, fare,
embarked_C, embarked_Q, embarked_S, title_Mr, title_Mrs, title_Miss, title_Master, title_Rare
```

`sex` кодируется как `female = 1`. `family = sibsp + parch + 1`, `alone` — её инверсия.
`age_missing` ловит 177 пропусков возраста, сам `age` заполняется медианой внутри модели,
чтобы заполнение не утекало из теста. Титул достаётся из `Name` регуляркой `,\s*([^.]+)\.`,
четыре частых титула остаются отдельными колонками, остальные (`Dr`, `Rev`, `Major` и прочие)
сваливаются в `title_Rare`. Пропуски `Embarked` считаются саутгемптонскими: их всего два,
и это порт большинства.

## Кросс-валидация (5 фолдов, стратифицированно)

| Модель | Accuracy | ROC AUC |
| --- | --- | --- |
| LogisticRegression + StandardScaler | 0,8283 ± 0,0105 | 0,8719 |
| RandomForest (300 деревьев) | **0,8316 ± 0,0120** | **0,8827** |
| HistGradientBoosting | 0,8215 ± 0,0224 | 0,8742 |

Метрики расхождения между моделями лежат внутри разброса фолдов: 0,822 против 0,832 при σ ≈ 0,012
это шум, а не преимущество. `train` выбирает лучшую по accuracy и пишет её в `work/metrics.json`.

Проверка на отложенных 20 % (179 строк, 56 выживших): accuracy 0,8268. Совпадает с CV в пределах
того же разброса, то есть модель не переобучена на фолды.

## Submission

`predict` без `--test` делит обучающую таблицу на train/holdout и пишет предсказания по holdout
(так видно accuracy). `predict --test file.csv` предсказывает файл без `Survived` и пишет все строки.
Формат: `PassengerId,Survived`, метки строго `0` и `1`.

`submit` отправляет файл через `kaggle` CLI. В этом окружении CLI и токена нет, поэтому команда
честно печатает, чего не хватает, и возвращает код `3`. Реальный балл на лидерборде не получен и
в README не заявлен: нужен `~/.kaggle/kaggle.json` (Kaggle → Settings → Create New Token).

## Проверка артефактов

```bash
python titanic.py check
CHECK features=OK
CHECK no_target_leak=OK
CHECK metrics=OK
CHECK accuracy=OK
CHECK submission=OK
CHECK model=OK
CHECKED=6 FAILED=0
best=forest accuracy=0.8316 rows=891 submission=...\work\submission.csv
```

Что проверяется: набор и порядок признаков; отсутствие `Survived` среди признаков и сохранение
числа строк; наличие метрик; порог accuracy (`--min-accuracy`, по умолчанию 0,78); формат
submission (колонки, непустота, целые `PassengerId` без дублей, метки в `{0,1}`); наличие файла модели.

```bash
python -m unittest discover -s tests -v
```

28 тестов, около 23 секунд: признаки (порядок колонок, `sex`, `family`/`alone`, флаг пропущенного
возраста, заполнение `Embarked`, разбиение титулов на частые и редкие, отказ при нехватке колонок);
модели (неизвестный тип, обучение и предсказание всеми тремя, диапазон CV-метрик); формат
submission (хороший файл, чужие колонки, пустой файл, дубли, дробные id, метка вне 0/1, создание
каталога); CLI-коды на собранных артефактах.

## Коды возврата

| Код | Значение |
| --- | --- |
| 0 | успех, все проверки прошли |
| 2 | неверные аргументы (`argparse`) |
| 3 | нет файла данных, модели или `kaggle` CLI |
| 4 | данные не разбираются, нет колонки `Survived`, формат submission битый, проверка провалилась |

## Границы

- Балл на Kaggle не получен: нет токена. Всё, что до отправки, воспроизводится локально.
- Модели базовые, подбор гиперпараметров не делался. Прирост от тюнинга на 891 строке будет
  в пределах ±0,01 accuracy, то есть внутри шума фолдов.
- Признак `Cabin` и номер билета не используются. `Cabin` отбрасывается из-за 77 % пропусков,
  `Ticket` — из-за ручного разбора формата.
- Одиночный holdout зависит от `random_state`. При другом seed accuracy меняется на 1–2 %;
  поэтому в README основой считаются CV-цифры, а holdout приведён как проверка согласованности.
- Возраст заполняется медианой внутри `Pipeline`, поэтому при CV медиана считается по фолду.
  Значение медианы в модель не сохраняется и не переиспользуется между запусками.
- Пол (главный признак) кодируется одним числом, без one-hot и без взаимодействий с классом.
- Данные не содержат персональных выгрузок: это публичный учебный набор 1912 года.

---

Другие проекты: [github.com/basovdanil](https://github.com/basovdanil) · разборы с числами: [portfolio](https://basovdanil.github.io/portfolio/)
