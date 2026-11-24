import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import titanic


def passenger(index, **over):
    row = {
        "PassengerId": index,
        "Pclass": 3,
        "Name": f"Test, Mr. Passenger {index}",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.0,
        "Embarked": "S",
    }
    row.update(over)
    return row


def quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


class FeaturesTest(unittest.TestCase):
    def test_feature_list_and_order(self):
        frame = pd.DataFrame([passenger(1), passenger(2, Sex="female")])
        out = titanic.add_features(frame)
        self.assertEqual(list(out.columns), titanic.FEATURES)
        self.assertEqual(len(out), 2)

    def test_sex_family_alone(self):
        frame = pd.DataFrame(
            [
                passenger(1, Sex="female", SibSp=1, Parch=1),
                passenger(2, Sex="male"),
            ]
        )
        out = titanic.add_features(frame)
        self.assertEqual(out.loc[0, "sex"], 1)
        self.assertEqual(out.loc[1, "sex"], 0)
        self.assertEqual(out.loc[0, "family"], 3)
        self.assertEqual(out.loc[0, "alone"], 0)
        self.assertEqual(out.loc[1, "family"], 1)
        self.assertEqual(out.loc[1, "alone"], 1)

    def test_missing_age_kept_as_nan_with_flag(self):
        frame = pd.DataFrame([passenger(1, Age=None), passenger(2, Age=40)])
        out = titanic.add_features(frame)
        self.assertTrue(pd.isna(out.loc[0, "age"]))
        self.assertEqual(out.loc[0, "age_missing"], 1)
        self.assertEqual(out.loc[1, "age_missing"], 0)

    def test_missing_embarked_defaults_to_southampton(self):
        frame = pd.DataFrame([passenger(1, Embarked=None), passenger(2, Embarked="C")])
        out = titanic.add_features(frame)
        self.assertEqual(out.loc[0, "embarked_S"], 1)
        self.assertEqual(out.loc[0, "embarked_C"], 0)
        self.assertEqual(out.loc[1, "embarked_C"], 1)
        self.assertEqual(out.loc[1, "embarked_S"], 0)

    def test_titles_split_common_and_rare(self):
        frame = pd.DataFrame(
            [
                passenger(1, Name="A, Mrs. B"),
                passenger(2, Name="A, Dr. B"),
                passenger(3, Name="A, Master. B"),
            ]
        )
        out = titanic.add_features(frame)
        self.assertEqual(out.loc[0, "title_Mrs"], 1)
        self.assertEqual(out.loc[1, "title_Rare"], 1)
        self.assertEqual(out.loc[2, "title_Master"], 1)
        self.assertEqual(int(out[["title_Mr", "title_Rare"]].loc[1].sum()), 1)

    def test_missing_column_raises(self):
        frame = pd.DataFrame([passenger(1)]).drop(columns=["Embarked"])
        with self.assertRaises(ValueError) as ctx:
            titanic.add_features(frame)
        self.assertIn("Embarked", str(ctx.exception))

    def test_required_columns_are_checked_once(self):
        frame = pd.DataFrame([passenger(1)]).drop(columns=["Fare", "Parch"])
        with self.assertRaises(ValueError) as ctx:
            titanic.add_features(frame)
        message = str(ctx.exception)
        self.assertIn("Fare", message)
        self.assertIn("Parch", message)


class ModelTest(unittest.TestCase):
    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            titanic.make_model("svm")

    def test_all_kinds_can_fit_and_predict(self):
        frame = pd.DataFrame([passenger(i, Sex="female" if i % 2 else "male") for i in range(1, 21)])
        X = titanic.add_features(frame)
        y = pd.Series([i % 2 for i in range(1, 21)])
        for kind in titanic.MODELS:
            model = titanic.make_model(kind).fit(X, y)
            pred = model.predict(X)
            self.assertEqual(len(pred), len(y), kind)

    def test_cv_reports_accuracy_and_auc(self):
        frame = pd.DataFrame([passenger(i, Sex="female" if i % 2 else "male") for i in range(1, 41)])
        X = titanic.add_features(frame)
        y = pd.Series([i % 2 for i in range(1, 41)])
        scores = titanic.cv_scores(X, y, ("logreg",), 2)
        self.assertIn("logreg", scores)
        self.assertGreaterEqual(scores["logreg"]["accuracy"], 0.0)
        self.assertLessEqual(scores["logreg"]["accuracy"], 1.0)
        self.assertGreaterEqual(scores["logreg"]["roc_auc"], 0.0)
        self.assertLessEqual(scores["logreg"]["roc_auc"], 1.0)


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, text, name="submission.csv"):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_good_file_has_no_problems(self):
        path = self.write("PassengerId,Survived\n1,0\n2,1\n3,0\n")
        frame, problems = titanic.read_submission(path)
        self.assertEqual(problems, [])
        self.assertEqual(len(frame), 3)

    def test_wrong_columns(self):
        path = self.write("id,Survived\n1,0\n")
        _, problems = titanic.read_submission(path)
        self.assertEqual(len(problems), 1)
        self.assertIn("PassengerId", problems[0])

    def test_empty_file(self):
        path = self.write("PassengerId,Survived\n")
        _, problems = titanic.read_submission(path)
        self.assertIn("нет строк", problems)

    def test_duplicate_passenger_id(self):
        path = self.write("PassengerId,Survived\n1,0\n1,1\n")
        _, problems = titanic.read_submission(path)
        self.assertIn("повтор PassengerId", problems)

    def test_non_integer_passenger_id(self):
        path = self.write("PassengerId,Survived\n1.5,0\n")
        _, problems = titanic.read_submission(path)
        self.assertIn("PassengerId не целые", problems)

    def test_label_outside_zero_one(self):
        path = self.write("PassengerId,Survived\n1,0\n2,2\n")
        _, problems = titanic.read_submission(path)
        self.assertIn("Survived вне 0/1", problems)

    def test_write_creates_parent_directory(self):
        frame = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1]})
        out = self.dir / "deep" / "nested" / "sub.csv"
        titanic.write_submission(frame, out)
        self.assertTrue(out.exists())
        self.assertEqual(len(pd.read_csv(out)), 2)


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.small = cls.dir / "train_small.csv"
        cls.test_only = cls.dir / "test_small.csv"
        cls.model = cls.dir / "work" / "model.pkl"
        cls.metrics = cls.dir / "work" / "metrics.json"
        cls.submission = cls.dir / "work" / "submission.csv"
        rows = [
            passenger(
                i,
                Sex="female" if i % 2 else "male",
                Pclass=1 + i % 3,
                Age=20.0 + i,
                Fare=7.0 + i,
                Name=f"Passenger, {'Mrs' if i % 2 else 'Mr'}. Number {i}",
            )
            for i in range(1, 61)
        ]
        frame = pd.DataFrame(rows)
        frame["Survived"] = [i % 2 for i in range(1, 61)]
        frame.to_csv(cls.small, index=False)
        frame.drop(columns=["Survived"]).to_csv(cls.test_only, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def test_01_train_writes_artifacts(self):
        code = quiet(
            titanic.main,
            ["train", "--data", str(self.small), "--folds", "2", "--model", str(self.model), "--metrics", str(self.metrics)],
        )
        self.assertEqual(code, 0)
        self.assertTrue(self.model.exists())
        self.assertTrue(self.metrics.exists())

    def test_02_predict_holdout(self):
        code = quiet(
            titanic.main,
            ["predict", "--data", str(self.small), "--holdout", "0.25", "--model", str(self.model), "--out", str(self.submission)],
        )
        self.assertEqual(code, 0)
        frame = pd.read_csv(self.submission)
        self.assertEqual(list(frame.columns), ["PassengerId", "Survived"])
        self.assertEqual(len(frame), 15)
        self.assertTrue(set(frame["Survived"].unique()) <= {0, 1})

    def test_03_predict_test_file_without_target(self):
        out = self.dir / "work" / "test_submission.csv"
        code = quiet(
            titanic.main,
            ["predict", "--test", str(self.test_only), "--model", str(self.model), "--out", str(out)],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(pd.read_csv(out)), 60)

    def test_04_check_passes_on_built_artifacts(self):
        code = quiet(
            titanic.main,
            [
                "check",
                "--data",
                str(self.small),
                "--metrics",
                str(self.metrics),
                "--model",
                str(self.model),
                "--submission",
                str(self.submission),
            ],
        )
        self.assertEqual(code, 0)

    def test_05_check_fails_on_impossible_threshold(self):
        code = quiet(
            titanic.main,
            [
                "check",
                "--data",
                str(self.small),
                "--metrics",
                str(self.metrics),
                "--model",
                str(self.model),
                "--submission",
                str(self.submission),
                "--min-accuracy",
                "1.01",
            ],
        )
        self.assertEqual(code, 4)

    def test_06_missing_csv_returns_three(self):
        code = quiet(titanic.main, ["train", "--data", str(self.dir / "nope.csv")])
        self.assertEqual(code, 3)

    def test_07_predict_without_model_returns_three(self):
        code = quiet(
            titanic.main,
            ["predict", "--data", str(self.small), "--model", str(self.dir / "nope.pkl")],
        )
        self.assertEqual(code, 3)

    def test_08_train_without_target_returns_four(self):
        code = quiet(titanic.main, ["train", "--data", str(self.test_only), "--folds", "2"])
        self.assertEqual(code, 4)

    def test_09_submit_rejects_broken_file(self):
        broken = self.dir / "broken.csv"
        broken.write_text("PassengerId,Survived\n1,0\n1,1\n", encoding="utf-8")
        code = quiet(titanic.main, ["submit", "--file", str(broken)])
        self.assertEqual(code, 4)

    def test_10_submit_reports_missing_cli(self):
        if shutil.which("kaggle") is not None:
            self.skipTest("kaggle CLI установлен, ветка с кодом 3 не воспроизводится")
        code = quiet(titanic.main, ["submit", "--file", str(self.submission)])
        self.assertEqual(code, 3)

    def test_11_submit_missing_file_returns_three(self):
        code = quiet(titanic.main, ["submit", "--file", str(self.dir / "gone.csv")])
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
