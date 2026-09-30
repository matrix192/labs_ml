"""Предобработка данных соревнования Kaggle Spaceship Titanic.

Скрипт не обучает модель: он готовит признаки так, чтобы преобразователь
можно было безопасно использовать внутри sklearn Pipeline.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
RESULTS_DIR = BASE_DIR / "results"
TARGET = "Transported"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {DATA_PATH}. Скачайте train.csv с Kaggle и положите его в папку data."
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # ID и полное имя почти уникальны, поэтому не используем их как признаки.
    # Номер каюты также слишком детален: извлекаем только палубу и сторону.
    # Это снижает размерность и риск переобучения на уникальных значениях.
    features = df.drop(columns=[TARGET, "PassengerId", "Name"]).copy()
    cabin_parts = features["Cabin"].str.split("/", expand=True)
    features["CabinDeck"] = cabin_parts[0]
    features["CabinSide"] = cabin_parts[2]
    features = features.drop(columns="Cabin")
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    missing_before = df.isna().sum().rename("missing_before")
    missing_before.to_csv(RESULTS_DIR / "missing_values_before.csv")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )

    # fit_transform используется только для демонстрации. В модели следует
    # помещать preprocessor в Pipeline и вызывать fit только на train-части.
    transformed = preprocessor.fit_transform(features)
    processed = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=df.index,
    )
    missing_after = processed.isna().sum().rename("missing_after")
    missing_after.to_csv(RESULTS_DIR / "missing_values_after.csv")
    processed.to_csv(RESULTS_DIR / "processed_train.csv", index=False)

    summary = [
        "Первые 5 строк исходного датасета:",
        df.head().to_string(),
        "\nРазмер исходного датасета: " + str(df.shape),
        "\nПропуски до обработки:",
        missing_before.to_string(),
        "\nПропуски после обработки признаков:",
        missing_after.to_string(),
        "\nРазмер после кодирования: " + str(processed.shape),
        "\nПроверка: всего пропусков после обработки = " + str(int(processed.isna().sum().sum())),
    ]
    text = "\n".join(summary)
    (RESULTS_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
