# labs_ml — практическое задание: предобработка Spaceship Titanic

Датасет: [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data).

Скрипт выполняет все пункты задания:

1. читает `train.csv` из Kaggle;
2. выводит первые строки и информацию о таблице;
3. считает пропуски по каждому столбцу до и после обработки;
4. заполняет числовые признаки медианой, категориальные — модой;
5. нормализует числовые признаки в диапазон от 0 до 1 (`MinMaxScaler`);
6. кодирует категории через `OneHotEncoder(handle_unknown="ignore")` внутри `ColumnTransformer`.

Последнее решение не вызывает утечки данных: статистики заполнения, масштабирования и набор категорий будут изучаться только на обучающей выборке, если объект `preprocessor` включить в `Pipeline` модели. Новые категории в тестовых данных не приведут к ошибке.

## Запуск

Скачайте `train.csv` на странице соревнования Kaggle и положите его в `data/train.csv`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python preprocess_spaceship_titanic.py
```

Результаты появятся в папке `results/`:

- `missing_values_before.csv` и `missing_values_after.csv`;
- `processed_train.csv` — итоговые подготовленные признаки;
- `summary.txt` — текстовый вывод для отчёта.

Исходный CSV не добавлен в GitHub: его нужно получить с Kaggle, поэтому репозиторий остаётся компактным и соответствует правилам источника.
