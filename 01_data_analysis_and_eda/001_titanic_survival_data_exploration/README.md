# Titanic Survival Data Exploration

This project analyzes actual passenger rows from a Titanic CSV. It reports overall survival rate with a 95% Wilson interval, survival rates by sex, passenger class, embarkation port, and age group when those columns are present, plus age and fare summaries.

## Run

No third-party Python packages are required. Use a passenger CSV with a binary Survived column, for example a standard Titanic train.csv.

~~~sh
python main.py --input path/to/train.csv --output titanic_survival_report.json
~~~

A different outcome column can be selected with --target-column. Accepted outcome values are 0/1, yes/no, true/false, survived/died, or dead. Rows with missing or unrecognized outcomes are counted and excluded from survival rates. Age, Sex, Pclass, Embarked, and Fare are optional; the report analyzes the fields that exist.

The JSON report contains counts, subgroup rates, confidence intervals, numeric summaries, and per-column missing-value counts. The analysis is descriptive; subgroup differences do not show that a characteristic caused survival. No data or results are fabricated or downloaded automatically.