# Text Classification Project

This is a simple machine learning project that classifies text into different topics.

The project uses the **20 Newsgroups** dataset from `scikit-learn`. It trains and tests two models:

- Naive Bayes
- Logistic Regression

## Topics Used

The project classifies text into these 4 categories:

- `sci.space`
- `rec.sport.hockey`
- `talk.politics.mideast`
- `comp.graphics`

## Features

- Loads text data from the 20 Newsgroups dataset
- Converts text into numbers using TF-IDF
- Trains Naive Bayes and Logistic Regression models
- Shows model accuracy
- Prints a classification report
- Saves a confusion matrix chart as `my_project_chart.png`
- Includes a small demo prediction for a sample sentence

## Requirements

Install the required Python libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run

Run this command:

```bash
python project.py
```

After running, the program will show the model results in the terminal and save the confusion matrix image.

## Group Members and Contribution

This was a group project of 3 members.

| Member | Contribution |
| --- | --- |
| Sawika Rajendra Gonarkar  | Completed the full project, including data loading, preprocessing, model training, evaluation, chart generation, and demo testing. |
| Kanishkaa W M  | No contribution. |
| Sayandeep Ghosh| No contribution. |

## Output

The project prints the accuracy of both models and generates a confusion matrix image named:

```text
my_project_chart.png
```

