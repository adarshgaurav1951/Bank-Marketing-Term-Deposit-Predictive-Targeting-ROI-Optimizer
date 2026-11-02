# Bank Marketing - Term Deposit Subscription: Project Idea (Frame by Frame)

## 1. One-Line Idea
Predict customer subscription to term deposits by building classification models that optimize marketing campaign effectiveness.

## 2. Business Context
A bank runs phone marketing campaigns to sell term deposits. Every call costs staff time and money, and most clients say no. If the bank can rank clients by how likely they are to subscribe, it can call the best prospects first, skip the unlikely ones, and get more subscriptions for the same effort.

The model is not the final goal. The goal is a better campaign: higher conversion rate, lower cost per subscriber, and a clear plan for whom to call.

## 3. Problem Statement
Using data from past phone campaigns of a Portuguese bank, build and evaluate classification models that predict whether a client will subscribe to a term deposit (`y` = yes or no). Handle the class imbalance, avoid using information that is not available before a call, evaluate the models with marketing metrics (lift, cumulative gains, profit), and turn the results into practical campaign recommendations.

## 4. Goals
- Understand which client, contact, and economic factors relate to subscription
- Build a leakage-free pipeline from raw data to predictions
- Compare several classifiers, starting from a simple baseline
- Handle the imbalance between yes and no outcomes correctly
- Evaluate with metrics that fit a marketing campaign, not only accuracy
- Choose how many clients to call using an explicit cost and revenue assumption
- Explain the drivers of subscription in plain business language
- Document the project so someone else can reproduce it

## 5. What the Project Title Means

| Phrase | What you do in this project |
|---|---|
| Predict customer subscription | Binary classification of `y` (yes = 1, no = 0) |
| Classification models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and optionally XGBoost or LightGBM |
| Optimize marketing campaign effectiveness | Rank clients by probability, measure lift and gains, and choose a call list size that maximizes profit |

## 6. Key Questions to Answer
1. What share of clients subscribe, and how imbalanced is the data?
2. Which client groups (age, job, education, marital status) subscribe most?
3. Does the way and time of contact matter (cellular vs. telephone, month, day of week)?
4. How do previous campaign outcomes (`poutcome`, `previous`, `pdays`) affect the result?
5. Do economic indicators (interest rates, employment, consumer confidence) relate to subscription?
6. Does contacting the same client many times help or hurt?
7. Which model ranks likely subscribers best?
8. If the bank can only call the top 10, 20, or 30 percent of clients, how many subscribers does it capture?
9. At what call list size is profit highest under your cost assumptions?

## 7. Dataset

**Source:** UCI Machine Learning Repository, "Bank Marketing" (Moro, Cortez, and Rita, 2014). Data from direct phone marketing campaigns of a Portuguese bank.

| File | Rows | Inputs | Notes |
|---|---|---|---|
| `bank-additional-full.csv` | 41,188 | 20 | **Recommended.** Ordered by date (May 2008 to November 2010), includes five economic indicators |
| `bank-additional.csv` | 4,119 | 20 | 10% random sample, for slow algorithms or quick tests |
| `bank-full.csv` | 45,211 | 16 | Older version with fewer inputs |
| `bank.csv` | 4,521 | 16 | 10% sample of `bank-full.csv` |

Kaggle also hosts copies. One copy removes the `duration` column to prevent leakage, so check which version you have.

**Format:** the files use a semicolon separator, so load with `pd.read_csv(path, sep=";")`.

**Column groups (additional version):**
- **Client:** age, job, marital, education, default, housing, loan
- **Last contact:** contact, month, day_of_week, duration
- **Campaign history:** campaign, pdays, previous, poutcome
- **Economic indicators:** emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- **Target:** y

Kaggle copies may use underscores instead of dots in column names (for example `emp_var_rate`).

### The `duration` problem (data leakage)
`duration` is the length of the last call in seconds. The dataset documentation warns that it strongly affects the outcome (a call of zero seconds always ends in no), and that it is not known before the call is made. A model that uses it looks excellent but cannot be used to decide whom to call. Remove it for the main model. You may keep it only in a separate benchmark experiment, clearly labelled.

### Other data details to check
- `unknown` appears as a category in several columns. Treat it as its own category before deciding to drop or impute.
- `pdays` = 999 is documented as "client was not previously contacted". Convert it to a flag instead of treating it as a number of days.
- The default rate (share of yes) is roughly 11 percent in the full additional file. Confirm with `y.mean()`.
- Because the rows are ordered by date, consider a time-based split as an extra experiment.

## 8. Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Modeling and metrics | scikit-learn |
| Imbalanced data | imbalanced-learn |
| Boosting (optional) | xgboost or lightgbm |
| Explainability | permutation importance, optionally shap |
| Environment | Google Colab or Antigravity with one Jupyter notebook |
| Version control (optional) | Git and GitHub |

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap jupyter
```

## 9. Project Structure (keep it simple)
```
bank-marketing/
├── data/
│   └── bank-additional-full.csv       <- never edited
├── bank_marketing_classification.ipynb <- all code, top to bottom
├── AGENTS.md                           <- rules for the AI agent (optional)
└── idea.md                             <- this file
```
Add `models/`, `app.py`, and `README.md` only after the notebook works.

### Suggested AGENTS.md
```markdown
# Agent Rules
- Project: bank term deposit classification in ONE notebook: bank_marketing_classification.ipynb
- Use only pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn (xgboost, shap optional)
- Never modify the original CSV
- Drop `duration` from the main model (leakage). Benchmark runs must be labelled
- Split with stratify=y before any resampling
- Apply SMOTE or undersampling ONLY inside an imblearn Pipeline on training data
- Never report accuracy alone: report recall, precision, F1, ROC-AUC, PR-AUC, and lift
- Keep cells short, one purpose per cell, with a comment explaining why
- Add a markdown heading before each frame and a 2-3 line summary after it
```

---

# The Frames

Each frame has a goal, the work to do, a code hint, and a "done when" check. Work through them in order, and do not start a frame before the previous one is done.

---

## Frame 1: Understand the Business Problem
**Goal:** know what the bank wants before touching data.

**Do:**
- Write the problem in three lines: who is called, what counts as success, and what the campaign costs
- Read the dataset documentation, especially the attribute list and the `duration` note
- Learn the basic terms: term deposit, conversion rate, lift, cumulative gains

**Done when:** you can explain, in two sentences, why a ranked list of clients is more useful to the bank than a simple yes/no prediction.

---

## Frame 2: Define Success Metrics
**Goal:** decide how you will judge results before you build anything.

**Do:**
- Business metrics: conversion rate in the called group, lift in the top decile, share of all subscribers captured in the top N percent, and profit
- ML metrics: recall, precision, F1, ROC-AUC, PR-AUC
- State a cost assumption, for example: cost per call = 1 unit and revenue per subscriber = 10 units (these are assumptions, not facts from the data)

**Done when:** a short "metrics and assumptions" markdown cell exists at the top of the notebook.

---

## Frame 3: Get the Data
**Goal:** have the right file in the right place.

**Do:**
- Download the Bank Marketing dataset from the UCI repository (or a Kaggle copy)
- Put `bank-additional-full.csv` in `data/` and never edit it
- Record the source and the citation

**Code hint:**
```python
df = pd.read_csv("data/bank-additional-full.csv", sep=";")
```

**Done when:** `df.shape` shows about 41,188 rows and 21 columns (20 inputs plus `y`) for the recommended file.

---

## Frame 4: Set Up the Workspace
**Goal:** a clean place to work.

**Do:**
- Create the project folder, the notebook, and (optionally) `AGENTS.md`
- Colab: upload the CSV and install extras with `!pip install imbalanced-learn shap`
- Antigravity: create a virtual environment, install the libraries, and select it as the kernel
- Add one markdown heading per frame in the notebook

**Done when:** the imports cell runs without errors.

---

## Frame 5: Load and Inspect
**Goal:** know the data before changing it.

**Do:**
- Check shape, column names, types, nulls, and duplicates
- Look at the target distribution and calculate the "always predict no" accuracy
- Count `unknown` values per column
- Check the values of `pdays`, `previous`, and `campaign`

**Code hint:**
```python
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print(df["y"].value_counts(normalize=True))
print((df == "unknown").sum())
```

**Done when:** you have written down the default rate, the "always no" accuracy, and the list of data problems.

---

## Frame 6: Clean the Data
**Goal:** fix problems without losing information.

**Do:**
- Remove exact duplicate rows and document how many were found
- Convert `y` to 1 and 0
- Keep `unknown` as a category, or group it carefully if a column is mostly unknown (the `default` column is a likely case)
- Convert `pdays` = 999 into a `previously_contacted` flag and set the days value to missing or zero for those clients
- Check for impossible values (negative or unrealistic ages, durations, counts)
- Document each decision in a markdown note

**Done when:** a clean DataFrame exists, every change is explained, and the raw file is untouched.

---

## Frame 7: Explore the Data (EDA)
**Goal:** find patterns that answer the key questions.

**Do:**
- Subscription rate by job, age band, education, marital status, month, day of week, and contact type
- Subscription rate by `poutcome` and by number of contacts (`campaign`)
- Distributions of age and the economic indicators, split by `y`
- Correlation heatmap for numeric columns (watch the highly correlated economic indicators)
- One-sentence finding under every chart

**Done when:** you can answer Key Questions 1 to 6 from your own charts.

---

## Frame 8: Engineer Features and Handle `duration`
**Goal:** create useful inputs without leaking the answer.

**Do:**
- **Drop `duration`** from the main feature set
- Create `previously_contacted` from `pdays`
- Create age bands
- Create a contacted-before flag from `poutcome`
- Group rare job categories if needed
- Optional: a campaign intensity feature such as total contacts, or a quarter or season feature from `month`
- Keep a separate, clearly labelled experiment with `duration` only as a benchmark to show how much leakage inflates the score

**Done when:** `X` contains only information known before a call, and the benchmark experiment is clearly separate.

---

## Frame 9: Split and Preprocess
**Goal:** a leak-free train and test setup.

**Do:**
- Stratified split: 80 percent train, 20 percent test
- Build a `ColumnTransformer`: scaling for numeric columns, one-hot encoding for categorical columns
- Optional second experiment: a time-based split (train on earlier months, test on later months), since the data is ordered by date
- Fit all preprocessing on training data only, inside a pipeline

**Code hint:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Done when:** train and test have the same yes rate, and nothing is fitted on the test set.

---

## Frame 10: Baseline Models
**Goal:** a reference point that every other model must beat.

**Do:**
- `DummyClassifier(strategy="most_frequent")`: high accuracy, zero recall
- Logistic Regression with default settings
- Report the confusion matrix, recall, precision, F1, ROC-AUC, and PR-AUC

**Done when:** you can explain why the dummy model's high accuracy is meaningless.

---

## Frame 11: Handle Class Imbalance
**Goal:** find out whether imbalance handling helps, and by how much.

**Do:** compare the same model under each strategy:

| Strategy | How |
|---|---|
| None | Baseline |
| Class weights | `class_weight="balanced"` |
| Oversampling | SMOTE or SMOTENC inside an `imblearn` pipeline |
| Undersampling | `RandomUnderSampler` |
| Threshold tuning | Keep the model, change the cutoff (Frame 13) |

**Code hint:**
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipe = ImbPipeline([
    ("prep", prep),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

**Done when:** a comparison table exists, and you know that resampling touches the training data only.

---

## Frame 12: Compare and Tune Models
**Goal:** pick the best classifier fairly.

**Do:**
- Train Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and optionally XGBoost or LightGBM
- Use stratified 5-fold cross-validation on the training data
- Tune the top two models with `RandomizedSearchCV`, scored on PR-AUC or ROC-AUC
- Plot ROC and precision-recall curves for all models on one chart

**Done when:** one model is chosen based on cross-validation, and the test set has not been used for any decision yet.

---

## Frame 13: Evaluate with Marketing Metrics
**Goal:** turn model scores into campaign decisions.

**Do:**
- Score the test set with the chosen model
- Build a **gains and lift table by decile**: conversion rate, lift, and cumulative share of subscribers captured
- Plot the cumulative gains curve against random targeting
- Plot precision and recall against the threshold
- Calculate **profit** at each call list size using your cost and revenue assumptions, and find the maximum
- Check calibration with a calibration curve and the Brier score
- Report the final confusion matrix at your chosen cutoff

**Code hint:**
```python
def gains_table(y_true, y_prob, bins=10):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": y_prob}).sort_values("p", ascending=False)
    d["decile"] = pd.qcut(np.arange(len(d)), bins, labels=False) + 1
    t = d.groupby("decile")["y"].agg(["count", "sum"])
    t["rate"] = t["sum"] / t["count"]
    t["lift"] = t["rate"] / d["y"].mean()
    t["cum_capture"] = t["sum"].cumsum() / d["y"].sum()
    return t
```

**Done when:** you can say, "Calling the top X percent captures Y percent of subscribers, with lift Z, at profit P under these assumptions."

---

## Frame 14: Explain the Model
**Goal:** know why the model ranks clients the way it does.

**Do:**
- Logistic Regression coefficients or odds ratios
- Permutation importance on the test set
- Optional: SHAP summary plot, plus an explanation for one high-probability client and one low-probability client
- Translate findings into plain language, for example "clients with a previous successful campaign are far more likely to subscribe"

**Done when:** the top drivers are listed and make business sense. Remember that explanations describe the model, not proven causes.

---

## Frame 15: Campaign Recommendations
**Goal:** convert analysis into actions a marketing manager can use.

**Do:**
- Who to prioritize (segments with the highest predicted rates)
- Whom to skip or contact less
- Best contact channel, month, and day, based on your EDA
- A suggested cap on repeated contacts per client
- The recommended call list size and the expected result under your assumptions
- Limits: old data, one bank and country, assumed costs, and the fact that correlation is not causation

**Done when:** a one-page summary answers all nine key questions.

---

## Frame 16: Document and Present
**Goal:** make the project easy to understand and reuse.

**Do:**
- Write the conclusion section in the notebook
- Restart the kernel and run all cells top to bottom to confirm it works
- Write `README.md`: problem, data source and citation, method, key results table, charts, and how to run it
- Optional: save the final pipeline with `joblib`, and build a small Streamlit app that scores a client and shows the risk band
- Publish on GitHub (link to the dataset source instead of re-uploading if the license is unclear)

**Done when:** a stranger can open the notebook, run it, and understand your result.

---

## 10. Expected Outputs
- Subscription rate by segment charts
- Class imbalance chart and the "always no" accuracy trap
- Imbalance strategy comparison table
- Model comparison table (recall, precision, F1, ROC-AUC, PR-AUC)
- ROC and precision-recall curves
- Gains and lift table, cumulative gains curve
- Profit vs. call list size chart
- Feature importance or SHAP plot
- Written campaign recommendations
- A benchmark showing how `duration` inflates scores (clearly labelled)

## 11. Success Criteria
- The notebook runs from top to bottom without errors
- `duration` is excluded from the main model, with the reason documented
- The split is stratified, and resampling happens only inside the training pipeline
- No fitting of preprocessing on the test set
- At least four classifiers and at least three imbalance strategies are compared
- Lift, cumulative gains, and profit are reported, not accuracy alone
- The call list size is justified by stated cost and revenue assumptions
- The top drivers are explained in plain language
- You can explain the project, including why accuracy fails and why `duration` is removed, without reading from the notebook

## 12. Common Mistakes to Avoid
- Keeping `duration` and reporting very high scores
- Reporting accuracy only
- Resampling before the split, or resampling the test set
- Dropping `unknown` rows without checking what they represent
- Reading `pdays` = 999 as a number of days
- Using the default 0.5 threshold without checking costs
- Using a random split when the data is ordered by date, without at least discussing a time-based check
- Treating feature importance as proof that a factor causes subscription
- Copying agent code blindly without understanding it

## 13. Ethics and Limits
- Age and marital status are personal attributes. Check whether the model treats groups very differently, and discuss whether such columns should be used.
- Marketing should respect customer consent and do-not-contact preferences. A ranked list is not permission to contact someone repeatedly.
- The data is from a Portuguese bank between 2008 and 2010, a period that includes the financial crisis. Patterns may not hold for other banks, countries, or years.
- Cost and revenue figures are assumptions you choose. Real campaign decisions need real figures from the bank.
- This is a learning project, not a production system.

## 14. Stretch Ideas
- Compare random and time-based splits and discuss the difference
- Cost-sensitive learning vs. resampling
- Probability calibration (Platt or isotonic) and Brier score comparison
- Customer segmentation with clustering, then a model per segment
- Uplift thinking: who subscribes because of a call vs. who would subscribe anyway (discussion only without experimental data)
- A Streamlit app that returns a subscription probability and a call or skip suggestion
- Repeat the pipeline on `bank-full.csv` to check whether the conclusions hold

## 15. Roadmap

| Phase | Frames | Rough time |
|---|---|---|
| Setup and understanding | 1 to 4 | 1 day |
| Data work | 5 to 8 | 3 to 4 days |
| Modeling | 9 to 12 | 3 to 4 days |
| Business evaluation and explanation | 13 to 15 | 2 to 3 days |
| Documentation and publishing | 16 | 1 to 2 days |

## 16. Citation
Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. Related paper: Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31.

## 17. Next Steps
1. Download `bank-additional-full.csv` from UCI and place it in `data/`
2. Create the notebook with the 16 frame headings
3. Complete Frames 1 to 5 and record the subscription rate and the "always no" accuracy
4. Continue frame by frame, running each cell before moving on
