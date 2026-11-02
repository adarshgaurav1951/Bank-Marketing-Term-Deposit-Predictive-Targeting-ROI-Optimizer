# Product Requirement Document (PRD)
## Bank Marketing: Predictive Term Deposit Targeting Engine & ROI Optimizer

---

### Document Information
- **Project Title:** Bank Marketing - Predictive Term Deposit Subscription
- **Document Version:** 1.0.0
- **Document Status:** Approved & Implemented
- **Target Release:** Q4 / Production Pilot
- **Primary Stakeholders:** Marketing Campaign Managers, Telemarketing Operations, Head of Retail Banking, Data Science & Analytics Team

---

## 1. Executive Summary & Business Objective

### 1.1 Context
Retail banks regularly conduct direct telemarketing campaigns to drive uptake of term deposit products (fixed-term deposits with guaranteed interest). Telemarketing campaigns incur substantial labor and operational overheads per outbound call. In historical Portuguese banking campaigns (UCI Bank Marketing dataset), over **88.7%** of outbound calls ended without a subscription, resulting in wasted operational expenditure, client call fatigue, and suboptimal conversion rates (~11.3% base rate).

### 1.2 Objective
Develop and deploy an end-to-end, leakage-free machine learning decision system that accurately predicts the probability of a client subscribing to a term deposit (`y = yes / no`) **prior** to initiating a call. 

By replacing blanket cold calling with probability-ranked decile targeting and unit-cost profit optimization, the bank will:
1. **Maximize Campaign ROI:** Target the top customer percentiles where response concentration is highest.
2. **Reduce Call Waste:** Skip bottom percentiles with negligible conversion propensity.
3. **Prevent Client Fatigue:** Implement contact capping based on diminishing returns.
4. **Empower Campaign Managers:** Provide an interactive decision-support application (`app.py`) for lead scoring, batch decile prioritization, and live profit-threshold tuning.

---

## 2. Five Golden Rules & Anti-Leakage Directives

To guarantee mathematical integrity and operational viability, the solution strictly enforces the **Five Golden Rules**:

| Rule # | Golden Rule | Rationale & Enforcement |
|:---|:---|:---|
| **Rule 1** | **Strictly Drop `duration` from Main Models** | The duration of a call is only known *after* a call is completed (a 0-second call is always non-converted; a 15-minute call is highly likely converted). Including `duration` introduces fatal data leakage. It may only be evaluated in a benchmark experiment to quantify artificial score inflation. |
| **Rule 2** | **Reject "Accuracy Alone" Trap** | A naive baseline predicting "No" achieves **88.73% accuracy** while capturing **0% of subscribers** (Zero Recall). The primary evaluation metrics must be **PR-AUC, ROC-AUC, Recall, Precision, Lift@Deciles, and Profit**. |
| **Rule 3** | **Leak-Free Stratified Splitting & Pipeline Resampling** | Train/Test split (80/20) must be stratified on target `y`. All categorical encoding, scaling, and class-imbalance resampling (SMOTE, Undersampling) must occur strictly **inside cross-validation / training pipelines** to eliminate data leakage into test sets. |
| **Rule 4** | **Economics-Driven Threshold & Cutoff Selection** | The default 0.5 decision cutoff is discarded. Optimal probability cutoffs and call list sizes are dynamically determined by business unit economics: `Profit = (True Positives × Revenue per Subscriber) - (Total Calls × Cost per Call)`. |
| **Rule 5** | **Measure Cumulative Gains & Decile Lift** | Marketing campaigns operate under budget and staffing caps. Evaluation must quantify: "If we contact only the top 10%, 20%, or 30% of ranked clients, what cumulative share of all buyers do we capture?" |

---

## 3. Scope & Key User Stories

### 3.1 In-Scope Deliverables
1. **Data Pipeline & Cleaning:** Handling unknown category values, transforming `pdays=999` into pre-contact flags, temporal feature alignment, and outlier validation.
2. **Comprehensive EDA:** Multivariate analysis linking conversion rates to client demographics, contact mechanics, macroeconomic variables, and campaign fatigue.
3. **Class Imbalance & Model Benchmark Suite:** Rigorous comparison of 4+ classifiers across 3+ imbalance mitigation strategies (Class Weights, SMOTE, Undersampling) evaluated with 5-fold Stratified Cross-Validation.
4. **Marketing Metrics & Unit Economics Engine:** Automated decile gains table, lift curve, profit curves under configurable cost-revenue assumptions.
5. **Model Explainability:** Permutation importance, Logistic Regression odds ratios, and SHAP attribution to deliver actionable business rules.
6. **Executable Frame-by-Frame Jupyter Notebook:** `bank_marketing_classification.ipynb` implementing all 16 frames.
7. **Production Web Application:** An interactive Streamlit app (`app.py`) providing lead scoring, batch CSV prioritization, and live profit simulator.

### 3.2 Key User Stories
- **As a Telemarketing Operations Manager**, I want a prioritized daily call list with predicted deciles so my call agents focus outbound time on high-propensity leads first.
- **As a Campaign Strategy Director**, I want to simulate marketing campaign profit by adjusting outbound call cost (€/call) and expected deposit margin (€/subscriber) to determine the exact call volume cutoff.
- **As a Compliance & Governance Officer**, I need assurance that data leakage is eliminated, personal attributes are ethically audited, and the model explains its primary drivers transparently.

---

## 4. Dataset Architecture & Schema

### 4.1 Data Source
- **Origin:** UCI Machine Learning Repository / Portuguese Banking Institution direct marketing data (May 2008 – November 2010).
- **Recommended File:** `bank-additional-full.csv` (41,188 rows, 20 input attributes + 1 target `y`).
- **Delimiter:** Semicolon (`;`).

### 4.2 Attribute Taxonomy & Preprocessing Rules

| Group | Features | Type | Handling Rules |
|:---|:---|:---|:---|
| **Client Demographics** | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` | Numeric & Categorical | - `age`: Scaled; grouped into strategic life-stage bands.<br>- `job`, `marital`, `education`: Categorical; `"unknown"` retained as distinct category.<br>- `default`: Heavily dominated by "no" and "unknown" (almost zero "yes"). Encoded carefully. |
| **Last Contact** | `contact`, `month`, `day_of_week`, `duration` | Categorical & Numeric | - `contact`: Cellular vs. Telephone.<br>- `month`, `day_of_week`: Seasonality indicators.<br>- `duration`: **DROPPED** from predictive features to prevent lookahead leakage. |
| **Campaign Dynamics** | `campaign`, `pdays`, `previous`, `poutcome` | Numeric & Categorical | - `campaign`: Number of contacts during current campaign. Used to assess fatigue.<br>- `pdays`: `999` represents "never previously contacted". Converted to binary flag `previously_contacted` (1 if pdays < 999, else 0).<br>- `poutcome`: Previous campaign outcome (`success`, `failure`, `nonexistent`). Strong predictor. |
| **Macroeconomic Indicators** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Continuous Numeric | Scaled using `StandardScaler`. Reflect monetary climate (Euribor rates, employment indices) during European sovereign debt crisis. |
| **Target** | `y` | Binary Target | Transformed: `yes` $\rightarrow 1$, `no` $\rightarrow 0$. Base subscription rate: ~11.27%. |

---

## 5. Machine Learning Architecture & Methodology

```
Raw CSV (bank-additional-full.csv)
       │
       ▼
Data Cleaning & Feature Engineering
  - Drop duration (Leakage prevention)
  - pdays == 999 -> previously_contacted flag
  - Age brackets & campaign fatigue indicators
       │
       ▼
Stratified Train/Test Split (80% Train / 20% Test, stratify=y)
       │
       ├─────────────────────────────────────────┐
       ▼ (Training Pipeline Only)                ▼ (Held-out Test)
Preprocess (ColumnTransformer: Num + Cat)    Never touched until
       │                                     final evaluation!
Imbalance Strategy Evaluation
  - Class Weights ('balanced')
  - SMOTE oversampling
  - Random Under-Sampling
       │
       ▼
Model Suite & Stratified 5-Fold Cross-Validation
  - Dummy Classifier (Benchmark)
  - Logistic Regression (Odds ratio interpretability)
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier / Gradient Boosting
       │
       ▼
Hyperparameter Tuning (RandomizedSearchCV on PR-AUC & ROC-AUC)
       │
       ▼
Model Evaluation on Unseen Test Set
  - PR-AUC, ROC-AUC, Recall, Precision, F1
  - Leakage Comparison Experiment (Model with vs without duration)
       │
       ▼
Marketing & Financial Optimization
  - Decile Lift Table (Gains & Lift@10%, @20%, @30%)
  - Cumulative Gains Curve & Lift Chart
  - Unit Economic Profit Curve: Profit = TP * Rev - (TP+FP) * Cost
  - Calibration Curve & Brier Score
       │
       ▼
Explainability & Deployment
  - Permutation Importance & SHAP Summary
  - Persist Pipeline via Joblib (artifacts/best_model.joblib)
  - Interactive Streamlit Dashboard (app.py)
```

---

## 6. Functional Requirements (16 Frames Breakdown)

| Frame # | Name | Core Functional Requirements | Success Criteria |
|:---|:---|:---|:---|
| **Frame 1** | Understand Business Problem | Formalize telemarketing operational constraints, business revenue model, and why ranked deciles beat binary classification. | Problem definition clearly documented. |
| **Frame 2** | Define Success Metrics | Establish dual-tier metrics: ML metrics (PR-AUC, ROC-AUC) + Business metrics (Decile lift, subscriber capture rate, net campaign profit). | Metric formulas and unit cost assumptions articulated. |
| **Frame 3** | Get the Data | Ingest `bank-additional-full.csv`, verify integrity, checksum, and cite source. | 41,188 rows loaded with 21 attributes. |
| **Frame 4** | Set Up Environment | Verify Python libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `streamlit`. | Clean execution environment with no dependency conflicts. |
| **Frame 5** | Load & Inspect | Inspect nulls, deduplicate, analyze base subscription rate (~11.27%), identify `"unknown"` frequency across categorical columns. | Verified 12 duplicates, zero raw nulls, documented category distributions. |
| **Frame 6** | Clean Data | Drop duplicates, binarize target `y`, create `previously_contacted` flag from `pdays=999`, retain `"unknown"` as legitimate category. | Clean dataframe ready for downstream pipeline. |
| **Frame 7** | Exploratory Data Analysis | Visualize conversion rates across jobs, age groups, education, contact type, month, economic indicators, and contact repetition. | Actionable EDA takeaways documented under each figure. |
| **Frame 8** | Feature Engineering & Leakage Isolation | Remove `duration` from primary training feature set. Construct benchmark dataset with `duration` to demonstrate leakage impact. | Explicit pre-call feature space `X` isolated. |
| **Frame 9** | Split & Preprocessing Pipeline | Execute 80/20 stratified split. Construct `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals). | Leakage-free pipeline; zero preprocessing fitted on test set. |
| **Frame 10** | Baseline Models | Implement `DummyClassifier(strategy='most_frequent')` and standard unweighted Logistic Regression. | Baseline benchmarks established (Dummy achieves 88.7% acc, 0% recall). |
| **Frame 11** | Imbalance Strategies | Test class weights, SMOTE, and RandomUnderSampler within `imblearn.pipeline.Pipeline`. | Systematic comparison matrix of imbalance techniques. |
| **Frame 12** | Model Comparison & Tuning | Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost using Stratified 5-Fold CV. Perform randomized hyperparameter search. | Best performing model selected by PR-AUC / ROC-AUC. |
| **Frame 13** | Marketing Evaluation | Generate 10-decile gains and lift table. Plot cumulative gains vs random baseline. Compute profit curve across threshold range. | Decile table generated; optimal calling cutoff determined. |
| **Frame 14** | Model Explainability | Compute permutation importance and SHAP summary plots. Interpret macro and behavioral feature effects. | Key drivers translated into plain banking language. |
| **Frame 15** | Campaign Recommendations | Synthesize recommendations: optimal call volume, channel preference (cellular), seasonality, and contact frequency caps ($\le 3$ calls). | 1-page executive campaign action plan. |
| **Frame 16** | Production Packaging | Serialize best model pipeline to disk (`artifacts/best_model.joblib`), generate standalone notebook, and deliver Streamlit app (`app.py`). | Reproducible notebook runs end-to-end; app starts cleanly. |

---

## 7. Business Metrics & Unit Economics Model

### 7.1 Mathematical Definitions

1. **Conversion Rate (called list):**
   $$\text{Conversion Rate} = \frac{\text{True Positives (TP)}}{\text{Total Contacts (TP + FP)}}$$

2. **Decile Lift:**
   $$\text{Lift}_{\text{decile}} = \frac{\text{Conversion Rate}_{\text{decile}}}{\text{Baseline Population Conversion Rate}}$$

3. **Cumulative Share of Subscribers Captured (Gains):**
   $$\text{Cumulative Gains} = \frac{\sum_{i=1}^{k} \text{Subscribers Captured in Decile } i}{\text{Total Actual Subscribers in Dataset}}$$

4. **Campaign Net Profit ($P$):**
   $$P = (\text{TP} \times R) - ((\text{TP} + \text{FP}) \times C)$$
   Where:
   - $R$ = Average net lifetime value / revenue earned per new term deposit subscriber (Default: **€50.00** or 10 units)
   - $C$ = Cost per outbound phone contact including staff labor and telephony overhead (Default: **€5.00** or 1 unit)
   - The user/manager can interactively tune $R$ and $C$ in the interactive application.

---

## 8. Web Application Requirements (`app.py`)

The application must be designed with **premium visual aesthetics** (modern dark/light theme, sleek glassmorphism, responsive controls, clear typography, dynamic KPI cards) and deliver 4 primary views:

1. **Campaign ROI Simulator:**
   - Sliders for Contact Cost ($C$), Deposit Value ($R$), and Total Customer Pool Size.
   - Interactive profit curve displaying the optimal cutoff threshold, peak profit, and optimal call volume.
2. **Single Lead Scorer:**
   - Real-time input form for client demographics, economic indicators, and previous campaign history.
   - Dynamic Gauge / Probability bar showing subscription likelihood and instant recommendation badge:
     * **Tier 1 (High Priority - Immediate Call):** Top decile propensity (>40%).
     * **Tier 2 (Standard Outreach):** Moderate propensity (15%–40%).
     * **Tier 3 (Skip / Digital Only):** Low propensity (<15%).
3. **Batch Lead Prioritization:**
   - CSV upload supporting raw client records.
   - Outputs ranked file with predicted probabilities, assigned deciles, and action flags for call center export.
4. **Model Insights & Governance:**
   - Visual comparison of models (ROC, PR Curves, Decile Lift, Leakage impact comparison).
   - Compliance review of personal attributes and campaign fatigue insights.

---

## 9. Non-Functional Requirements & Compliance

- **Reproducibility:** A single random seed (`random_state=42`) is pinned across all train/test splits, pipelines, and models.
- **Latency:** Single-lead real-time inference latency $< 50\text{ ms}$. Batch scoring of 10,000 leads $< 2.0\text{ s}$.
- **Data Protection & Ethics:** Explicit analysis of demographic variables (`age`, `marital status`). Models must be monitored for adverse bias, and telemarketing teams must strictly enforce national do-not-call registries and GDPR consent rules.

---

## 10. Sign-off Criteria
1. `PRD.md` approved and committed.
2. `bank_marketing_classification.ipynb` executes from top to bottom with zero errors, generating all 16 frames with visual plots and decile analysis.
3. Clean separation between leak-free production model and `duration` benchmark.
4. Serialized production model saved in `artifacts/` or `models/`.
5. Streamlit app `app.py` fully operational with high visual fidelity and interactive ROI simulator.
6. Comprehensive `README.md` with operational guidance and business findings.
