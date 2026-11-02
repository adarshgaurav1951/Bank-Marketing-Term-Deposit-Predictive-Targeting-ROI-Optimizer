# 🏦 Bank Marketing: Term Deposit Predictive Targeting & ROI Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, leak-free machine learning decision system and interactive ROI simulation dashboard built for telemarketing campaigns of retail bank term deposits.

---

## 🎯 Executive Summary & Business Impact

Direct phone marketing campaigns for term deposits suffer from low organic conversion rates (~11.3%), incurring massive operational costs on cold calls that end in rejection.

By applying **decile-ranked propensity scoring** and **unit economic optimization**:
- **4.57x Lift in Top Decile:** Decile 1 conversion rate reaches **51.5%** (vs. 11.3% base rate).
- **65.7% Buyer Capture in Top 20%:** Over **65% of all deposit subscribers** are captured by calling just the top 2 quintiles (80% call volume reduction).
- **+331.3% Profit Uplift:** At €5/call and €50/deposit value, model-guided prioritization delivers **€22,515 net campaign profit** compared to only **€5,220** under blanket cold calling.
- **Enforced 3-Call Limit:** Demonstrates that conversion plummets past 3 contacts, preventing costly campaign fatigue.

---

## 🛡️ The Five Golden Rules

| # | Golden Rule | Implementation & Safeguard |
|---|---|---|
| **1** | **Drop `duration` from Main Feature Matrix** | `duration` (call length) is lookahead leakage because it is fundamentally unknown prior to dialing. A separate benchmark proves `duration` artificially inflates ROC-AUC from 0.81 to 0.95, but is completely unusable in operational production. |
| **2** | **Never Rely on Accuracy Alone** | A naive baseline predicting "No" achieves **88.73% accuracy** with **0% recall**. Evaluation strictly relies on PR-AUC, ROC-AUC, Decile Lift, Cumulative Gains, and Profit. |
| **3** | **Stratified Split Before Pipeline Resampling** | 80/20 train/test split is stratified on target `y`. Preprocessing (`ColumnTransformer`) and imbalance handling (Class Weights, SMOTE) are fitted strictly on training folds inside pipelines. |
| **4** | **Economics-Driven Threshold Cutoff** | The default 0.5 threshold is replaced with an economics optimizer: $\text{Profit} = (\text{TP} \times \text{Rev}) - (\text{Calls} \times \text{Cost})$. The optimal threshold is dynamically identified (e.g. 0.52 for €5 cost / €50 rev). |
| **5** | **Measure Cumulative Gains by Decile** | Call center capacity is fixed. Models are judged by how many buyers are captured in the top 10%, 20%, and 30% of ranked prospects. |

---

## 📊 Benchmark Results

### 1. Model Comparison Leaderboard (Leak-Free Pre-Call Features)

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Test Recall | Test Precision | Lift@10% | Capture@20% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Weighted)** ⭐ | **0.4598** | **0.4819** | **0.8096** | **65.2%** | **38.2%** | **4.57x** | **65.7%** |
| Random Forest (Balanced) | 0.4602 | 0.4798 | 0.8133 | 63.9% | 40.6% | 4.66x | 65.5% |
| Logistic Regression (Balanced) | 0.4451 | 0.4468 | 0.7998 | 64.6% | 35.9% | 4.46x | 64.2% |
| Decision Tree (Balanced) | 0.4130 | 0.4464 | 0.7951 | 62.6% | 39.8% | 4.62x | 65.0% |
| *Dummy Baseline (Most Frequent)* | *0.1127* | *0.1127* | *0.5000* | *0.0%* | *0.0%* | *1.00x* | *20.0%* |

### 2. Proof of Data Leakage (The `duration` Trap)

| Experiment | Features Used | Test ROC-AUC | Test PR-AUC | Production Viability |
|---|---|:---:|:---:|---|
| **Production Model** | Pre-Call Features Only | **0.8096** | **0.4819** | ✅ **100% Operational (Deployable)** |
| **Benchmark (Leakage)** | Includes `duration` | **0.9515** (+0.14) | **0.6708** (+0.19) | ❌ **Fatal Leakage (Lookahead Trap)** |

---

## 📈 Decile Gains Table (Test Set Evaluation)

| Decile | Contacts | Subscribers | Conversion Rate | Decile Lift | Cum. Contacts (%) | Cum. Gains (% Captured) | Cum. Lift |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 824 | 424 | **51.5%** | **4.57x** | 10.0% | **45.7%** | 4.57x |
| **2** | 824 | 186 | **22.6%** | **2.00x** | 20.0% | **65.7%** | 3.29x |
| **3** | 823 | 80 | **9.7%** | 0.86x | 30.0% | **74.4%** | 2.48x |
| **4** | 824 | 41 | **5.0%** | 0.44x | 40.0% | **78.8%** | 1.97x |
| **5** | 823 | 35 | **4.3%** | 0.38x | 50.0% | **82.5%** | 1.65x |
| **6** | 824 | 46 | **5.6%** | 0.50x | 60.0% | **87.5%** | 1.46x |
| **7** | 823 | 35 | **4.3%** | 0.38x | 70.0% | **91.3%** | 1.30x |
| **8** | 824 | 32 | **3.9%** | 0.34x | 80.0% | **94.7%** | 1.18x |
| **9** | 823 | 25 | **3.0%** | 0.27x | 90.0% | **97.4%** | 1.08x |
| **10** | 824 | 24 | **2.9%** | 0.26x | 100.0% | **100.0%** | 1.00x |

---

## 🚀 Quickstart & Installation

### 1. Requirements & Setup
```bash
# Clone and enter project workspace
cd "Bank Marketing: Term Deposit Subscription"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Model Training & Pipeline
```bash
python3 -m src.train
```

### 3. Launch Interactive Streamlit Dashboard
To run on your dedicated, unique project host port (**8592**):
```bash
streamlit run app.py --server.port 8592 --server.address 0.0.0.0
```
Then navigate to: **`http://localhost:8592`**

---

## 📂 Repository Architecture

```
.
├── PRD.md                                 # Comprehensive Product Requirement Document
├── README.md                              # This documentation
├── bank_marketing_classification.ipynb    # Executable Jupyter Notebook covering all 16 Frames
├── app.py                                 # Interactive Streamlit Web Application (Port 8592)
├── build_notebook.py                      # Notebook generator
├── idea_bank_marketing.md                 # Initial 16-frame blueprint and strategy specification
├── data/
│   └── bank-additional-full.csv           # Raw UCI Dataset (41,188 rows, 21 columns)
├── src/
│   ├── data_loader.py                     # Data loading, validation, and deduplication
│   ├── features.py                        # Leakage-free feature engineering and ColumnTransformer
│   ├── metrics.py                         # Decile gains, lift, and profit optimization engine
│   └── train.py                           # Full pipeline: 5-fold CV, benchmarking, explainability
└── artifacts/
    ├── best_bank_model.joblib             # Serialized production XGBoost pipeline
    ├── decile_gains_table.csv             # Decile conversion and cumulative capture table
    ├── model_comparison.csv               # Classifier leaderboard across PR-AUC, ROC-AUC, Lift
    ├── imbalance_comparison.csv           # Performance across imbalance mitigation techniques
    ├── leakage_comparison.json            # Empirical verification of the duration leakage impact
    ├── profit_curve.csv                   # Financial evaluation across 100 probability thresholds
    ├── feature_importance.csv             # Permutation feature importances
    ├── test_leads_scored.csv              # 8,236 test leads with predicted probabilities & deciles
    ├── eda_summary.png                    # Multi-panel exploratory data visual
    ├── roc_pr_curves.png                  # ROC and Precision-Recall comparison plots
    ├── gains_lift_curves.png              # Cumulative gains and decile lift charts
    ├── profit_curve.png                   # Campaign net profit curve
    ├── feature_importance.png             # Permutation feature importance chart
    └── leakage_comparison.png             # Production vs duration benchmark visual
```

---

## 💡 Practical Campaign Recommendations

1. **Focus Exclusively on Top 2 Deciles:**
   Calling just the top 20% of ranked prospects achieves **65.7% of total potential conversions**, cutting 80% of calling expenditures.
2. **Prioritize Previous Campaign Successes (`poutcome = 'success'`):**
   Prior converters have an extraordinary **65.1% conversion rate**. Assign these directly to premier personal banking advisors.
3. **Hard Cap Outbound Contacts at 3 Calls:**
   Conversion rate drops from 13.0% on call 1 down to <5% past 3 calls. Repetitive cold calls waste money and cause severe brand fatigue.
4. **Monitor Euribor Rate Environment:**
   Conversions surge when interbank interest rates are low (<1.5%), because retail clients aggressively seek higher guaranteed yield in term deposit products.
5. **Mobile First Contact:**
   Cellular contact methods yield **3x higher response** rates compared to landline telephone calls.
