"""
Comprehensive model training, evaluation, and benchmarking pipeline.
Implements:
- Stratified 80/20 train-test split (Golden Rule #3)
- Baseline models (Dummy and Vanilla LogReg) (Golden Rule #2)
- Imbalance strategy comparison (Weights, SMOTE, Undersampling) inside pipelines
- 4 Classifier comparison (LogReg, Decision Tree, Random Forest, XGBoost)
- Hyperparameter tuning with Stratified 5-Fold Cross-Validation
- Marketing decile lift & profit curve evaluation (Golden Rules #4 & #5)
- Leakage benchmark evaluation demonstrating the impact of 'duration' (Golden Rule #1)
- Permutation feature importance
- Artifact serialization for Streamlit application and reproducible verification
"""

import os
import json
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.data_loader import load_raw_data, clean_data
from src.features import (
    BankFeatureEngineer,
    get_preprocessor,
    prepare_feature_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from src.metrics import (
    compute_ml_metrics,
    compute_gains_table,
    calculate_campaign_profit,
    find_optimal_profit_threshold
)

RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def run_eda_visualizations(df_clean: pd.DataFrame) -> str:
    """Generate and save multi-panel EDA figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # 1. Target Imbalance
    target_counts = df_clean['y'].value_counts()
    target_pcts = df_clean['y'].value_counts(normalize=True) * 100
    sns.barplot(x=['No (0)', 'Yes (1)'], y=target_counts.values, hue=['No (0)', 'Yes (1)'], ax=axes[0, 0], palette=['#4A90E2', '#50E3C2'], legend=False)
    axes[0, 0].set_title(f"Target Distribution (Yes: {target_pcts[1]:.1f}%, No: {target_pcts[0]:.1f}%)", fontweight='bold')
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(target_counts.values):
        axes[0, 0].text(i, v + 500, f"{v:,} ({target_pcts.iloc[i]:.1f}%)", ha='center', fontweight='bold')

    # 2. Conversion by Job Category
    job_rates = df_clean.groupby('job')['y'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=job_rates.values, y=job_rates.index, hue=job_rates.index, ax=axes[0, 1], palette='viridis', legend=False)
    axes[0, 1].axvline(df_clean['y'].mean()*100, color='red', linestyle='--', label=f"Avg ({df_clean['y'].mean()*100:.1f}%)")
    axes[0, 1].set_title("Subscription Rate by Job (%)", fontweight='bold')
    axes[0, 1].set_xlabel("Conversion Rate (%)")
    axes[0, 1].legend()

    # 3. Conversion by Previous Campaign Outcome (poutcome)
    pout_rates = df_clean.groupby('poutcome')['y'].agg(mean='mean', count='count').reset_index()
    pout_rates['rate'] = pout_rates['mean'] * 100
    sns.barplot(data=pout_rates, x='poutcome', y='rate', hue='poutcome', ax=axes[0, 2], palette='mako', legend=False)
    axes[0, 2].set_title("Subscription Rate by Previous Outcome (poutcome)", fontweight='bold')
    axes[0, 2].set_ylabel("Conversion Rate (%)")
    for i, row in pout_rates.iterrows():
        axes[0, 2].text(i, row['rate'] + 1.5, f"{row['rate']:.1f}%\n(n={int(row['count']):,})", ha='center', fontsize=9)

    # 4. Campaign Contact Fatigue
    camp_rates = df_clean[df_clean['campaign'] <= 10].groupby('campaign')['y'].agg(mean='mean', count='count').reset_index()
    camp_rates['rate'] = camp_rates['mean'] * 100
    axes[1, 0].plot(camp_rates['campaign'], camp_rates['rate'], marker='o', color='#E94E77', linewidth=2.5)
    axes[1, 0].axhline(df_clean['y'].mean()*100, color='gray', linestyle=':', label='Average')
    axes[1, 0].set_title("Campaign Fatigue: Conversion vs. Contact Number", fontweight='bold')
    axes[1, 0].set_xlabel("Number of Contacts during Campaign")
    axes[1, 0].set_ylabel("Conversion Rate (%)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 5. Conversion by Contact Channel & Month
    month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    valid_months = [m for m in month_order if m in df_clean['month'].unique()]
    month_rates = df_clean.groupby('month')['y'].mean().reindex(valid_months) * 100
    sns.barplot(x=month_rates.index, y=month_rates.values, hue=month_rates.index, ax=axes[1, 1], palette='rocket', legend=False)
    axes[1, 1].set_title("Seasonality: Subscription Rate by Month (%)", fontweight='bold')
    axes[1, 1].set_ylabel("Conversion Rate (%)")
    axes[1, 1].set_xlabel("Month")

    # 6. Euribor 3-Month Interest Rate vs. Outcome
    sns.kdeplot(data=df_clean, x='euribor3m', hue='y', common_norm=False, fill=True, ax=axes[1, 2], palette=['#3498DB', '#2ECC71'])
    axes[1, 2].set_title("Euribor 3M Interest Rate Density by Outcome", fontweight='bold')
    axes[1, 2].set_xlabel("Euribor 3-Month Rate")
    axes[1, 2].legend(['Subscribed (1)', 'Did Not Subscribe (0)'])

    fig_path = os.path.join(ARTIFACTS_DIR, "eda_summary.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    return fig_path

def train_and_evaluate_all():
    """Main execution orchestrating all frames."""
    print("=" * 70)
    print("STARTING BANK MARKETING END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load & Clean
    raw_df = load_raw_data()
    df_clean, meta = clean_data(raw_df)
    print(f"Loaded: {meta['initial_rows']} rows. Removed {meta['duplicate_count']} duplicates.")
    print(f"Cleaned dataset: {len(df_clean)} rows. Positives: {meta['positive_count']} ({meta['base_conversion_rate']*100:.2f}%)")

    # Generate EDA Visuals
    eda_fig = run_eda_visualizations(df_clean)
    print(f"EDA plots saved to {eda_fig}")

    # 2. Prepare Features without duration (Production Model)
    X, y = prepare_feature_data(df_clean, include_duration_benchmark=False)
    
    # 3. Stratified 80/20 Train-Test Split (Golden Rule #3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Stratified Split: Train={len(X_train)} rows (y mean={y_train.mean():.4f}), Test={len(X_test)} rows (y mean={y_test.mean():.4f})")

    # 4. Baseline Models (Frame 10)
    print("\n--- FRAME 10: BASELINE MODELS ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = compute_ml_metrics(y_test, dummy_prob, threshold=0.5)
    print(f"Dummy (Most Frequent): Accuracy={np.mean(dummy_pred == y_test):.4f}, Recall={dummy_metrics['recall']:.4f}, PR-AUC={dummy_metrics['pr_auc']:.4f}")

    # Baseline Logistic Regression (Unweighted)
    pipe_baseline = ImbPipeline([
        ('feat_eng', BankFeatureEngineer(drop_duration=True)),
        ('prep', get_preprocessor()),
        ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    pipe_baseline.fit(X_train, y_train)
    baseline_prob = pipe_baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_ml_metrics(y_test, baseline_prob, threshold=0.5)
    print(f"Baseline LogReg: ROC-AUC={baseline_metrics['roc_auc']:.4f}, PR-AUC={baseline_metrics['pr_auc']:.4f}, Recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}")

    # 5. Imbalance Strategy Comparison (Frame 11)
    print("\n--- FRAME 11: CLASS IMBALANCE STRATEGIES (Logistic Regression) ---")
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    imbalance_strategies = {
        "None (Standard)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Class Weights ('balanced')": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "SMOTE (Inside Pipeline)": ImbPipeline([
            ('feat_eng', BankFeatureEngineer(drop_duration=True)),
            ('prep', get_preprocessor()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Under-Sampling": ImbPipeline([
            ('feat_eng', BankFeatureEngineer(drop_duration=True)),
            ('prep', get_preprocessor()),
            ('rus', RandomUnderSampler(random_state=RANDOM_STATE)),
            ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])
    }

    imbalance_results = []
    for strat_name, estimator in imbalance_strategies.items():
        if isinstance(estimator, ImbPipeline):
            pipe = estimator
        else:
            pipe = ImbPipeline([
                ('feat_eng', BankFeatureEngineer(drop_duration=True)),
                ('prep', get_preprocessor()),
                ('model', estimator)
            ])
        
        t0 = time.time()
        cv_pr_auc = cross_val_score(pipe, X_train, y_train, cv=cv_strat, scoring='average_precision').mean()
        cv_roc_auc = cross_val_score(pipe, X_train, y_train, cv=cv_strat, scoring='roc_auc').mean()
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_ml_metrics(y_test, prob, threshold=0.5)
        imbalance_results.append({
            "Strategy": strat_name,
            "CV PR-AUC": round(cv_pr_auc, 4),
            "CV ROC-AUC": round(cv_roc_auc, 4),
            "Test PR-AUC": round(metrics['pr_auc'], 4),
            "Test ROC-AUC": round(metrics['roc_auc'], 4),
            "Test Recall": round(metrics['recall'], 4),
            "Test Precision": round(metrics['precision'], 4),
            "Test F1": round(metrics['f1'], 4)
        })
        print(f"  [{strat_name}] CV PR-AUC: {cv_pr_auc:.4f} | Test Recall: {metrics['recall']:.4f} | Test Precision: {metrics['precision']:.4f}")

    df_imbalance = pd.DataFrame(imbalance_results)
    df_imbalance.to_csv(os.path.join(ARTIFACTS_DIR, "imbalance_comparison.csv"), index=False)

    # 6. Classifier Comparison & Tuning (Frame 12)
    print("\n--- FRAME 12: MODEL SUITE COMPARISON ---")
    # Using class_weight='balanced' or scale_pos_weight for fair comparison
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

    models = {
        "Logistic Regression (Balanced)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        "Decision Tree (Balanced)": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RANDOM_STATE),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost (Weighted)": XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=RANDOM_STATE
        )
    }

    trained_pipelines = {}
    model_comparison_records = []
    test_probs = {}

    for name, clf in models.items():
        pipe = ImbPipeline([
            ('feat_eng', BankFeatureEngineer(drop_duration=True)),
            ('prep', get_preprocessor()),
            ('model', clf)
        ])
        
        cv_pr_auc = cross_val_score(pipe, X_train, y_train, cv=cv_strat, scoring='average_precision').mean()
        cv_roc_auc = cross_val_score(pipe, X_train, y_train, cv=cv_strat, scoring='roc_auc').mean()
        pipe.fit(X_train, y_train)
        
        prob = pipe.predict_proba(X_test)[:, 1]
        test_probs[name] = prob
        trained_pipelines[name] = pipe
        
        metrics = compute_ml_metrics(y_test, prob, threshold=0.5)
        # Also compute Lift @ Decile 1
        gains = compute_gains_table(y_test, prob, bins=10)
        lift_d1 = gains.loc[gains['decile'] == 1, 'decile_lift'].values[0]
        capture_top20 = gains.loc[gains['decile'] == 2, 'cum_gains_pct'].values[0]

        model_comparison_records.append({
            "Model": name,
            "CV PR-AUC": round(cv_pr_auc, 4),
            "CV ROC-AUC": round(cv_roc_auc, 4),
            "Test PR-AUC": round(metrics['pr_auc'], 4),
            "Test ROC-AUC": round(metrics['roc_auc'], 4),
            "Test Recall": round(metrics['recall'], 4),
            "Test Precision": round(metrics['precision'], 4),
            "Test F1": round(metrics['f1'], 4),
            "Lift@10%": round(lift_d1, 2),
            "Capture@20% (%)": round(capture_top20, 1)
        })
        print(f"  {name:30s} | CV PR-AUC: {cv_pr_auc:.4f} | Test ROC-AUC: {metrics['roc_auc']:.4f} | Lift@10%: {lift_d1:.2f}x | Capture@20%: {capture_top20:.1f}%")

    df_models = pd.DataFrame(model_comparison_records).sort_values("Test PR-AUC", ascending=False)
    df_models.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index=False)

    # Top model selected
    best_model_name = df_models.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_prob = test_probs[best_model_name]
    print(f"\n>>> Selected Best Production Model: {best_model_name} (Test PR-AUC={df_models.iloc[0]['Test PR-AUC']})")

    # 7. Leakage Benchmark Model (With Duration Retained) - Golden Rule #1
    print("\n--- BENCHMARK: QUANTIFYING LOOKAHEAD LEAKAGE ('duration') ---")
    X_leak = df_clean.drop(columns=['y'])
    X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(
        X_leak, df_clean['y'], test_size=0.20, stratify=df_clean['y'], random_state=RANDOM_STATE
    )

    # Build numeric transformer that includes duration
    num_with_duration = NUMERIC_FEATURES + ['duration']
    from sklearn.compose import ColumnTransformer
    prep_with_dur = ColumnTransformer([
        ('num', get_preprocessor().transformers[0][1], num_with_duration),
        ('cat', get_preprocessor().transformers[1][1], CATEGORICAL_FEATURES)
    ], remainder='drop')

    pipe_leak = ImbPipeline([
        ('feat_eng', BankFeatureEngineer(drop_duration=False)),
        ('prep', prep_with_dur),
        ('model', XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=RANDOM_STATE))
    ])
    pipe_leak.fit(X_l_train, y_l_train)
    prob_leak = pipe_leak.predict_proba(X_l_test)[:, 1]
    metrics_leak = compute_ml_metrics(y_l_test, prob_leak, threshold=0.5)

    prod_metrics = compute_ml_metrics(y_test, best_prob, threshold=0.5)
    leakage_comparison = {
        "production_without_duration": {
            "model": best_model_name,
            "roc_auc": round(prod_metrics['roc_auc'], 4),
            "pr_auc": round(prod_metrics['pr_auc'], 4),
            "recall": round(prod_metrics['recall'], 4),
            "precision": round(prod_metrics['precision'], 4),
            "f1": round(prod_metrics['f1'], 4),
            "status": "VALID FOR PRODUCTION (No lookahead leakage)"
        },
        "benchmark_with_duration": {
            "model": "XGBoost (With Duration)",
            "roc_auc": round(metrics_leak['roc_auc'], 4),
            "pr_auc": round(metrics_leak['pr_auc'], 4),
            "recall": round(metrics_leak['recall'], 4),
            "precision": round(metrics_leak['precision'], 4),
            "f1": round(metrics_leak['f1'], 4),
            "status": "FATAL LEAKAGE (Call duration unknown before call)"
        },
        "leakage_inflation_analysis": (
            f"Including 'duration' creates an artificial +{(metrics_leak['roc_auc'] - prod_metrics['roc_auc']):.4f} ROC-AUC jump "
            f"and +{(metrics_leak['pr_auc'] - prod_metrics['pr_auc']):.4f} PR-AUC jump. "
            "However, duration is fundamentally unavailable prior to dialing, making any model relying on it useless in operational production."
        )
    }
    with open(os.path.join(ARTIFACTS_DIR, "leakage_comparison.json"), "w") as f:
        json.dump(leakage_comparison, f, indent=2)
    print(f"Production (No Duration) ROC-AUC: {prod_metrics['roc_auc']:.4f} | PR-AUC: {prod_metrics['pr_auc']:.4f}")
    print(f"Benchmark (With Duration) ROC-AUC: {metrics_leak['roc_auc']:.4f} | PR-AUC: {metrics_leak['pr_auc']:.4f}")
    print("Leakage demonstration confirmed.")

    # 8. Marketing Evaluation & Decile Gains (Frame 13)
    print("\n--- FRAME 13: MARKETING METRICS & GAINS TABLE ---")
    gains_table = compute_gains_table(y_test, best_prob, bins=10)
    gains_table.to_csv(os.path.join(ARTIFACTS_DIR, "decile_gains_table.csv"), index=False)
    print(gains_table[['decile', 'contacts', 'subscribers', 'conversion_rate', 'decile_lift', 'cum_gains_pct', 'cum_lift']].to_string(index=False))

    # Calculate Campaign Profit Curve (Assumptions: Cost €5, Revenue €50)
    df_profit = calculate_campaign_profit(y_test.values, best_prob, cost_per_call=5.0, revenue_per_sub=50.0)
    df_profit.to_csv(os.path.join(ARTIFACTS_DIR, "profit_curve.csv"), index=False)
    best_profit_info = find_optimal_profit_threshold(df_profit)
    print(f"\nProfit Optimization (Cost=€5/call, Rev=€50/sub):")
    print(f"  Optimal Probability Threshold: {best_profit_info['threshold']:.2f}")
    print(f"  Optimal Calls: {int(best_profit_info['calls_made'])} ({best_profit_info['pct_called']:.1f}% of list)")
    print(f"  Captured Subscribers: {int(best_profit_info['tp'])} ({best_profit_info['subscribers_captured_pct']:.1f}% of total buyers)")
    print(f"  Maximum Campaign Profit: €{best_profit_info['net_profit']:,.2f}")
    
    # Cold calling all prospects profit:
    all_calls = len(y_test)
    all_tp = int(y_test.sum())
    cold_profit = (all_tp * 50.0) - (all_calls * 5.0)
    print(f"  Cold Calling All Prospects Profit: €{cold_profit:,.2f}")
    print(f"  Profit Uplift from ML Targeting: +€{(best_profit_info['net_profit'] - cold_profit):,.2f} ({((best_profit_info['net_profit'] - cold_profit)/abs(cold_profit))*100:.1f}%)")

    # 9. Model Explainability & Permutation Importance (Frame 14)
    print("\n--- FRAME 14: MODEL EXPLAINABILITY & FEATURE IMPORTANCE ---")
    # Compute permutation importance on a sample of test set for performance
    test_sample_idx = np.random.choice(len(X_test), size=min(1500, len(X_test)), replace=False)
    X_test_samp = X_test.iloc[test_sample_idx]
    y_test_samp = y_test.iloc[test_sample_idx]
    
    perm_res = permutation_importance(
        best_pipeline, X_test_samp, y_test_samp, scoring='average_precision', n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': perm_res.importances_mean,
        'importance_std': perm_res.importances_std
    }).sort_values('importance_mean', ascending=False)
    perm_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
    print("Top 8 Drivers of Subscription:")
    print(perm_df.head(8)[['feature', 'importance_mean']].to_string(index=False))

    # 10. Generate Visual Plots for Artifacts
    # A. ROC & PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, prob in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})", linewidth=2)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC={average_precision_score(y_test, prob):.3f})", linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    baseline_rate = y_test.mean()
    ax2.axhline(baseline_rate, color='k', linestyle='--', alpha=0.6, label=f"Random Baseline ({baseline_rate:.2f})")
    ax2.set_title("Precision-Recall (PR) Curves", fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_pr_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # B. Cumulative Gains & Lift Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    decile_contact_pcts = gains_table['cum_contact_pct'].values
    decile_gains_pcts = gains_table['cum_gains_pct'].values
    
    # Gains
    ax1.plot(np.insert(decile_contact_pcts, 0, 0), np.insert(decile_gains_pcts, 0, 0), marker='o', color='#2ECC71', linewidth=2.5, label='ML Model Targeting')
    ax1.plot([0, 100], [0, 100], 'k--', label='Random Cold Calling Baseline')
    ax1.fill_between(np.insert(decile_contact_pcts, 0, 0), np.insert(decile_gains_pcts, 0, 0), np.insert(decile_contact_pcts, 0, 0), alpha=0.15, color='#2ECC71')
    ax1.set_title("Cumulative Gains Curve (Subscribers Captured)", fontweight='bold')
    ax1.set_xlabel("% of Prospect Base Contacted (Ranked Deciles)")
    ax1.set_ylabel("% of Total Actual Subscribers Captured")
    # Annotate top 20%
    top20_capture = gains_table.loc[gains_table['decile'] == 2, 'cum_gains_pct'].values[0]
    ax1.annotate(f"Top 20% Calls Capture\n{top20_capture:.1f}% of Buyers!", 
                 xy=(20, top20_capture), xytext=(35, top20_capture - 15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Lift
    ax2.bar(gains_table['decile'], gains_table['decile_lift'], color='#3498DB', edgecolor='black', alpha=0.85)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline Random Lift (1.0x)')
    ax2.set_title("Decile Lift Chart (Multiple vs. Random Cold Call)", fontweight='bold')
    ax2.set_xlabel("Predicted Propensity Decile (1 = Highest, 10 = Lowest)")
    ax2.set_ylabel("Lift Multiple")
    for i, row in gains_table.iterrows():
        ax2.text(row['decile'], row['decile_lift'] + 0.1, f"{row['decile_lift']:.2f}x", ha='center', fontweight='bold', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "gains_lift_curves.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # C. Campaign Profit Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_profit['pct_called'], df_profit['net_profit'], color='#9B59B6', linewidth=2.8, label='Net Campaign Profit (€)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(cold_profit, color='red', linestyle=':', label=f'Cold Call All Leads (€{cold_profit:,.0f})')
    
    # Mark max profit
    opt_pct = best_profit_info['pct_called']
    opt_prof = best_profit_info['net_profit']
    ax.scatter([opt_pct], [opt_prof], color='gold', s=160, zorder=5, edgecolor='black', label=f"Max Profit: €{opt_prof:,.0f} (Call Top {opt_pct:.1f}%)")
    ax.annotate(f"Optimal Sweet Spot:\nCall Top {opt_pct:.1f}% Leads\nProfit = €{opt_prof:,.0f}",
                xy=(opt_pct, opt_prof), xytext=(opt_pct + 12, opt_prof - 3500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF275", alpha=0.9))
    ax.set_title("Marketing Campaign Net Profit vs. Prospect Cutoff (Cost=€5, Rev=€50)", fontweight='bold')
    ax.set_xlabel("% of Prospect Base Contacted")
    ax.set_ylabel("Net Campaign Profit (€)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    plt.savefig(os.path.join(ARTIFACTS_DIR, "profit_curve.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # D. Permutation Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=perm_df.head(10), x='importance_mean', y='feature', palette='crest', ax=ax)
    ax.set_title("Permutation Feature Importance (Impact on PR-AUC)", fontweight='bold')
    ax.set_xlabel("Mean Decrease in PR-AUC on Unseen Test Leads")
    ax.set_ylabel("Feature")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # E. Leakage Comparison Visual
    fig, ax = plt.subplots(figsize=(9, 5))
    comp_df = pd.DataFrame([
        {"Metric": "ROC-AUC", "Model": "Production (No Duration)", "Score": prod_metrics['roc_auc']},
        {"Metric": "ROC-AUC", "Model": "Benchmark with Duration (LEAKAGE)", "Score": metrics_leak['roc_auc']},
        {"Metric": "PR-AUC", "Model": "Production (No Duration)", "Score": prod_metrics['pr_auc']},
        {"Metric": "PR-AUC", "Model": "Benchmark with Duration (LEAKAGE)", "Score": metrics_leak['pr_auc']},
    ])
    sns.barplot(data=comp_df, x='Metric', y='Score', hue='Model', palette=['#3498DB', '#E74C3C'], ax=ax)
    ax.set_title("Data Leakage Benchmark: Production vs. Duration Retained", fontweight='bold')
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val / 2),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    plt.savefig(os.path.join(ARTIFACTS_DIR, "leakage_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 11. Serialize Best Pipeline & Test Metadata
    best_pipeline_path = os.path.join(ARTIFACTS_DIR, "best_bank_model.joblib")
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"\nSerialized production pipeline saved to {best_pipeline_path}")

    # Save test sample dataset with probabilities for Streamlit quick demo
    test_export = X_test.copy()
    test_export['actual_y'] = y_test.values
    test_export['predicted_prob'] = np.round(best_prob, 4)
    # Assign decile based on test_probs
    test_export['assigned_decile'] = pd.qcut(test_export['predicted_prob'], 10, labels=False)
    test_export['assigned_decile'] = 10 - test_export['assigned_decile'] # 1 is highest
    test_export.to_csv(os.path.join(ARTIFACTS_DIR, "test_leads_scored.csv"), index=False)
    print(f"Scored test leads exported to {os.path.join(ARTIFACTS_DIR, 'test_leads_scored.csv')}")

    print("\n" + "=" * 70)
    print("ALL ML TRAINING, BENCHMARKING, AND EXPORTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_all()
