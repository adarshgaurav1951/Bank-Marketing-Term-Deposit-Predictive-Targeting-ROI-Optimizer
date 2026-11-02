"""
Evaluation metrics module focusing on marketing campaign effectiveness:
- Decile gains and lift tables
- Cumulative gains and lift curves
- Unit economics profit curves under cost/revenue assumptions
- Standard classification metrics (PR-AUC, ROC-AUC, F1, Recall, Precision, Brier)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

def compute_ml_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard machine learning metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold)
    }

def compute_gains_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Construct decile gains and lift table.
    Ranks prospects by predicted probability descending.
    """
    df = pd.DataFrame({
        'y': np.asarray(y_true).astype(int),
        'prob': np.asarray(y_prob)
    }).sort_values('prob', ascending=False).reset_index(drop=True)
    
    # Assign deciles (1 = top 10% highest predicted probability, 10 = lowest)
    df['decile'] = pd.qcut(np.arange(len(df)), bins, labels=False) + 1
    
    overall_mean = df['y'].mean()
    total_positives = df['y'].sum()
    total_count = len(df)
    
    grouped = df.groupby('decile').agg(
        contacts=('y', 'count'),
        subscribers=('y', 'sum'),
        min_prob=('prob', 'min'),
        max_prob=('prob', 'max'),
        avg_prob=('prob', 'mean')
    ).reset_index()
    
    grouped['conversion_rate'] = grouped['subscribers'] / grouped['contacts']
    grouped['decile_lift'] = grouped['conversion_rate'] / overall_mean
    
    grouped['cum_contacts'] = grouped['contacts'].cumsum()
    grouped['cum_subscribers'] = grouped['subscribers'].cumsum()
    grouped['cum_contact_pct'] = (grouped['cum_contacts'] / total_count) * 100.0
    grouped['cum_gains_pct'] = (grouped['cum_subscribers'] / total_positives) * 100.0
    grouped['cum_conversion_rate'] = grouped['cum_subscribers'] / grouped['cum_contacts']
    grouped['cum_lift'] = grouped['cum_conversion_rate'] / overall_mean
    
    return grouped

def calculate_campaign_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_per_call: float = 5.0,
    revenue_per_sub: float = 50.0,
    num_thresholds: int = 100
) -> pd.DataFrame:
    """
    Calculate campaign net profit across thresholds.
    Profit = (True Positives * Revenue) - ((True Positives + False Positives) * Cost)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    records = []
    
    total_leads = len(y_true)
    total_positives = int(np.sum(y_true))
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        calls_made = int(np.sum(y_pred))
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        call_cost = calls_made * cost_per_call
        gross_revenue = tp * revenue_per_sub
        net_profit = gross_revenue - call_cost
        
        conversion_rate = (tp / calls_made) if calls_made > 0 else 0.0
        capture_rate = (tp / total_positives) if total_positives > 0 else 0.0
        pct_called = (calls_made / total_leads) * 100.0
        
        records.append({
            "threshold": thresh,
            "calls_made": calls_made,
            "pct_called": pct_called,
            "tp": tp,
            "fp": fp,
            "net_profit": net_profit,
            "conversion_rate": conversion_rate,
            "subscribers_captured_pct": capture_rate * 100.0,
            "cost": call_cost,
            "revenue": gross_revenue
        })
        
    df_profit = pd.DataFrame(records)
    return df_profit

def find_optimal_profit_threshold(df_profit: pd.DataFrame) -> Dict[str, Any]:
    """Identify threshold and metrics yielding maximum net profit."""
    best_row = df_profit.loc[df_profit['net_profit'].idxmax()]
    return best_row.to_dict()
