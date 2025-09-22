import pandas as pd
from typing import Dict, Any


def calculate_roc_metrics(df: pd.DataFrame, prediction_col: str, target_col: str) -> Dict[str, Any]:
    """
    Calculate ROC metrics including AUC, Gini coefficient, and KS statistic.
    
    Args:
        df (pd.DataFrame): DataFrame containing predictions, targets, and weights
        prediction_col (str): Column name for model predictions
        target_col (str): Column name for actual binary outcomes (0/1)
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'auc': Area Under Curve (0-100 scale)
            - 'gini': Gini coefficient 
            - 'ks': Kolmogorov-Smirnov statistic (0-100 scale)
    """
    # Create probability distribution table
    pdf = pd.crosstab(
        df[prediction_col], 
        df[target_col], 
        df['w'], 
        aggfunc='sum', 
        normalize='columns'
    ).rename_axis(None, axis=1)
    
    # Calculate AUC
    auc = ((pdf[1].cumsum() - pdf[1] / 2) * pdf[0]).sum() * 100
    auc = auc if auc > 50 else 100 - auc
    
    # Calculate Gini coefficient
    gini = 2 * auc - 100
    
    # Calculate cumulative distribution (needed for KS statistic)
    cdf = (pdf if auc > 50 else pdf[::-1]).cumsum()
    
    # Calculate KS statistic
    ks = (abs(cdf[1] - cdf[0])).max() * 100
    
    return {
        'auc': float(round(auc,3)),
        'gini': float(round(gini,3)),
        'ks': float(round(ks,3))
    }


# Example usage:
if __name__ == "__main__":
    # Sample data
    import numpy as np
    
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'predictions': np.random.rand(100),
        'actual': np.random.choice([0, 1], 100),
        'w': np.random.uniform(0.5, 2.0, 100)
    })
    
    # Calculate metrics
    metrics = calculate_roc_metrics(sample_df, 'predictions', 'actual')
    
    # Access results
    print(f"AUC: {metrics['auc']:.2f}")
    print(f"Gini: {metrics['gini']:.2f}")
    print(f"KS: {metrics['ks']:.2f}")