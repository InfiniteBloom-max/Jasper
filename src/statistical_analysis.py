"""
JASPER: Statistical Analysis Module

Performs comparative analysis between Japanese and Sri Lankan textile features:
- Independent t-tests
- Cohen's d effect size calculation
- Multiple comparison correction (Benjamini-Hochberg)
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size
    
    Effect size interpretation (Cohen, 1988):
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8
    - Very large: d > 1.2
    
    Args:
        group1: First group data (numpy array or pandas Series)
        group2: Second group data
        
    Returns:
        float: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg procedure for multiple testing correction
    
    Args:
        p_values: List or array of p-values
        alpha: Family-wise error rate (default: 0.05)
        
    Returns:
        Array of boolean values indicating significance after correction
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate critical values
    critical_values = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest i where p(i) <= (i/n)*alpha
    significant = sorted_p <= critical_values
    
    # Create result array
    result = np.zeros(n, dtype=bool)
    if np.any(significant):
        max_significant_idx = np.where(significant)[0][-1]
        result[sorted_indices[:max_significant_idx + 1]] = True
    
    return result


def compare_textile_groups(df, group1_label='japanese_textiles', group2_label='sri_lankan_textiles', 
                          alpha=0.05, apply_correction=True):
    """
    Comprehensive statistical comparison between two textile groups
    
    Args:
        df: DataFrame containing features and labels
        group1_label: Label for first group (default: 'japanese_textiles')
        group2_label: Label for second group (default: 'sri_lankan_textiles')
        alpha: Significance level (default: 0.05)
        apply_correction: Whether to apply Benjamini-Hochberg correction
        
    Returns:
        DataFrame with statistical comparison results
    """
    # Separate groups
    group1 = df[df['label'] == group1_label]
    group2 = df[df['label'] == group2_label]
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in ['filepath', 'label']]
    
    results = []
    
    for feature in feature_cols:
        # Extract feature values
        values1 = group1[feature].dropna()
        values2 = group2[feature].dropna()
        
        # Skip if insufficient data
        if len(values1) < 2 or len(values2) < 2:
            continue
        
        # Descriptive statistics
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        std1 = np.std(values1, ddof=1)
        std2 = np.std(values2, ddof=1)
        
        # Independent t-test
        t_stat, p_value = ttest_ind(values1, values2, equal_var=False)  # Welch's t-test
        
        # Cohen's d effect size
        cohens_d = calculate_cohens_d(values1, values2)
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size_interp = 'negligible'
        elif abs(cohens_d) < 0.5:
            effect_size_interp = 'small'
        elif abs(cohens_d) < 0.8:
            effect_size_interp = 'medium'
        elif abs(cohens_d) < 1.2:
            effect_size_interp = 'large'
        else:
            effect_size_interp = 'very large'
        
        results.append({
            'feature': feature,
            f'{group1_label}_mean': mean1,
            f'{group1_label}_std': std1,
            f'{group2_label}_mean': mean2,
            f'{group2_label}_std': std2,
            'mean_difference': mean1 - mean2,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size_interp,
            'n_group1': len(values1),
            'n_group2': len(values2)
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction if requested
    if apply_correction:
        corrected_sig = benjamini_hochberg_correction(results_df['p_value'].values, alpha=alpha)
        results_df['significant_after_correction'] = corrected_sig
        results_df['significant_uncorrected'] = results_df['p_value'] < alpha
    else:
        results_df['significant'] = results_df['p_value'] < alpha
    
    # Sort by effect size (absolute value)
    results_df['abs_cohens_d'] = abs(results_df['cohens_d'])
    results_df = results_df.sort_values('abs_cohens_d', ascending=False)
    
    return results_df


def generate_summary_statistics(df, label_col='label'):
    """
    Generate summary statistics for all features by group
    
    Args:
        df: DataFrame with features
        label_col: Column name for group labels
        
    Returns:
        DataFrame with summary statistics
    """
    feature_cols = [col for col in df.columns if col not in ['filepath', label_col]]
    
    summary = df.groupby(label_col)[feature_cols].agg(['mean', 'std', 'min', 'max', 'median'])
    
    return summary


def identify_key_differentiators(comparison_df, effect_size_threshold=1.2, p_value_threshold=0.001):
    """
    Identify features with very large effect sizes and high significance
    
    Args:
        comparison_df: DataFrame from compare_textile_groups()
        effect_size_threshold: Minimum |Cohen's d| to consider (default: 1.2 for "very large")
        p_value_threshold: Maximum p-value to consider (default: 0.001)
        
    Returns:
        DataFrame with key differentiating features
    """
    key_features = comparison_df[
        (abs(comparison_df['cohens_d']) >= effect_size_threshold) &
        (comparison_df['p_value'] < p_value_threshold)
    ].copy()
    
    return key_features


def generate_statistical_report(comparison_df, output_file=None):
    """
    Generate a comprehensive statistical report
    
    Args:
        comparison_df: DataFrame from compare_textile_groups()
        output_file: Optional path to save report as text file
        
    Returns:
        String containing the formatted report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("JASPER: Statistical Analysis Report")
    report_lines.append("Japanese x Sri Lankan Textile Design Comparison")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall summary
    total_features = len(comparison_df)
    significant_features = comparison_df['significant_after_correction'].sum() if 'significant_after_correction' in comparison_df.columns else comparison_df['significant'].sum()
    
    report_lines.append(f"Total features analyzed: {total_features}")
    report_lines.append(f"Statistically significant features (after correction): {significant_features}")
    report_lines.append(f"Percentage significant: {significant_features/total_features*100:.1f}%")
    report_lines.append("")
    
    # Effect size distribution
    very_large = (abs(comparison_df['cohens_d']) >= 1.2).sum()
    large = ((abs(comparison_df['cohens_d']) >= 0.8) & (abs(comparison_df['cohens_d']) < 1.2)).sum()
    medium = ((abs(comparison_df['cohens_d']) >= 0.5) & (abs(comparison_df['cohens_d']) < 0.8)).sum()
    small = ((abs(comparison_df['cohens_d']) >= 0.2) & (abs(comparison_df['cohens_d']) < 0.5)).sum()
    negligible = (abs(comparison_df['cohens_d']) < 0.2).sum()
    
    report_lines.append("Effect Size Distribution:")
    report_lines.append(f"  Very Large (|d| >= 1.2): {very_large} features ({very_large/total_features*100:.1f}%)")
    report_lines.append(f"  Large (0.8 <= |d| < 1.2): {large} features ({large/total_features*100:.1f}%)")
    report_lines.append(f"  Medium (0.5 <= |d| < 0.8): {medium} features ({medium/total_features*100:.1f}%)")
    report_lines.append(f"  Small (0.2 <= |d| < 0.5): {small} features ({small/total_features*100:.1f}%)")
    report_lines.append(f"  Negligible (|d| < 0.2): {negligible} features ({negligible/total_features*100:.1f}%)")
    report_lines.append("")
    
    # Top 10 differentiating features
    report_lines.append("Top 10 Differentiating Features (by |Cohen's d|):")
    report_lines.append("-" * 80)
    
    top_10 = comparison_df.head(10)
    for idx, row in top_10.iterrows():
        sig_marker = "***" if row.get('significant_after_correction', row.get('significant', False)) else ""
        report_lines.append(f"{row['feature']:30s} | d={row['cohens_d']:7.3f} | p={row['p_value']:.4e} {sig_marker}")
    
    report_lines.append("")
    
    # Key findings (very large effect sizes)
    key_features = comparison_df[abs(comparison_df['cohens_d']) >= 1.2]
    if len(key_features) > 0:
        report_lines.append(f"Key Differentiators (Very Large Effect Size, |d| >= 1.2): {len(key_features)} features")
        report_lines.append("-" * 80)
        
        for idx, row in key_features.iterrows():
            sig_marker = "***" if row.get('significant_after_correction', row.get('significant', False)) else ""
            col1 = comparison_df.columns[1]  # First group mean column
            col2 = comparison_df.columns[3]  # Second group mean column
            report_lines.append(f"{row['feature']:30s}")
            report_lines.append(f"  Cohen's d: {row['cohens_d']:.3f} ({row['effect_size']})")
            report_lines.append(f"  p-value: {row['p_value']:.4e} {sig_marker}")
            report_lines.append(f"  {col1}: {row[col1]:.3f} ± {row[comparison_df.columns[2]]:.3f}")
            report_lines.append(f"  {col2}: {row[col2]:.3f} ± {row[comparison_df.columns[4]]:.3f}")
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    
    return report_text


if __name__ == "__main__":
    print("JASPER Statistical Analysis Module")
    print("Use: from statistical_analysis import compare_textile_groups")
