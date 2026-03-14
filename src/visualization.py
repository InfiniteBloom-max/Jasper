"""
JASPER: Visualization Module

Generates publication-quality visualizations for textile analysis:
- Color palette comparisons
- Distribution plots
- Effect size visualizations
- Feature importance charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("Set2")


def plot_color_palette_comparison(df, group1_label='japanese_textiles', 
                                  group2_label='sri_lankan_textiles', 
                                  n_samples=10, save_path=None):
    """
    Visualize dominant color palettes for both textile groups
    
    Args:
        df: DataFrame with color features
        group1_label: First group label
        group2_label: Second group label
        n_samples: Number of samples to display per group
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    for idx, (label, ax) in enumerate(zip([group1_label, group2_label], axes)):
        group_data = df[df['label'] == label].sample(n=min(n_samples, len(df[df['label'] == label])), 
                                                      random_state=42)
        
        # Extract dominant colors for each sample
        for i, (_, row) in enumerate(group_data.iterrows()):
            # Get top 5 dominant colors
            for j in range(5):
                r = row[f'dominant_color_{j+1}_r'] / 255.0
                g = row[f'dominant_color_{j+1}_g'] / 255.0
                b = row[f'dominant_color_{j+1}_b'] / 255.0
                
                rect = Rectangle((j, i), 1, 1, facecolor=(r, g, b), edgecolor='white', linewidth=1)
                ax.add_patch(rect)
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, n_samples)
        ax.set_xlabel('Dominant Colors (Rank 1-5)')
        ax.set_ylabel('Sample Index')
        ax.set_title(f'{label.replace("_", " ").title()} - Color Palettes', fontweight='bold')
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved color palette comparison to {save_path}")
    
    plt.show()
    return fig


def plot_feature_distributions(df, features, group1_label='japanese_textiles', 
                               group2_label='sri_lankan_textiles', save_path=None):
    """
    Plot distribution comparisons for key features
    
    Args:
        df: DataFrame with features
        features: List of feature names to plot
        group1_label: First group label
        group2_label: Second group label
        save_path: Optional path to save figure
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Get data for both groups
        group1_data = df[df['label'] == group1_label][feature].dropna()
        group2_data = df[df['label'] == group2_label][feature].dropna()
        
        # Plot distributions
        ax.hist(group1_data, bins=30, alpha=0.6, label=group1_label.replace('_', ' ').title(), 
                color='steelblue', density=True)
        ax.hist(group2_data, bins=30, alpha=0.6, label=group2_label.replace('_', ' ').title(), 
                color='coral', density=True)
        
        # Add mean lines
        ax.axvline(group1_data.mean(), color='steelblue', linestyle='--', linewidth=2, 
                   label=f'Mean: {group1_data.mean():.3f}')
        ax.axvline(group2_data.mean(), color='coral', linestyle='--', linewidth=2, 
                   label=f'Mean: {group2_data.mean():.3f}')
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution: {feature.replace("_", " ").title()}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved distribution plots to {save_path}")
    
    plt.show()
    return fig


def plot_effect_sizes(comparison_df, top_n=20, save_path=None):
    """
    Visualize effect sizes (Cohen's d) for top features
    
    Args:
        comparison_df: DataFrame from statistical_analysis.compare_textile_groups()
        top_n: Number of top features to display
        save_path: Optional path to save figure
    """
    # Get top features by absolute effect size
    top_features = comparison_df.head(top_n).copy()
    top_features = top_features.sort_values('cohens_d')
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Create color map based on effect size magnitude
    colors = ['coral' if d < 0 else 'steelblue' for d in top_features['cohens_d']]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['cohens_d'], color=colors, alpha=0.7)
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    
    # Add effect size reference lines
    for threshold, label, style in [(0.2, 'Small', ':'), (0.5, 'Medium', '--'), 
                                     (0.8, 'Large', '-.'), (1.2, 'Very Large', '-')]:
        ax.axvline(threshold, color='gray', linewidth=0.8, linestyle=style, alpha=0.5)
        ax.axvline(-threshold, color='gray', linewidth=0.8, linestyle=style, alpha=0.5)
    
    # Labels and formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']], fontsize=9)
    ax.set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
    ax.set_title(f"Top {top_n} Features by Effect Size", fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Japanese > Sri Lankan'),
        Patch(facecolor='coral', alpha=0.7, label='Sri Lankan > Japanese')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved effect size plot to {save_path}")
    
    plt.show()
    return fig


def plot_pvalue_volcano(comparison_df, save_path=None):
    """
    Create volcano plot showing effect size vs statistical significance
    
    Args:
        comparison_df: DataFrame from statistical_analysis.compare_textile_groups()
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    cohens_d = comparison_df['cohens_d'].values
    neg_log_p = -np.log10(comparison_df['p_value'].values)
    
    # Significance after correction
    if 'significant_after_correction' in comparison_df.columns:
        significant = comparison_df['significant_after_correction'].values
    else:
        significant = comparison_df['significant'].values
    
    # Color points by significance
    colors = ['red' if sig else 'gray' for sig in significant]
    sizes = [50 if sig else 20 for sig in significant]
    
    # Scatter plot
    scatter = ax.scatter(cohens_d, neg_log_p, c=colors, s=sizes, alpha=0.6)
    
    # Add reference lines
    ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax.axhline(-np.log10(0.05), color='blue', linewidth=1, linestyle='--', 
               label='p = 0.05', alpha=0.5)
    ax.axhline(-np.log10(0.001), color='darkblue', linewidth=1, linestyle='--', 
               label='p = 0.001', alpha=0.5)
    
    # Effect size thresholds
    for threshold in [0.8, 1.2]:
        ax.axvline(threshold, color='green', linewidth=0.8, linestyle=':', alpha=0.4)
        ax.axvline(-threshold, color='green', linewidth=0.8, linestyle=':', alpha=0.4)
    
    # Labels
    ax.set_xlabel("Cohen's d (Effect Size)", fontweight='bold', fontsize=11)
    ax.set_ylabel('-log10(p-value)', fontweight='bold', fontsize=11)
    ax.set_title('Volcano Plot: Effect Size vs Statistical Significance', 
                 fontweight='bold', fontsize=13)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Significant (after correction)'),
        Patch(facecolor='gray', alpha=0.6, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved volcano plot to {save_path}")
    
    plt.show()
    return fig


def plot_correlation_heatmap(df, features=None, save_path=None):
    """
    Create correlation heatmap for selected features
    
    Args:
        df: DataFrame with features
        features: List of features to include (default: all numeric features)
        save_path: Optional path to save figure
    """
    if features is None:
        features = [col for col in df.columns if col not in ['filepath', 'label'] and df[col].dtype in ['float64', 'int64']]
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved correlation heatmap to {save_path}")
    
    plt.show()
    return fig


def plot_rgb_comparison(df, group1_label='japanese_textiles', 
                       group2_label='sri_lankan_textiles', save_path=None):
    """
    Compare average RGB profiles between groups
    
    Args:
        df: DataFrame with color features
        group1_label: First group label
        group2_label: Second group label
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate average RGB for each group
    group1_rgb = df[df['label'] == group1_label][['avg_r', 'avg_g', 'avg_b']].mean()
    group2_rgb = df[df['label'] == group2_label][['avg_r', 'avg_g', 'avg_b']].mean()
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, group1_rgb, width, label=group1_label.replace('_', ' ').title(),
                   color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, group2_rgb, width, label=group2_label.replace('_', ' ').title(),
                   color='coral', alpha=0.7)
    
    ax.set_ylabel('Average Intensity (0-255)', fontweight='bold')
    ax.set_title('Average RGB Profile Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Red', 'Green', 'Blue'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved RGB comparison to {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("JASPER Visualization Module")
    print("Use: from visualization import plot_*")
