"""
JASPER: Main Analysis Pipeline
Japanese x Sri Lankan Textile Design Analysis

Complete reproducible pipeline for the research paper.
Executes: data loading -> feature extraction -> statistical analysis -> visualization
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_extraction import extract_dataset_features
from statistical_analysis import (
    compare_textile_groups, 
    generate_summary_statistics,
    identify_key_differentiators,
    generate_statistical_report
)
from visualization import (
    plot_color_palette_comparison,
    plot_feature_distributions,
    plot_effect_sizes,
    plot_pvalue_volcano,
    plot_rgb_comparison
)


def load_dataset_from_kaggle(dataset_path, sample_size=None, random_state=42):
    """
    Load textile dataset from Kaggle download path
    
    Args:
        dataset_path: Path to dataset directory
        sample_size: Optional sample size per class (for testing)
        random_state: Random seed for sampling
        
    Returns:
        Tuple of (image_paths, labels)
    """
    print("Loading dataset...")
    
    # Find all PNG images
    file_paths = list(glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True))
    
    if len(file_paths) == 0:
        raise ValueError(f"No PNG files found in {dataset_path}")
    
    # Extract labels from directory structure
    labels = []
    for path in file_paths:
        # Assumes structure: .../sri_lankan_textiles/*.png or .../japanese_textiles/*.png
        parent_dir = Path(path).parent.name
        labels.append(parent_dir)
    
    # Create DataFrame
    df = pd.DataFrame({'filepath': file_paths, 'label': labels})
    
    # Sample if requested (for testing)
    if sample_size is not None:
        df = df.groupby('label').sample(n=min(sample_size, df.groupby('label').size().min()), 
                                        random_state=random_state)
        df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} images")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df['filepath'].tolist(), df['label'].tolist()


def run_complete_analysis(dataset_path, output_dir='results', sample_size=None, verbose=True):
    """
    Run complete JASPER analysis pipeline
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save results
        sample_size: Optional sample size per class (for testing)
        verbose: Print progress
        
    Returns:
        Dictionary with analysis results
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    print("=" * 80)
    print("JASPER: Japanese x Sri Lankan Textile Design Analysis")
    print("=" * 80)
    print()
    
    # Step 1: Load dataset
    print("STEP 1: Loading dataset...")
    image_paths, labels = load_dataset_from_kaggle(dataset_path, sample_size=sample_size)
    print(f"✓ Loaded {len(image_paths)} images")
    print()
    
    # Step 2: Extract features
    print("STEP 2: Extracting features (44 features per image)...")
    features_csv = os.path.join(output_dir, 'extracted_features.csv')
    
    if os.path.exists(features_csv):
        print(f"Loading existing features from {features_csv}")
        features_df = pd.read_csv(features_csv)
    else:
        features_df = extract_dataset_features(
            image_paths, 
            labels, 
            output_csv=features_csv,
            verbose=verbose
        )
    
    print(f"✓ Extracted features for {len(features_df)} images")
    print(f"✓ Features shape: {features_df.shape}")
    print()
    
    # Step 3: Statistical Analysis
    print("STEP 3: Statistical analysis...")
    comparison_results = compare_textile_groups(
        features_df,
        group1_label='japanese_textiles',
        group2_label='sri_lankan_textiles',
        alpha=0.05,
        apply_correction=True
    )
    
    # Save comparison results
    comparison_csv = os.path.join(output_dir, 'statistical_comparison.csv')
    comparison_results.to_csv(comparison_csv, index=False)
    print(f"✓ Saved statistical comparison to {comparison_csv}")
    
    # Identify key differentiators (very large effect sizes)
    key_features = identify_key_differentiators(
        comparison_results,
        effect_size_threshold=1.2,
        p_value_threshold=0.001
    )
    
    key_features_csv = os.path.join(output_dir, 'key_differentiators.csv')
    key_features.to_csv(key_features_csv, index=False)
    print(f"✓ Found {len(key_features)} key differentiating features (|d| >= 1.2, p < 0.001)")
    print()
    
    # Generate statistical report
    print("STEP 4: Generating statistical report...")
    report_file = os.path.join(output_dir, 'statistical_report.txt')
    report = generate_statistical_report(comparison_results, output_file=report_file)
    print(report)
    print()
    
    # Step 5: Visualizations
    print("STEP 5: Generating visualizations...")
    
    # Color palette comparison
    print("  - Color palette comparison...")
    plot_color_palette_comparison(
        features_df,
        save_path=os.path.join(output_dir, 'figures', 'color_palettes.png')
    )
    
    # RGB comparison
    print("  - RGB profile comparison...")
    plot_rgb_comparison(
        features_df,
        save_path=os.path.join(output_dir, 'figures', 'rgb_comparison.png')
    )
    
    # Feature distributions for key metrics
    print("  - Feature distributions...")
    key_metric_names = ['warm_cool_score', 'texture_complexity', 'texture_contrast', 'symmetry_score']
    key_metric_names = [m for m in key_metric_names if m in features_df.columns]
    
    if len(key_metric_names) > 0:
        plot_feature_distributions(
            features_df,
            features=key_metric_names,
            save_path=os.path.join(output_dir, 'figures', 'key_distributions.png')
        )
    
    # Effect sizes
    print("  - Effect size visualization...")
    plot_effect_sizes(
        comparison_results,
        top_n=20,
        save_path=os.path.join(output_dir, 'figures', 'effect_sizes.png')
    )
    
    # Volcano plot
    print("  - Volcano plot...")
    plot_pvalue_volcano(
        comparison_results,
        save_path=os.path.join(output_dir, 'figures', 'volcano_plot.png')
    )
    
    print("✓ All visualizations saved to", os.path.join(output_dir, 'figures'))
    print()
    
    # Summary
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}/")
    print(f"  - extracted_features.csv: All 44 features for {len(features_df)} images")
    print(f"  - statistical_comparison.csv: Full statistical analysis results")
    print(f"  - key_differentiators.csv: Features with very large effect sizes")
    print(f"  - statistical_report.txt: Comprehensive text report")
    print(f"  - figures/: All publication-quality visualizations")
    print()
    
    return {
        'features': features_df,
        'comparison': comparison_results,
        'key_features': key_features,
        'report': report
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='JASPER: Textile Design Analysis')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size per class for testing (default: use all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_complete_analysis(
        dataset_path=args.dataset,
        output_dir=args.output,
        sample_size=args.sample,
        verbose=args.verbose
    )
    
    print("\nTo reproduce paper results:")
    print("  python main_analysis.py --dataset /path/to/dataset --output results --verbose")
