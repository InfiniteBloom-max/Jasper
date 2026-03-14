"""
JASPER: Japanese x Sri Lankan Textile Design Analysis

Research code for computational analysis of cross-cultural textile patterns.
"""

__version__ = "1.0.0"
__author__ = "JASPER Research Team"
__email__ = "contact@jasper-research.org"

from .feature_extraction import TextileFeatureExtractor, extract_dataset_features
from .statistical_analysis import compare_textile_groups, generate_statistical_report
from .visualization import (
    plot_color_palette_comparison,
    plot_feature_distributions,
    plot_effect_sizes,
    plot_pvalue_volcano,
    plot_rgb_comparison
)

__all__ = [
    'TextileFeatureExtractor',
    'extract_dataset_features',
    'compare_textile_groups',
    'generate_statistical_report',
    'plot_color_palette_comparison',
    'plot_feature_distributions',
    'plot_effect_sizes',
    'plot_pvalue_volcano',
    'plot_rgb_comparison'
]
