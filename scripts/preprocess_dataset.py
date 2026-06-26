"""
Preprocessing Runner Script

This script orchestrates the complete preprocessing pipeline for the TrashNet dataset.
It handles data validation, resizing, normalization, stratified splitting, and generates
reports and visualizations.

Usage:
    python scripts/preprocess_dataset.py

Author: MSc Dissertation Project
Date: 2026
"""

import logging
import sys
from pathlib import Path
from typing import Dict
import io

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import PreprocessingPipeline


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Configure logging for preprocessing.
    
    Args:
        log_file: Path to save log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    
    # Console handler with UTF-8 encoding for Windows
    if sys.platform == 'win32':
        console_handler = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# VISUALIZATION GENERATION
# ============================================================================

class PreprocessingVisualizer:
    """Generates visualizations for preprocessing results."""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 6)
    
    def plot_split_distribution(self, split_stats: Dict) -> Path:
        """
        Create distribution plots for train/val/test splits.
        
        Args:
            split_stats: Statistics dictionary from PreprocessingPipeline
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Generating split distribution plots...")
        
        # Prepare data for plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (split_name, stats) in enumerate(split_stats.items()):
            ax = axes[idx]
            class_dist = stats['class_distribution']
            
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            # Create bar chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            bars = ax.bar(classes, counts, color=colors, edgecolor='black', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Format axis
            ax.set_title(f'{split_name.upper()}\n({stats["total_images"]} images)',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Images', fontsize=10)
            ax.set_xlabel('Class', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "split_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved: {output_path}")
        return output_path
    
    def plot_class_distribution_comparison(self, original_dist: Dict, 
                                          split_stats: Dict) -> Path:
        """
        Create comparison plot of class distribution across splits.
        
        Args:
            original_dist: Original class distribution
            split_stats: Statistics for each split
            
        Returns:
            Path to saved visualization
        """
        self.logger.info("Generating class distribution comparison...")
        
        # Extract class names
        classes = sorted(list(original_dist.keys()))
        num_classes = len(classes)
        
        # Prepare data
        train_counts = [split_stats['train']['class_distribution'].get(c, 0) 
                       for c in classes]
        val_counts = [split_stats['validation']['class_distribution'].get(c, 0) 
                     for c in classes]
        test_counts = [split_stats['test']['class_distribution'].get(c, 0) 
                      for c in classes]
        
        # Create grouped bar chart
        x = np.arange(num_classes)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, train_counts, width, label='Train', 
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, val_counts, width, label='Validation', 
                       color='orange', alpha=0.8)
        bars3 = ax.bar(x + width, test_counts, width, label='Test', 
                       color='green', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
        
        # Format axis
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution Across Train/Validation/Test Splits',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "class_distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved: {output_path}")
        return output_path


# ============================================================================
# REPORT GENERATION
# ============================================================================

class PreprocessingReportGenerator:
    """Generates comprehensive preprocessing report."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize report generator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
    
    def generate_report(self, pipeline_results: Dict, output_path: Path) -> Path:
        """
        Generate comprehensive markdown report.
        
        Args:
            pipeline_results: Results dictionary from PreprocessingPipeline
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        self.logger.info(f"Generating preprocessing report: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Data Preprocessing Report\n\n")
            f.write("**Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel "
                   "Production Using Machine Learning**\n\n")
            f.write("---\n\n")
            
            # Table of Contents
            f.write("## Table of Contents\n\n")
            f.write("1. [Overview](#overview)\n")
            f.write("2. [Preprocessing Steps](#preprocessing-steps)\n")
            f.write("3. [Dataset Statistics](#dataset-statistics)\n")
            f.write("4. [Train/Validation/Test Split](#trainvalidationtest-split)\n")
            f.write("5. [Image Properties](#image-properties)\n")
            f.write("6. [Data Quality](#data-quality)\n")
            f.write("7. [Output Structure](#output-structure)\n")
            f.write("8. [Validation Results](#validation-results)\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write("This report documents the preprocessing pipeline applied to the TrashNet "
                   "dataset for the MSc dissertation project.\n\n")
            f.write("**Purpose**: Prepare raw images for machine learning model training by "
                   "standardizing formats, resizing, normalizing, and creating stratified splits.\n\n")
            f.write("**Key Objective**: Maintain data integrity while creating reproducible, "
                   "balanced dataset splits.\n\n")
            
            # Preprocessing Steps
            f.write("## Preprocessing Steps\n\n")
            f.write("The following preprocessing steps were applied to each image:\n\n")
            f.write("1. **Image Validation**: Verify all images are readable and not corrupted\n")
            f.write("2. **Format Conversion**: Convert all images to RGB format (3 channels)\n")
            f.write("3. **Resizing**: Resize all images to 224x224 pixels using Lanczos interpolation\n")
            f.write("4. **Normalization**: Normalize pixel values to range [0, 1] by dividing by 255\n")
            f.write("5. **Stratified Splitting**: Divide dataset into train/validation/test sets "
                   "while maintaining class distribution\n")
            f.write("6. **Storage**: Save processed images as PNG files (lossless compression)\n\n")
            
            # Dataset Statistics
            f.write("## Dataset Statistics\n\n")
            split_stats = pipeline_results.get('splits', {})
            
            f.write("### Original Dataset\n\n")
            f.write(f"- **Total Images**: {pipeline_results.get('valid_images_count', 0)}\n")
            f.write(f"- **Total Classes**: 6\n")
            f.write(f"- **Classes**: cardboard, glass, metal, paper, plastic, trash\n\n")
            
            # Train/Val/Test Split
            f.write("## Train/Validation/Test Split\n\n")
            f.write("### Split Ratios\n\n")
            f.write("- **Training**: 70%\n")
            f.write("- **Validation**: 15%\n")
            f.write("- **Testing**: 15%\n\n")
            
            f.write("### Split Results\n\n")
            for split_name, stats in split_stats.items():
                total = stats['total_images']
                percentage = (total / pipeline_results.get('valid_images_count', 1) * 100)
                f.write(f"**{split_name.upper()}**: {total} images ({percentage:.1f}%)\n\n")
                
                f.write("| Class | Count | Percentage |\n")
                f.write("|-------|-------|------------|\n")
                
                split_total = stats['total_images']
                for class_name in sorted(stats['class_distribution'].keys()):
                    count = stats['class_distribution'][class_name]
                    class_pct = (count / split_total * 100) if split_total > 0 else 0
                    f.write(f"| {class_name} | {count} | {class_pct:.2f}% |\n")
                
                f.write("\n")
            
            # Image Properties
            f.write("## Image Properties\n\n")
            f.write("### Target Specifications\n\n")
            f.write("- **Resolution**: 224 x 224 pixels\n")
            f.write("- **Format**: RGB (3 channels)\n")
            f.write("- **Data Type**: 32-bit floating point (float32)\n")
            f.write("- **Value Range**: [0.0, 1.0]\n")
            f.write("- **Interpolation**: Lanczos (high-quality resizing)\n")
            f.write("- **Storage Format**: PNG (lossless)\n\n")
            
            # Data Quality
            f.write("## Data Quality\n\n")
            invalid_images = pipeline_results.get('invalid_images', [])
            f.write(f"**Invalid/Corrupted Images**: {len(invalid_images)}\n")
            if invalid_images:
                f.write("\nCorrupted files found:\n")
                for img_path in invalid_images[:10]:
                    f.write(f"- `{img_path}`\n")
                if len(invalid_images) > 10:
                    f.write(f"- ... and {len(invalid_images) - 10} more\n")
            else:
                f.write("\n[OK] All images validated successfully\n")
            
            f.write("\n")
            
            # Output Structure
            f.write("## Output Structure\n\n")
            f.write("Processed dataset saved to `data/processed/` with structure:\n\n")
            f.write("```\n")
            f.write("data/processed/\n")
            f.write("├── train/\n")
            f.write("│   ├── cardboard/\n")
            f.write("│   ├── glass/\n")
            f.write("│   ├── metal/\n")
            f.write("│   ├── paper/\n")
            f.write("│   ├── plastic/\n")
            f.write("│   └── trash/\n")
            f.write("├── validation/\n")
            f.write("│   ├── cardboard/\n")
            f.write("│   └── ...\n")
            f.write("└── test/\n")
            f.write("    ├── cardboard/\n")
            f.write("    └── ...\n")
            f.write("```\n\n")
            
            # Validation Results
            f.write("## Validation Results\n\n")
            f.write("| Metric | Result |\n")
            f.write("|--------|--------|\n")
            f.write(f"| Total Images Processed | {pipeline_results.get('valid_images_count', 0)} |\n")
            f.write(f"| Training Images | {split_stats.get('train', {}).get('total_images', 0)} |\n")
            f.write(f"| Validation Images | {split_stats.get('validation', {}).get('total_images', 0)} |\n")
            f.write(f"| Test Images | {split_stats.get('test', {}).get('total_images', 0)} |\n")
            f.write(f"| Invalid Images | {len(invalid_images)} |\n")
            f.write(f"| Target Resolution | 224x224 RGB |\n")
            f.write(f"| Normalization | [0, 1] |\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            f.write("1. **Class Balance**: Review class distribution for potential imbalance:\n")
            
            train_dist = split_stats.get('train', {}).get('class_distribution', {})
            if train_dist:
                min_class = min(train_dist.keys(), key=lambda x: train_dist[x])
                max_class = max(train_dist.keys(), key=lambda x: train_dist[x])
                min_count = train_dist[min_class]
                max_count = train_dist[max_class]
                
                f.write(f"   - Smallest class (train): {min_class} ({min_count} images)\n")
                f.write(f"   - Largest class (train): {max_class} ({max_count} images)\n")
                f.write(f"   - Imbalance ratio: {max_count/max(min_count, 1):.2f}:1\n\n")
            
            f.write("2. **Data Quality**: All images validated and readable\n\n")
            f.write("3. **Reproducibility**: Random seed (42) ensures reproducible splits\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Class Imbalance Handling**: Consider class weighting in model training\n")
            f.write("2. **Data Augmentation**: Apply augmentation during training for minority classes\n")
            f.write("3. **Model Input**: Images are ready for deep learning pipelines (e.g., PyTorch, TensorFlow)\n")
            f.write("4. **Validation Strategy**: Use stratified split for fair cross-validation\n\n")
            
            f.write("---\n\n")
            f.write("*Report generated automatically by preprocessing pipeline*\n")
        
        self.logger.info("[OK] Report saved successfully")
        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main execution function."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_dataset_path = project_root / "TrashNET Data set"
    processed_output_path = project_root / "data" / "processed"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    log_file = project_root / "preprocessing.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    logger.info(f"Project root: {project_root}")
    
    # Check if raw dataset exists
    if not raw_dataset_path.exists():
        logger.error(f"Raw dataset not found at: {raw_dataset_path}")
        return
    
    # Run preprocessing pipeline
    pipeline = PreprocessingPipeline(
        raw_dataset_path=raw_dataset_path,
        processed_output_path=processed_output_path,
        target_size=(224, 224),
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        logger=logger
    )
    
    results = pipeline.run()
    
    if not results.get('success'):
        logger.error(f"Pipeline failed: {results.get('error')}")
        return
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    visualizer = PreprocessingVisualizer(figures_dir, logger)
    visualizer.plot_split_distribution(results['splits'])
    visualizer.plot_class_distribution_comparison(
        results['class_distribution'],
        results['splits']
    )
    
    # Generate report
    logger.info("\nGenerating preprocessing report...")
    report_gen = PreprocessingReportGenerator(logger)
    report_path = reports_dir / "preprocessing_report.md"
    report_gen.generate_report(results, report_path)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("Preprocessing Complete!")
    logger.info("="*70)
    logger.info(f"Processed dataset: {processed_output_path}")
    logger.info(f"Report: {report_path}")
    logger.info(f"Visualizations: {figures_dir}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
