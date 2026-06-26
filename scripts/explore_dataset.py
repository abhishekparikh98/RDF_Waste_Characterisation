"""
Dataset Exploration and Validation Script

This script performs comprehensive exploration and validation of the TrashNet and TACO
datasets for the Multi-Modal Waste Characterisation project. It detects dataset locations,
counts images, identifies corrupted files, analyzes image properties, and generates
visualizations and reports.

Author: MSc Dissertation Project
Date: 2026
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configure UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logging for the exploration script.
    
    Args:
        log_file: Optional path to save log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
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
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATASET DETECTION
# ============================================================================

class DatasetDetector:
    """Detects and locates available datasets in the project."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, base_path: Path, logger: logging.Logger):
        """
        Initialize dataset detector.
        
        Args:
            base_path: Base path to search for datasets
            logger: Logger instance
        """
        self.base_path = base_path
        self.logger = logger
        self.datasets: Dict[str, Path] = {}
        
    def detect_datasets(self) -> Dict[str, Path]:
        """
        Auto-detect available datasets in the project.
        
        Returns:
            Dictionary mapping dataset names to their paths
        """
        self.logger.info(f"Searching for datasets in: {self.base_path}")
        
        # Check for TrashNet
        trashnet_paths = [
            self.base_path / "TrashNET Data set",
            self.base_path / "data" / "raw" / "TrashNET Data set",
            self.base_path / "data" / "raw" / "TrashNET",
        ]
        
        for path in trashnet_paths:
            if path.exists():
                self.datasets['TrashNet'] = path
                self.logger.info(f"[FOUND] TrashNet detected at: {path}")
                break
        
        # Check for TACO
        taco_paths = [
            self.base_path / "TACO-master",
            self.base_path / "data" / "raw" / "TACO-master",
            self.base_path / "data" / "raw" / "TACO",
        ]
        
        for path in taco_paths:
            if path.exists():
                self.datasets['TACO'] = path
                self.logger.info(f"[FOUND] TACO detected at: {path}")
                break
        
        if not self.datasets:
            self.logger.warning("No datasets detected in expected locations")
        
        return self.datasets


# ============================================================================
# IMAGE ANALYSIS
# ============================================================================

class ImageAnalyzer:
    """Analyzes images in datasets for validity and properties."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize image analyzer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
    def is_valid_image(self, image_path: Path) -> bool:
        """
        Check if an image file is valid and readable.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except (IOError, OSError, Image.UnidentifiedImageError):
            return False
        
    def get_image_resolution(self, image_path: Path) -> Optional[Tuple[int, int]]:
        """
        Get the resolution of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height) or None if unable to read
        """
        try:
            with Image.open(image_path) as img:
                return img.size
        except (IOError, OSError, Image.UnidentifiedImageError):
            return None


# ============================================================================
# DATASET EXPLORER
# ============================================================================

class DatasetExplorer:
    """Explores and validates datasets comprehensively."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, dataset_path: Path, dataset_name: str, logger: logging.Logger):
        """
        Initialize dataset explorer.
        
        Args:
            dataset_path: Path to dataset
            dataset_name: Name of dataset
            logger: Logger instance
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.logger = logger
        self.analyzer = ImageAnalyzer(logger)
        self.stats: Dict = {}
        
    def explore(self) -> Dict:
        """
        Perform comprehensive exploration of the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Exploring {self.dataset_name} dataset")
        self.logger.info(f"{'='*70}")
        
        # Collect images by class
        class_images = self._collect_images_by_class()
        
        if not class_images:
            self.logger.warning(f"No images found in {self.dataset_name}")
            return {}
        
        # Calculate statistics
        self.stats = self._calculate_statistics(class_images)
        
        # Report findings
        self._report_findings()
        
        return self.stats
    
    def _collect_images_by_class(self) -> Dict[str, List[Path]]:
        """
        Collect images organized by class/category.
        
        Returns:
            Dictionary mapping class names to list of image paths
        """
        class_images: Dict[str, List[Path]] = defaultdict(list)
        
        self.logger.info("Collecting images...")
        
        # Search for images in subdirectories (class folders)
        if self.dataset_name == "TrashNet":
            # TrashNet uses class folders directly
            for class_dir in self.dataset_path.glob("dataset-resized/*"):
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for image_path in class_dir.glob("*"):
                        if image_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                            class_images[class_name].append(image_path)
        else:
            # Generic search for any dataset structure
            for image_path in self.dataset_path.rglob("*"):
                if (image_path.is_file() and 
                    image_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS):
                    # Use parent directory as class if no explicit class structure
                    class_name = image_path.parent.name
                    class_images[class_name].append(image_path)
        
        self.logger.info(f"Found {len(class_images)} classes")
        return class_images
    
    def _calculate_statistics(self, class_images: Dict[str, List[Path]]) -> Dict:
        """
        Calculate detailed statistics about the dataset.
        
        Args:
            class_images: Dictionary mapping classes to image paths
            
        Returns:
            Dictionary containing comprehensive statistics
        """
        stats = {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path),
            'total_images': 0,
            'total_classes': len(class_images),
            'class_distribution': {},
            'corrupted_images': [],
            'image_resolutions': [],
            'resolution_stats': {},
        }
        
        self.logger.info("Analyzing images...")
        
        for class_name, image_paths in sorted(class_images.items()):
            valid_images = []
            corrupted = []
            
            for image_path in image_paths:
                if self.analyzer.is_valid_image(image_path):
                    valid_images.append(image_path)
                    
                    # Get resolution
                    resolution = self.analyzer.get_image_resolution(image_path)
                    if resolution:
                        stats['image_resolutions'].append(resolution)
                else:
                    corrupted.append(str(image_path))
            
            if corrupted:
                stats['corrupted_images'].extend(corrupted)
            
            stats['class_distribution'][class_name] = len(valid_images)
            stats['total_images'] += len(valid_images)
            
            self.logger.debug(
                f"  {class_name}: {len(valid_images)} valid, "
                f"{len(corrupted)} corrupted"
            )
        
        # Calculate resolution statistics
        if stats['image_resolutions']:
            resolutions_array = np.array(stats['image_resolutions'])
            stats['resolution_stats'] = {
                'min_width': int(resolutions_array[:, 0].min()),
                'max_width': int(resolutions_array[:, 0].max()),
                'avg_width': float(resolutions_array[:, 0].mean()),
                'min_height': int(resolutions_array[:, 1].min()),
                'max_height': int(resolutions_array[:, 1].max()),
                'avg_height': float(resolutions_array[:, 1].mean()),
            }
        
        return stats
    
    def _report_findings(self) -> None:
        """Log findings about the dataset."""
        self.logger.info(f"\nDataset Summary for {self.dataset_name}:")
        self.logger.info(f"  Total Images: {self.stats['total_images']}")
        self.logger.info(f"  Total Classes: {self.stats['total_classes']}")
        
        if self.stats['corrupted_images']:
            self.logger.warning(
                f"  Corrupted Images: {len(self.stats['corrupted_images'])}"
            )
        
        self.logger.info("\n  Class Distribution:")
        for class_name, count in sorted(self.stats['class_distribution'].items()):
            percentage = (count / self.stats['total_images'] * 100 
                         if self.stats['total_images'] > 0 else 0)
            self.logger.info(f"    {class_name}: {count} ({percentage:.2f}%)")
        
        if self.stats['resolution_stats']:
            res = self.stats['resolution_stats']
            self.logger.info("\n  Image Resolutions:")
            self.logger.info(f"    Width  - Min: {res['min_width']}, "
                           f"Max: {res['max_width']}, "
                           f"Avg: {res['avg_width']:.2f}")
            self.logger.info(f"    Height - Min: {res['min_height']}, "
                           f"Max: {res['max_height']}, "
                           f"Avg: {res['avg_height']:.2f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

class VisualizationGenerator:
    """Generates visualizations for dataset analysis."""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def generate_class_distribution_chart(
        self,
        stats: Dict,
        dataset_name: str
    ) -> Path:
        """
        Generate class distribution bar chart.
        
        Args:
            stats: Dataset statistics
            dataset_name: Name of dataset
            
        Returns:
            Path to saved chart
        """
        if not stats.get('class_distribution'):
            self.logger.warning(f"No class distribution data for {dataset_name}")
            return None
        
        self.logger.info(f"Generating class distribution chart for {dataset_name}")
        
        classes = list(stats['class_distribution'].keys())
        counts = list(stats['class_distribution'].values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(classes, counts, color='steelblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name} - Class Distribution\n'
                    f'Total Images: {stats["total_images"]}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{dataset_name.lower()}_class_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")
        return output_path
    
    def generate_combined_comparison_chart(
        self,
        all_stats: Dict[str, Dict]
    ) -> Optional[Path]:
        """
        Generate comparison chart for all datasets.
        
        Args:
            all_stats: Statistics for all datasets
            
        Returns:
            Path to saved chart or None
        """
        if len(all_stats) < 2:
            self.logger.info("Only one dataset found, skipping comparison chart")
            return None
        
        self.logger.info("Generating dataset comparison chart")
        
        fig, axes = plt.subplots(1, len(all_stats), figsize=(15, 6))
        if len(all_stats) == 1:
            axes = [axes]
        
        for idx, (dataset_name, stats) in enumerate(all_stats.items()):
            if not stats.get('class_distribution'):
                continue
            
            ax = axes[idx]
            classes = list(stats['class_distribution'].keys())
            counts = list(stats['class_distribution'].values())
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
            ax.bar(classes, counts, color=colors, edgecolor='black', alpha=0.7)
            
            ax.set_title(f'{dataset_name}\n({stats["total_images"]} images)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Number of Images', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "dataset_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")
        return output_path


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generates comprehensive dataset exploration reports."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize report generator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
    
    def generate_markdown_report(
        self,
        all_stats: Dict[str, Dict],
        output_path: Path
    ) -> Path:
        """
        Generate comprehensive markdown report.
        
        Args:
            all_stats: Statistics for all datasets
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        self.logger.info(f"\nGenerating markdown report: {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Dataset Exploration and Validation Report\n\n")
            f.write("**Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel "
                   "Production Using Machine Learning**\n\n")
            f.write("---\n\n")
            
            # Table of Contents
            f.write("## Table of Contents\n\n")
            f.write("1. [Executive Summary](#executive-summary)\n")
            f.write("2. [Dataset Overview](#dataset-overview)\n")
            f.write("3. [Detailed Statistics](#detailed-statistics)\n")
            f.write("4. [Image Analysis](#image-analysis)\n")
            f.write("5. [Data Quality](#data-quality)\n")
            f.write("6. [Visualizations](#visualizations)\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_images = sum(stats.get('total_images', 0) for stats in all_stats.values())
            total_classes = sum(stats.get('total_classes', 0) for stats in all_stats.values())
            f.write(f"- **Total Images Analyzed**: {total_images}\n")
            f.write(f"- **Total Classes**: {total_classes}\n")
            f.write(f"- **Datasets Found**: {len(all_stats)}\n\n")
            
            # Dataset Overview
            f.write("## Dataset Overview\n\n")
            for dataset_name, stats in all_stats.items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- **Path**: `{stats.get('dataset_path', 'Unknown')}`\n")
                f.write(f"- **Total Images**: {stats.get('total_images', 0)}\n")
                f.write(f"- **Number of Classes**: {stats.get('total_classes', 0)}\n\n")
            
            # Detailed Statistics
            f.write("## Detailed Statistics\n\n")
            for dataset_name, stats in all_stats.items():
                if not stats:
                    continue
                
                f.write(f"### {dataset_name} - Class Distribution\n\n")
                f.write("| Class | Image Count | Percentage |\n")
                f.write("|-------|-------------|------------|\n")
                
                total = stats.get('total_images', 1)
                for class_name in sorted(stats.get('class_distribution', {}).keys()):
                    count = stats['class_distribution'][class_name]
                    percentage = (count / total * 100) if total > 0 else 0
                    f.write(f"| {class_name} | {count} | {percentage:.2f}% |\n")
                
                f.write(f"\n**Total**: {total} images\n\n")
            
            # Image Analysis
            f.write("## Image Analysis\n\n")
            for dataset_name, stats in all_stats.items():
                if not stats.get('resolution_stats'):
                    continue
                
                f.write(f"### {dataset_name} - Image Resolutions\n\n")
                res = stats['resolution_stats']
                f.write(f"**Width (pixels)**\n")
                f.write(f"- Minimum: {res.get('min_width', 'N/A')}\n")
                f.write(f"- Maximum: {res.get('max_width', 'N/A')}\n")
                f.write(f"- Average: {res.get('avg_width', 'N/A'):.2f}\n\n")
                
                f.write(f"**Height (pixels)**\n")
                f.write(f"- Minimum: {res.get('min_height', 'N/A')}\n")
                f.write(f"- Maximum: {res.get('max_height', 'N/A')}\n")
                f.write(f"- Average: {res.get('avg_height', 'N/A'):.2f}\n\n")
            
            # Data Quality
            f.write("## Data Quality\n\n")
            for dataset_name, stats in all_stats.items():
                corrupted = stats.get('corrupted_images', [])
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- **Corrupted Images Detected**: {len(corrupted)}\n")
                if corrupted:
                    f.write("\nCorrupted files:\n")
                    for filepath in corrupted[:10]:  # Show first 10
                        f.write(f"- `{filepath}`\n")
                    if len(corrupted) > 10:
                        f.write(f"- ... and {len(corrupted) - 10} more\n")
                else:
                    f.write("- [OK] No corrupted images detected\n")
                f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("Class distribution charts have been generated and saved to "
                   "`reports/figures/`\n\n")
            
            for dataset_name in all_stats.keys():
                f.write(f"- `{dataset_name.lower()}_class_distribution.png`\n")
            
            f.write("- `dataset_comparison.png`\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Data Balance**: Review class distribution for imbalance issues\n")
            f.write("2. **Image Sizes**: Consider standardizing image resolutions\n")
            f.write("3. **Data Quality**: Investigate and handle any corrupted images\n")
            f.write("4. **Train/Val/Test Split**: Plan appropriate data splits before modeling\n\n")
            
            f.write("---\n\n")
            f.write("*Report generated automatically by dataset exploration script*\n")
        
        self.logger.info(f"[OK] Report saved successfully")
        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main execution function."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    log_path = project_root / "exploration.log"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    
    # Setup logging
    logger = setup_logging(log_path)
    logger.info(f"Starting dataset exploration from: {project_root}")
    
    # Detect datasets
    detector = DatasetDetector(project_root, logger)
    datasets = detector.detect_datasets()
    
    if not datasets:
        logger.error("No datasets detected. Please ensure TrashNet and/or TACO "
                    "are in the project root or data/raw/ directory.")
        return
    
    # Explore datasets
    all_stats = {}
    for dataset_name, dataset_path in datasets.items():
        explorer = DatasetExplorer(dataset_path, dataset_name, logger)
        stats = explorer.explore()
        if stats:
            all_stats[dataset_name] = stats
    
    # Generate visualizations
    viz_gen = VisualizationGenerator(figures_dir, logger)
    for dataset_name, stats in all_stats.items():
        viz_gen.generate_class_distribution_chart(stats, dataset_name)
    
    if len(all_stats) > 1:
        viz_gen.generate_combined_comparison_chart(all_stats)
    
    # Generate report
    report_gen = ReportGenerator(logger)
    report_path = reports_dir / "dataset_report.md"
    report_gen.generate_markdown_report(all_stats, report_path)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Dataset Exploration Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Reports saved to: {reports_dir}")
    logger.info(f"Visualizations saved to: {figures_dir}")
    logger.info(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
