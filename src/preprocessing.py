"""
Data Preprocessing Module for TrashNet Dataset

This module contains all preprocessing utilities for the Multi-Modal Waste Characterisation
project. It handles image loading, validation, resizing, normalization, and stratified dataset
splitting without modifying the original dataset.

Classes:
    - ImageValidator: Validates image readability
    - ImagePreprocessor: Handles resizing and normalization
    - DatasetSplitter: Performs stratified train/val/test splits
    - PreprocessingPipeline: Orchestrates the complete pipeline

Author: MSc Dissertation Project
Date: 2026
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================================
# IMAGE VALIDATION
# ============================================================================

class ImageValidator:
    """Validates image files for readability and integrity."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize image validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.invalid_images: List[str] = []
        
    def is_valid_image(self, image_path: Path) -> bool:
        """
        Verify if an image file is readable.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid and readable, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            self.logger.debug(f"Invalid image: {image_path} - {str(e)}")
            self.invalid_images.append(str(image_path))
            return False
    
    def validate_dataset(self, dataset_path: Path) -> Dict[str, List[Path]]:
        """
        Validate all images in a dataset.
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            Dictionary mapping class names to valid image paths
        """
        self.logger.info("Validating dataset images...")
        valid_images: Dict[str, List[Path]] = defaultdict(list)
        
        # Scan TrashNet structure
        dataset_classes = (dataset_path / "dataset-resized").glob("*")
        
        for class_dir in dataset_classes:
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob("*"))
            
            valid_count = 0
            for image_path in image_files:
                if (image_path.suffix.lower() in self.SUPPORTED_EXTENSIONS and
                    self.is_valid_image(image_path)):
                    valid_images[class_name].append(image_path)
                    valid_count += 1
            
            self.logger.debug(f"  {class_name}: {valid_count} valid images")
        
        if self.invalid_images:
            self.logger.warning(f"Found {len(self.invalid_images)} invalid images")
        else:
            self.logger.info("All images validated successfully")
        
        return valid_images


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Handles image resizing, normalization, and preprocessing."""
    
    def __init__(self, target_size: Tuple[int, int], logger: logging.Logger):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            logger: Logger instance
        """
        self.target_size = target_size
        self.logger = logger
        
    def load_and_preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Operations:
        1. Load image as RGB
        2. Resize to target size using high-quality interpolation
        3. Normalize pixel values to [0, 1]
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array [H, W, 3] with values in [0, 1],
            or None if processing fails
        """
        try:
            # Load image as RGB
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (removes alpha channel, converts grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize using high-quality Lanczos interpolation
                img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img_resized, dtype=np.float32)
                
                # Normalize to [0, 1]
                img_normalized = img_array / 255.0
                
                return img_normalized
                
        except Exception as e:
            self.logger.error(f"Failed to preprocess {image_path}: {str(e)}")
            return None
    
    def preprocess_batch(self, image_paths: List[Path]) -> Dict[str, np.ndarray]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary with 'images' (array) and 'paths' (list of original paths)
        """
        images = []
        valid_paths = []
        
        for image_path in image_paths:
            img = self.load_and_preprocess_image(image_path)
            if img is not None:
                images.append(img)
                valid_paths.append(image_path)
        
        return {
            'images': np.array(images) if images else np.array([]),
            'paths': valid_paths
        }


# ============================================================================
# DATASET SPLITTING
# ============================================================================

class DatasetSplitter:
    """Performs stratified train/validation/test split."""
    
    def __init__(self, train_ratio: float, val_ratio: float, test_ratio: float, 
                 random_state: int, logger: logging.Logger):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Proportion for training (e.g., 0.70)
            val_ratio: Proportion for validation (e.g., 0.15)
            test_ratio: Proportion for testing (e.g., 0.15)
            random_state: Random seed for reproducibility
            logger: Logger instance
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.logger = logger
        
    def stratified_split(self, valid_images: Dict[str, List[Path]]) -> Dict[str, Dict]:
        """
        Perform stratified split of dataset by class.
        
        Uses sklearn's stratified split to ensure class distribution is maintained
        across train/val/test sets.
        
        Args:
            valid_images: Dictionary mapping class names to image paths
            
        Returns:
            Dictionary with 'train', 'validation', 'test' keys, each containing
            a dict of class -> image paths
        """
        self.logger.info(f"Performing stratified split: "
                        f"train={self.train_ratio}, val={self.val_ratio}, "
                        f"test={self.test_ratio}")
        
        splits = {
            'train': defaultdict(list),
            'validation': defaultdict(list),
            'test': defaultdict(list)
        }
        
        # Split each class independently to maintain stratification
        for class_name, image_paths in valid_images.items():
            image_list = list(image_paths)
            num_images = len(image_list)
            
            if num_images < 3:
                self.logger.warning(
                    f"Class '{class_name}' has only {num_images} images; "
                    f"stratified split may not be reliable"
                )
            
            # First split: separate test set
            train_val, test = train_test_split(
                image_list,
                test_size=self.test_ratio,
                random_state=self.random_state,
                stratify=None  # Single class, no stratification needed
            )
            
            # Second split: separate validation from training
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio_adjusted,
                random_state=self.random_state,
                stratify=None
            )
            
            splits['train'][class_name] = train
            splits['validation'][class_name] = val
            splits['test'][class_name] = test
            
            self.logger.debug(
                f"  {class_name}: train={len(train)}, val={len(val)}, test={len(test)}"
            )
        
        return splits
    
    def get_split_statistics(self, splits: Dict[str, Dict]) -> Dict:
        """
        Calculate statistics about the splits.
        
        Args:
            splits: Dictionary of splits from stratified_split()
            
        Returns:
            Dictionary with statistics for each split
        """
        stats = {}
        
        for split_name, split_data in splits.items():
            total_images = sum(len(paths) for paths in split_data.values())
            class_counts = {cls: len(paths) for cls, paths in split_data.items()}
            
            stats[split_name] = {
                'total_images': total_images,
                'num_classes': len(split_data),
                'class_distribution': class_counts
            }
        
        return stats


# ============================================================================
# DATA SAVING
# ============================================================================

class DatasetSaver:
    """Saves preprocessed images to disk."""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        """
        Initialize dataset saver.
        
        Args:
            output_dir: Base directory for processed data
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger
        
    def save_preprocessed_dataset(self, splits: Dict[str, Dict], 
                                  preprocessor: ImagePreprocessor) -> Dict[str, int]:
        """
        Save preprocessed images to disk, organized by split and class.
        
        Directory structure created:
        ```
        data/processed/
        ├── train/
        │   ├── cardboard/
        │   ├── glass/
        │   └── ...
        ├── validation/
        └── test/
        ```
        
        Args:
            splits: Dictionary of splits from stratified_split()
            preprocessor: ImagePreprocessor instance for preprocessing
            
        Returns:
            Dictionary with count of saved images per split
        """
        self.logger.info("Saving preprocessed dataset...")
        saved_counts = {}
        
        for split_name, split_data in splits.items():
            split_dir = self.output_dir / split_name
            saved_count = 0
            
            for class_name, image_paths in split_data.items():
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for idx, image_path in enumerate(image_paths):
                    # Load and preprocess
                    img_array = preprocessor.load_and_preprocess_image(image_path)
                    if img_array is None:
                        continue
                    
                    # Convert back to uint8 for saving as PNG (lossless)
                    img_uint8 = (img_array * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8, mode='RGB')
                    
                    # Save with original filename
                    output_path = class_dir / image_path.name
                    img_pil.save(output_path, quality=95)
                    saved_count += 1
                
                self.logger.debug(
                    f"  {split_name}/{class_name}: {len(image_paths)} images saved"
                )
            
            saved_counts[split_name] = saved_count
            self.logger.info(
                f"  {split_name}: {saved_count} images saved"
            )
        
        return saved_counts


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline."""
    
    def __init__(self,
                 raw_dataset_path: Path,
                 processed_output_path: Path,
                 target_size: Tuple[int, int] = (224, 224),
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42,
                 logger: logging.Logger = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            raw_dataset_path: Path to raw TrashNet dataset
            processed_output_path: Path to save processed dataset
            target_size: Target image dimensions (default: 224x224)
            train_ratio: Proportion for training (default: 0.70)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            random_state: Random seed (default: 42)
            logger: Logger instance
        """
        self.raw_dataset_path = Path(raw_dataset_path)
        self.processed_output_path = Path(processed_output_path)
        self.target_size = target_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.validator = ImageValidator(self.logger)
        self.preprocessor = ImagePreprocessor(target_size, self.logger)
        self.splitter = DatasetSplitter(
            train_ratio, val_ratio, test_ratio, random_state, self.logger
        )
        self.saver = DatasetSaver(self.processed_output_path, self.logger)
        
        # Pipeline state
        self.valid_images: Optional[Dict] = None
        self.splits: Optional[Dict] = None
        self.split_stats: Optional[Dict] = None
        self.saved_counts: Optional[Dict] = None
        
    def run(self) -> Dict:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        self.logger.info("="*70)
        self.logger.info("Starting Preprocessing Pipeline")
        self.logger.info("="*70)
        
        try:
            # Step 1: Validate images
            self.logger.info("\n[STEP 1/5] Validating dataset images...")
            self.valid_images = self.validator.validate_dataset(self.raw_dataset_path)
            
            if not self.valid_images:
                self.logger.error("No valid images found in dataset!")
                return {'success': False, 'error': 'No valid images found'}
            
            total_valid = sum(len(paths) for paths in self.valid_images.values())
            self.logger.info(f"  Validation complete: {total_valid} valid images found")
            
            # Step 2: Perform stratified split
            self.logger.info("\n[STEP 2/5] Performing stratified split...")
            self.splits = self.splitter.stratified_split(self.valid_images)
            self.split_stats = self.splitter.get_split_statistics(self.splits)
            
            for split_name, stats in self.split_stats.items():
                self.logger.info(f"  {split_name}: {stats['total_images']} images")
            
            # Step 3: Save preprocessed dataset
            self.logger.info("\n[STEP 3/5] Saving preprocessed images...")
            self.saved_counts = self.saver.save_preprocessed_dataset(
                self.splits, self.preprocessor
            )
            
            # Step 4: Summary report
            self.logger.info("\n[STEP 4/5] Pipeline Summary")
            self._log_summary()
            
            self.logger.info("\n[STEP 5/5] Pipeline Complete!")
            self.logger.info("="*70)
            
            return {
                'success': True,
                'valid_images_count': total_valid,
                'splits': self.split_stats,
                'saved_counts': self.saved_counts,
                'class_distribution': dict(self.valid_images)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _log_summary(self) -> None:
        """Log summary statistics."""
        self.logger.info("  Dataset Split Summary:")
        for split_name, stats in self.split_stats.items():
            self.logger.info(f"    {split_name.upper()}: {stats['total_images']} images")
            for class_name, count in sorted(stats['class_distribution'].items()):
                percentage = (count / stats['total_images'] * 100 
                             if stats['total_images'] > 0 else 0)
                self.logger.info(
                    f"      {class_name}: {count} ({percentage:.2f}%)"
                )
    
    def get_pipeline_results(self) -> Dict:
        """
        Get detailed results from the pipeline.
        
        Returns:
            Dictionary with all pipeline results
        """
        if not self.split_stats:
            return {'error': 'Pipeline not yet executed'}
        
        return {
            'valid_images_total': sum(len(paths) for paths in 
                                     self.valid_images.values()) if self.valid_images else 0,
            'split_statistics': self.split_stats,
            'saved_counts': self.saved_counts,
            'invalid_images': self.validator.invalid_images,
        }
