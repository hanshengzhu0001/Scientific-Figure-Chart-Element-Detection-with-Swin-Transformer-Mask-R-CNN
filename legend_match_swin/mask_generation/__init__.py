"""
Mask Generation Module for Data Point Instance Segmentation

This module provides tools for semi-automatic mask generation for data points
in scientific charts, using OpenCV for contour detection and COCO format conversion.

Quick Start:
1. Run: python check_setup.py (to verify your environment)
2. Run: python test_mask_generation.py (to test with sample data)
3. Run: python generate_masks_demo.py (for full pipeline)
"""

from .mask_generator import DataPointMaskGenerator
from .coco_converter import COCOMaskConverter
from .utils import visualize_masks, validate_masks

__all__ = [
    'DataPointMaskGenerator',
    'COCOMaskConverter', 
    'visualize_masks',
    'validate_masks'
] 