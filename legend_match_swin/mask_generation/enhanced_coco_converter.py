#!/usr/bin/env python3
"""
Enhanced COCO format converter for multiple chart element types.

Converts OpenCV masks to COCO format with polygon segmentation annotations
for various chart elements including data points, lines, bars, boxplots, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import json
from pathlib import Path
import logging
from datetime import datetime

from coco_converter import COCOMaskConverter

logger = logging.getLogger(__name__)


class EnhancedCOCOMaskConverter(COCOMaskConverter):
    """
    Enhanced COCO converter for multiple chart element types.
    
    Supports conversion of masks for:
    - Data points (category ID 11)
    - Data lines (category ID 12) 
    - Data bars (category ID 13)
    - Boxplots (category ID 21)
    - Other chart elements
    """
    
    # Standard 22-category mapping for chart elements
    CATEGORY_MAPPING = {
        'title': 0, 'subtitle': 1, 'x-axis': 2, 'y-axis': 3, 
        'x-axis-label': 4, 'y-axis-label': 5, 'x-tick-label': 6, 'y-tick-label': 7,
        'legend': 8, 'legend-title': 9, 'legend-item': 10,
        'data-point': 11, 'data-line': 12, 'data-bar': 13, 'data-area': 14, 
        'grid-line': 15, 'axis-title': 16, 'tick-label': 17, 'data-label': 18, 
        'legend-text': 19, 'plot-area': 20, 'boxplot': 21
    }
    
    def __init__(self, 
                 target_categories: List[str] = None,
                 simplify_tolerance: float = 0.1,  # Much lower for small data points
                 min_polygon_points: int = 3,      # Allow triangles (simplified circles)
                 use_rle: bool = False):
        """
        Initialize the enhanced COCO converter.
        
        Args:
            target_categories: List of category names to generate masks for. 
                             If None, uses all visual element categories.
            simplify_tolerance: Tolerance for polygon simplification
            min_polygon_points: Minimum number of points required for a valid polygon
            use_rle: Whether to use RLE format instead of polygons
        """
        # Default to visual element categories if not specified
        if target_categories is None:
            target_categories = ['data-point', 'data-line', 'data-bar', 'boxplot']
        
        self.target_categories = target_categories
        self.simplify_tolerance = simplify_tolerance
        self.min_polygon_points = min_polygon_points
        self.use_rle = use_rle
        
        # Validate categories
        for cat in target_categories:
            if cat not in self.CATEGORY_MAPPING:
                logger.warning(f"Unknown category: {cat}. Available: {list(self.CATEGORY_MAPPING.keys())}")
    
    def get_category_id(self, category_name: str) -> int:
        """Get category ID for a given category name."""
        return self.CATEGORY_MAPPING.get(category_name, -1)
    
    def create_category_info(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Create COCO category info for specified categories.
        
        Args:
            categories: List of category names. If None, uses target_categories.
            
        Returns:
            List containing category info dictionaries
        """
        if categories is None:
            categories = self.target_categories
        
        category_info = []
        for cat_name in categories:
            cat_id = self.get_category_id(cat_name)
            if cat_id >= 0:
                # Determine supercategory
                if 'data' in cat_name or cat_name == 'boxplot':
                    supercategory = 'visual_element'
                elif 'axis' in cat_name or 'tick' in cat_name:
                    supercategory = 'axis_element'
                elif 'legend' in cat_name:
                    supercategory = 'legend_element'
                elif cat_name in ['title', 'subtitle']:
                    supercategory = 'text_element'
                else:
                    supercategory = 'chart_element'
                
                category_info.append({
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": supercategory
                })
        
        return category_info
    
    def create_annotation_for_category(self, 
                                     mask: np.ndarray, 
                                     annotation_id: int, 
                                     image_id: int,
                                     category_name: str,
                                     additional_info: Dict = None) -> Dict[str, Any]:
        """
        Create a COCO annotation for a specific category.
        
        Args:
            mask: Binary mask
            annotation_id: Unique annotation ID
            image_id: Image ID this annotation belongs to
            category_name: Name of the category
            additional_info: Additional metadata to include
            
        Returns:
            COCO annotation dictionary
        """
        category_id = self.get_category_id(category_name)
        if category_id < 0:
            raise ValueError(f"Unknown category: {category_name}")
        
        # Calculate bbox and area
        bbox = self.calculate_bbox_from_mask(mask)
        area = self.calculate_area_from_mask(mask)
        
        # Create segmentation
        if self.use_rle:
            segmentation = self.mask_to_rle(mask)
        else:
            polygons = self.mask_to_polygon(mask)
            segmentation = polygons
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": 0
        }
        
        # Add additional metadata if provided
        if additional_info:
            for key, value in additional_info.items():
                if key not in annotation:  # Don't override standard fields
                    annotation[key] = value
        
        return annotation
    
    def create_coco_dataset_multi_category(self, 
                                         image_mask_pairs: List[Tuple[Union[str, Path], Dict[str, List[np.ndarray]]]],
                                         dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete COCO dataset from image-mask pairs with multiple categories.
        
        Args:
            image_mask_pairs: List of (image_path, category_masks_dict) tuples
                            where category_masks_dict maps category names to mask lists
            dataset_info: Optional dataset metadata
            
        Returns:
            Complete COCO dataset dictionary
        """
        if dataset_info is None:
            dataset_info = {
                "description": "Multi-category chart element instance segmentation dataset",
                "version": "2.0",
                "year": datetime.now().year,
                "contributor": "Enhanced mask generation system",
                "date_created": datetime.now().isoformat(),
                "categories_supported": self.target_categories
            }
        
        # Collect all unique categories from the data
        all_categories = set()
        for _, category_masks in image_mask_pairs:
            all_categories.update(category_masks.keys())
        
        # Filter to target categories
        valid_categories = [cat for cat in all_categories if cat in self.target_categories]
        
        # Initialize COCO structure
        coco_dataset = {
            "info": dataset_info,
            "licenses": [],
            "categories": self.create_category_info(valid_categories),
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        
        for image_id, (image_path, category_masks) in enumerate(image_mask_pairs, 1):
            # Add image info
            image_info = self.create_image_info(image_path, image_id)
            coco_dataset["images"].append(image_info)
            
            # Add annotations for each category
            for category_name, masks in category_masks.items():
                if category_name not in self.target_categories:
                    continue
                
                for mask in masks:
                    try:
                        annotation = self.create_annotation_for_category(
                            mask, annotation_id, image_id, category_name
                        )
                        coco_dataset["annotations"].append(annotation)
                        annotation_id += 1
                    except Exception as e:
                        logger.warning(f"Failed to create annotation for {category_name}: {e}")
        
        # Add statistics to info
        category_counts = {}
        for ann in coco_dataset["annotations"]:
            cat_id = ann["category_id"]
            # Find category name
            cat_name = None
            for cat in coco_dataset["categories"]:
                if cat["id"] == cat_id:
                    cat_name = cat["name"]
                    break
            
            if cat_name:
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        coco_dataset["info"]["annotation_counts"] = category_counts
        
        total_annotations = len(coco_dataset["annotations"])
        logger.info(f"Created multi-category COCO dataset:")
        logger.info(f"  - Images: {len(coco_dataset['images'])}")
        logger.info(f"  - Total annotations: {total_annotations}")
        logger.info(f"  - Categories: {list(category_counts.keys())}")
        for cat, count in category_counts.items():
            logger.info(f"    • {cat}: {count} annotations")
        
        return coco_dataset
    
    def add_masks_to_existing_coco_multi_category(self, 
                                                existing_coco_path: Union[str, Path],
                                                image_mask_pairs: List[Tuple[Union[str, Path], Dict[str, List[np.ndarray]]]],
                                                output_path: Union[str, Path]) -> None:
        """
        Add multi-category mask annotations to an existing COCO dataset.
        
        Args:
            existing_coco_path: Path to existing COCO JSON file
            image_mask_pairs: List of (image_path, category_masks_dict) tuples
            output_path: Path to save the updated COCO dataset
        """
        # Load existing dataset
        with open(existing_coco_path, 'r') as f:
            coco_data = json.load(f)
        
        # Get next available IDs
        max_image_id = max(img["id"] for img in coco_data["images"]) if coco_data["images"] else 0
        max_annotation_id = max(ann["id"] for ann in coco_data["annotations"]) if coco_data["annotations"] else 0
        
        # Create filename to image_id mapping
        filename_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
        
        # Ensure all target categories exist
        existing_categories = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        for cat_name in self.target_categories:
            if cat_name not in existing_categories:
                cat_id = self.get_category_id(cat_name)
                if cat_id >= 0:
                    new_category = self.create_category_info([cat_name])[0]
                    coco_data["categories"].append(new_category)
                    logger.info(f"Added new category: {cat_name} (ID: {cat_id})")
        
        annotation_id = max_annotation_id + 1
        added_annotations = 0
        
        for image_path, category_masks in image_mask_pairs:
            image_path = Path(image_path)
            filename = image_path.name
            
            # Check if image already exists
            if filename in filename_to_id:
                image_id = filename_to_id[filename]
                logger.debug(f"Adding masks to existing image: {filename}")
            else:
                # Add new image
                max_image_id += 1
                image_id = max_image_id
                image_info = self.create_image_info(image_path, image_id)
                coco_data["images"].append(image_info)
                logger.debug(f"Added new image: {filename}")
            
            # Add mask annotations for each category
            for category_name, masks in category_masks.items():
                if category_name not in self.target_categories:
                    continue
                
                for mask in masks:
                    try:
                        annotation = self.create_annotation_for_category(
                            mask, annotation_id, image_id, category_name
                        )
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
                        added_annotations += 1
                    except Exception as e:
                        logger.warning(f"Failed to add annotation for {category_name}: {e}")
        
        # Update dataset info
        if "info" not in coco_data:
            coco_data["info"] = {}
        
        coco_data["info"]["last_updated"] = datetime.now().isoformat()
        coco_data["info"]["enhanced_with_masks"] = True
        coco_data["info"]["mask_categories"] = self.target_categories
        
        # Save updated dataset
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Enhanced COCO dataset saved to {output_path}")
        logger.info(f"Added {added_annotations} mask annotations across {len(self.target_categories)} categories")
    
    def save_coco_dataset(self, 
                         coco_dataset: Dict[str, Any], 
                         output_path: Union[str, Path]) -> None:
        """Save multi-category COCO dataset to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
        
        logger.info(f"Multi-category COCO dataset saved to {output_path}")
        
        # Log category statistics
        if "info" in coco_dataset and "annotation_counts" in coco_dataset["info"]:
            logger.info("Category breakdown:")
            for cat, count in coco_dataset["info"]["annotation_counts"].items():
                logger.info(f"  • {cat}: {count} annotations") 