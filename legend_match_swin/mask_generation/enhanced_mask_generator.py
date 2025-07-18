#!/usr/bin/env python3
"""
Enhanced mask generator for multiple chart element types.

Extends the DataPointMaskGenerator to handle:
- Data points (circles, squares, triangles, etc.)
- Data bars (simple rectangle masks)
- Boxplots (box shapes with whiskers)
- Other chart elements

Note: Data lines are excluded as they don't require segmentation masks.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
from pathlib import Path
import logging
from dataclasses import dataclass

from mask_generator import DataPointMaskGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ElementBBox:
    """Represents a chart element bounding box with category info."""
    x: float
    y: float
    width: float
    height: float
    category_name: str
    category_id: int
    additional_info: dict = None


class EnhancedMaskGenerator(DataPointMaskGenerator):
    """
    Enhanced mask generator for multiple chart element types.
    
    Supports mask generation for:
    - Data points: Small geometric shapes (circles, squares, triangles, etc.)
    - Data bars: Simple rectangular masks (vertical/horizontal bars)
    - Boxplots: Box-and-whisker plot elements
    - Other visual elements based on shape characteristics
    
    Note: Data lines are excluded as they're better handled as line annotations
    rather than segmentation masks.
    """
    
    def __init__(self, **kwargs):
        """Initialize with enhanced parameters for multiple element types."""
        super().__init__(**kwargs)
        
        # Element-specific parameters (no line params - lines don't need masks)
        self.bar_params = {
            'simple_rectangles': True  # Use simple rectangle masks for bars
        }
        
        self.boxplot_params = {
            'whisker_detection': True,
            'outlier_detection': True,
            'median_line_detection': True
        }
    
    # Data lines no longer generate masks - they are better handled as line annotations
    
    def generate_bar_masks(self, image: np.ndarray, bar_bboxes: List[ElementBBox]) -> List[np.ndarray]:
        """Generate simple rectangle masks for data bars."""
        logger.info(f"Generating rectangle masks for {len(bar_bboxes)} data bars")
        
        masks = []
        
        for bbox in bar_bboxes:
            try:
                # Extract bbox coordinates
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                
                # Validate bbox dimensions
                if w <= 0 or h <= 0:
                    continue
                
                # Create simple rectangle mask (bars are rectangular by nature)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
                masks.append(mask)
                    
            except Exception as e:
                logger.warning(f"Error generating bar mask: {e}")
                
        logger.info(f"Generated {len(masks)} rectangle bar masks")
        return masks
    
    def generate_boxplot_masks(self, image: np.ndarray, boxplot_bboxes: List[ElementBBox]) -> List[np.ndarray]:
        """Generate masks for boxplot elements."""
        logger.info(f"Generating masks for {len(boxplot_bboxes)} boxplots")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masks = []
        
        for bbox in boxplot_bboxes:
            try:
                # Extract ROI
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                roi = gray[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Detect rectangular box (main body of boxplot)
                edges = cv2.Canny(roi, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
                # Find the main box (largest rectangular contour)
                best_contour = None
                best_score = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 20:
                        continue
                    
                    # Check rectangularity
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:  # Rectangular
                        score = area
                        if score > best_score:
                            best_score = score
                            best_contour = contour
                
                if best_contour is not None:
                    # Adjust coordinates and draw
                    adjusted_contour = best_contour + np.array([x, y])
                    cv2.fillPoly(mask, [adjusted_contour], 255)
                    
                    # Also detect whiskers (lines extending from box)
                    if self.boxplot_params['whisker_detection']:
                        # Look for vertical/horizontal lines extending from the box
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                              minLineLength=10, maxLineGap=5)
                        
                        if lines is not None:
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                # Adjust coordinates and draw whisker
                                cv2.line(mask, (x1+x, y1+y), (x2+x, y2+y), 255, 2)
                    
                    masks.append(mask)
                    
            except Exception as e:
                logger.warning(f"Error generating boxplot mask: {e}")
                
        logger.info(f"Generated {len(masks)} boxplot masks")
        return masks
    
    def generate_masks_by_category(self, image: np.ndarray, 
                                  element_bboxes: List[ElementBBox], 
                                  image_path: Union[str, Path] = None) -> Dict[str, List[np.ndarray]]:
        """Generate masks for multiple element categories."""
        logger.info(f"Generating masks for {len(element_bboxes)} elements across multiple categories")
        
        # Group bboxes by category
        category_groups = {}
        for bbox in element_bboxes:
            if bbox.category_name not in category_groups:
                category_groups[bbox.category_name] = []
            category_groups[bbox.category_name].append(bbox)
        
        all_masks = {}
        
        for category, bboxes in category_groups.items():
            logger.info(f"Processing {len(bboxes)} {category} elements")
            
            if category == 'data-point':
                # Use existing data point detection
                bbox_list = [[b.x, b.y, b.width, b.height] for b in bboxes]
                if image_path is not None:
                    # Use image path for parent method
                    mask_results = super().generate_masks(
                        image_path, bbox_list=bbox_list, 
                        methods=['color', 'circles', 'contours']
                    )
                else:
                    # Fallback: call parent method with image array
                    mask_results = super().generate_masks(
                        image, bbox_list=bbox_list, 
                        methods=['color', 'circles', 'contours']
                    )
                # Combine all methods
                masks = []
                for method_masks in mask_results.values():
                    masks.extend(method_masks)
                all_masks[category] = masks
                
            elif category == 'data-line':
                # Data lines no longer generate masks
                logger.info(f"Skipping mask generation for {category} - not needed for line elements")
                all_masks[category] = []
                
            elif category == 'data-bar':
                all_masks[category] = self.generate_bar_masks(image, bboxes)
                
            elif category == 'boxplot':
                all_masks[category] = self.generate_boxplot_masks(image, bboxes)
                
            else:
                # Generic approach for other elements
                logger.info(f"Using generic mask generation for {category}")
                masks = []
                for bbox in bboxes:
                    # Simple bbox-based mask
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    masks.append(mask)
                all_masks[category] = masks
        
        total_masks = sum(len(masks) for masks in all_masks.values())
        logger.info(f"Generated {total_masks} total masks across {len(all_masks)} categories")
        
        return all_masks
    
    def generate_masks_from_annotations(self, image_path: Union[str, Path], 
                                      annotations: List[Dict]) -> Dict[str, List[np.ndarray]]:
        """Generate masks from COCO-style annotations with multiple categories."""
        
        # Load image
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert annotations to ElementBBox objects
        element_bboxes = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_name = ann.get('category_name', f"category_{ann.get('category_id', 0)}")
            category_id = ann.get('category_id', 0)
            
            element_bbox = ElementBBox(
                x=bbox[0], y=bbox[1], 
                width=bbox[2], height=bbox[3],
                category_name=category_name,
                category_id=category_id,
                additional_info=ann
            )
            element_bboxes.append(element_bbox)
        
        # Generate masks by category
        return self.generate_masks_by_category(image, element_bboxes, image_path) 