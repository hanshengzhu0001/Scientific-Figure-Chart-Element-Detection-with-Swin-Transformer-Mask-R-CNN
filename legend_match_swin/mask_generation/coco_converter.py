"""
COCO format converter for mask annotations.

Converts OpenCV masks to COCO format with polygon segmentation annotations
suitable for instance segmentation training.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import json
from pathlib import Path
import logging
from datetime import datetime
import pycocotools.mask as mask_utils

logger = logging.getLogger(__name__)


class COCOMaskConverter:
    """
    Converts OpenCV masks to COCO format annotations with segmentation polygons.
    
    Supports both polygon and RLE (Run-Length Encoding) formats for segmentation.
    """
    
    def __init__(self, 
                 data_point_category_id: int = 11,
                 simplify_tolerance: float = 0.1,  # Much lower for small data points  
                 min_polygon_points: int = 3,      # Allow triangles (simplified circles)
                 use_rle: bool = False):
        """
        Initialize the COCO converter.
        
        Args:
            data_point_category_id: Category ID for data points in COCO format
            simplify_tolerance: Tolerance for polygon simplification (higher = simpler)
            min_polygon_points: Minimum number of points required for a valid polygon
            use_rle: Whether to use RLE format instead of polygons
        """
        self.data_point_category_id = data_point_category_id
        self.simplify_tolerance = simplify_tolerance
        self.min_polygon_points = min_polygon_points
        self.use_rle = use_rle
        
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert a binary mask to polygon format.
        
        Args:
            mask: Binary mask (0s and 255s)
            
        Returns:
            List of polygons, each polygon is a list of [x1,y1,x2,y2,...] coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Simplify contour
            epsilon = self.simplify_tolerance * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) >= self.min_polygon_points:
                # Convert to flat list of coordinates
                polygon = simplified.reshape(-1, 2).flatten().astype(float).tolist()
                polygons.append(polygon)
                
        return polygons
        
    def mask_to_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert a binary mask to RLE (Run-Length Encoding) format.
        
        Args:
            mask: Binary mask (0s and 255s)
            
        Returns:
            RLE dictionary compatible with COCO format
        """
        # Convert to binary (0s and 1s)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Convert to Fortran order for pycocotools
        binary_mask = np.asfortranarray(binary_mask)
        
        # Encode to RLE
        rle = mask_utils.encode(binary_mask)
        
        # Convert bytes to string for JSON serialization
        rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle
        
    def calculate_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """
        Calculate bounding box from mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Bounding box in COCO format [x, y, width, height]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0, 0, 0, 0]
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        return [x, y, w, h]
        
    def calculate_area_from_mask(self, mask: np.ndarray) -> int:
        """
        Calculate area from mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Area in pixels
        """
        return int(np.sum(mask > 0))
        
    def create_annotation(self, 
                         mask: np.ndarray, 
                         annotation_id: int, 
                         image_id: int,
                         category_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a single COCO annotation from a mask.
        
        Args:
            mask: Binary mask
            annotation_id: Unique annotation ID
            image_id: Image ID this annotation belongs to
            category_id: Category ID (uses default data_point_category_id if None)
            
        Returns:
            COCO annotation dictionary
        """
        if category_id is None:
            category_id = self.data_point_category_id
            
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
        
        return annotation
        
    def create_image_info(self, 
                         image_path: Union[str, Path], 
                         image_id: int) -> Dict[str, Any]:
        """
        Create COCO image info dictionary.
        
        Args:
            image_path: Path to the image file
            image_id: Unique image ID
            
        Returns:
            COCO image info dictionary
        """
        image_path = Path(image_path)
        
        # Load image to get dimensions
        import cv2
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat()
        }
        
        return image_info
        
    def create_category_info(self) -> List[Dict[str, Any]]:
        """
        Create COCO category info for data points.
        
        Returns:
            List containing category info for data points
        """
        return [{
            "id": self.data_point_category_id,
            "name": "data-point",
            "supercategory": "chart-element"
        }]
        
    def create_coco_dataset(self, 
                           image_mask_pairs: List[Tuple[Union[str, Path], List[np.ndarray]]],
                           dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete COCO dataset from image-mask pairs.
        
        Args:
            image_mask_pairs: List of (image_path, masks) tuples
            dataset_info: Optional dataset metadata
            
        Returns:
            Complete COCO dataset dictionary
        """
        if dataset_info is None:
            dataset_info = {
                "description": "Data point instance segmentation dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Automatic mask generation",
                "date_created": datetime.now().isoformat()
            }
            
        # Initialize COCO structure
        coco_dataset = {
            "info": dataset_info,
            "licenses": [],
            "categories": self.create_category_info(),
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        
        for image_id, (image_path, masks) in enumerate(image_mask_pairs, 1):
            # Add image info
            image_info = self.create_image_info(image_path, image_id)
            coco_dataset["images"].append(image_info)
            
            # Add annotations for each mask
            for mask in masks:
                annotation = self.create_annotation(mask, annotation_id, image_id)
                coco_dataset["annotations"].append(annotation)
                annotation_id += 1
                
        logger.info(f"Created COCO dataset with {len(coco_dataset['images'])} images "
                   f"and {len(coco_dataset['annotations'])} annotations")
        
        return coco_dataset
        
    def add_masks_to_existing_coco(self, 
                                  existing_coco_path: Union[str, Path],
                                  image_mask_pairs: List[Tuple[Union[str, Path], List[np.ndarray]]],
                                  output_path: Union[str, Path]) -> None:
        """
        Add mask annotations to an existing COCO dataset.
        
        Args:
            existing_coco_path: Path to existing COCO JSON file
            image_mask_pairs: List of (image_path, masks) tuples to add
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
        
        annotation_id = max_annotation_id + 1
        
        for image_path, masks in image_mask_pairs:
            image_path = Path(image_path)
            filename = image_path.name
            
            # Check if image already exists
            if filename in filename_to_id:
                image_id = filename_to_id[filename]
                logger.info(f"Adding masks to existing image: {filename}")
            else:
                # Add new image
                max_image_id += 1
                image_id = max_image_id
                image_info = self.create_image_info(image_path, image_id)
                coco_data["images"].append(image_info)
                logger.info(f"Added new image: {filename}")
                
            # Add mask annotations
            for mask in masks:
                annotation = self.create_annotation(mask, annotation_id, image_id)
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                
        # Ensure data-point category exists
        category_exists = any(cat["name"] == "data-point" for cat in coco_data["categories"])
        if not category_exists:
            coco_data["categories"].extend(self.create_category_info())
            
        # Save updated dataset
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        logger.info(f"Saved updated COCO dataset to {output_path}")
        
    def save_coco_dataset(self, 
                         coco_dataset: Dict[str, Any], 
                         output_path: Union[str, Path]) -> None:
        """
        Save COCO dataset to JSON file.
        
        Args:
            coco_dataset: COCO dataset dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
            
        logger.info(f"Saved COCO dataset to {output_path}")
        
    def validate_coco_dataset(self, coco_dataset: Dict[str, Any]) -> bool:
        """
        Validate COCO dataset structure.
        
        Args:
            coco_dataset: COCO dataset dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["info", "licenses", "categories", "images", "annotations"]
        
        for key in required_keys:
            if key not in coco_dataset:
                logger.error(f"Missing required key: {key}")
                return False
                
        # Validate annotations
        for ann in coco_dataset["annotations"]:
            required_ann_keys = ["id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"]
            for key in required_ann_keys:
                if key not in ann:
                    logger.error(f"Missing required annotation key: {key}")
                    return False
                    
        logger.info("COCO dataset validation passed")
        return True 