#!/usr/bin/env python3
"""
Enhanced batch mask generation for multiple chart element types.

This script processes chart elements (data points, bars, boxplots, etc.)
and generates masks for each category, creating comprehensive COCO annotation files
with segmentation masks for visual elements.

Note: Data lines are excluded as they don't require segmentation masks.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_mask_generator import EnhancedMaskGenerator, ElementBBox
from enhanced_coco_converter import EnhancedCOCOMaskConverter
from utils import filter_valid_masks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_with_categories(ann_file: Path, target_categories: List[str]) -> Tuple[Dict, Dict, List]:
    """Load dataset and extract image info with specified categories."""
    logger.info(f"Loading dataset: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Create category name to ID mapping
    category_mapping = {}
    for cat in data['categories']:
        category_mapping[cat['name']] = cat['id']
    
    logger.info(f"Available categories: {list(category_mapping.keys())}")
    
    # Find target category IDs
    target_category_ids = []
    for cat_name in target_categories:
        if cat_name in category_mapping:
            target_category_ids.append(category_mapping[cat_name])
            logger.info(f"Will process {cat_name} (ID: {category_mapping[cat_name]})")
        else:
            logger.warning(f"Category {cat_name} not found in dataset")
    
    if not target_category_ids:
        raise ValueError(f"None of the target categories {target_categories} found in dataset")
    
    # Create filename to image mapping
    filename_to_image = {img['file_name']: img for img in data['images']}
    
    # Group annotations by image and category
    image_annotations = {}
    for ann in data['annotations']:
        if ann['category_id'] in target_category_ids:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = {}
            
            # Find category name
            cat_name = None
            for cat in data['categories']:
                if cat['id'] == ann['category_id']:
                    cat_name = cat['name']
                    break
            
            if cat_name:
                if cat_name not in image_annotations[image_id]:
                    image_annotations[image_id][cat_name] = []
                image_annotations[image_id][cat_name].append(ann)
    
    # Filter to images with target elements
    images_with_elements = []
    for img in data['images']:
        if img['id'] in image_annotations:
            images_with_elements.append(img)
    
    total_annotations = sum(
        sum(len(anns) for anns in img_anns.values()) 
        for img_anns in image_annotations.values()
    )
    
    logger.info(f"Found {len(images_with_elements)} images with target elements")
    logger.info(f"Total target annotations: {total_annotations}")
    
    # Log category breakdown
    category_counts = {}
    for img_anns in image_annotations.values():
        for cat_name, anns in img_anns.items():
            category_counts[cat_name] = category_counts.get(cat_name, 0) + len(anns)
    
    for cat_name, count in category_counts.items():
        logger.info(f"  - {cat_name}: {count} annotations")
    
    return data, image_annotations, images_with_elements


def process_single_image_multi_category(image_path: Path, 
                                       category_annotations: Dict[str, List[Dict]], 
                                       mask_generator: EnhancedMaskGenerator,
                                       max_masks_per_category: int = 100) -> Dict[str, List[np.ndarray]]:
    """Process a single image to generate masks for multiple categories."""
    
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return {}
    
    try:
        # Prepare element bboxes for all categories
        all_element_bboxes = []
        for cat_name, annotations in category_annotations.items():
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                element_bbox = ElementBBox(
                    x=bbox[0], y=bbox[1], 
                    width=bbox[2], height=bbox[3],
                    category_name=cat_name,
                    category_id=ann['category_id'],
                    additional_info=ann
                )
                all_element_bboxes.append(element_bbox)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return {}
        
        # Generate masks by category
        category_masks = mask_generator.generate_masks_by_category(image, all_element_bboxes, image_path)
        
        # Filter and limit masks for each category (more permissive)
        filtered_category_masks = {}
        for cat_name, masks in category_masks.items():
            if masks:
                # Filter for validity (lower thresholds)
                valid_masks = filter_valid_masks(masks, min_area=2, max_area=50000)
                
                # Limit number of masks per category
                if len(valid_masks) > max_masks_per_category:
                    logger.warning(f"Too many {cat_name} masks ({len(valid_masks)}) for {image_path.name}, limiting to {max_masks_per_category}")
                    valid_masks = valid_masks[:max_masks_per_category]
                
                if valid_masks:
                    filtered_category_masks[cat_name] = valid_masks
        
        total_masks = sum(len(masks) for masks in filtered_category_masks.values())
        logger.debug(f"Generated {total_masks} valid masks for {image_path.name}: {[(cat, len(masks)) for cat, masks in filtered_category_masks.items()]}")
        
        return filtered_category_masks
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return {}


def generate_multi_category_masks(train_ann_file: str = "legend_data/annotations_JSON_cleaned/train_enriched.json",
                                 val_ann_file: str = "legend_data/annotations_JSON_cleaned/val_enriched_with_info.json",
                                 images_dir: str = "legend_data/train/images",
                                 output_dir: str = "legend_match_swin/mask_generation/enhanced_datasets",
                                 target_categories: List[str] = None,
                                 max_images: int = None):
    """Generate masks for multiple chart element categories."""
    
    # Default target categories for visual elements (excluding data-line)
    if target_categories is None:
        target_categories = ['data-point', 'data-bar', 'boxplot']
    
    # Setup paths
    train_ann_path = Path(train_ann_file)
    val_ann_path = Path(val_ann_file)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not train_ann_path.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_ann_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    logger.info("🎭 Starting multi-category mask generation")
    logger.info(f"📁 Images directory: {images_path}")
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"🎯 Target categories: {target_categories}")
    
    # Initialize enhanced mask generator (more permissive settings)
    mask_generator = EnhancedMaskGenerator(
        color_tolerance=40,        # More permissive color matching
        min_contour_area=3,        # Lower minimum area threshold
        max_contour_area=5000,     # Back to original max area
        circle_min_radius=1,       # Detect smaller circles
        circle_max_radius=30       # Detect larger circles
    )
    
    # Initialize enhanced COCO converter (optimized for small data points)
    converter = EnhancedCOCOMaskConverter(
        target_categories=target_categories,
        simplify_tolerance=0.1,    # Very low tolerance to preserve small data points
        min_polygon_points=3       # Allow triangles (simplified circles)
    )
    
    # Process datasets
    for split, ann_file in [("train", train_ann_path), ("val", val_ann_path)]:
        if not ann_file.exists():
            logger.warning(f"Skipping {split} - annotation file not found: {ann_file}")
            continue
            
        logger.info(f"\n🔄 Processing {split} split...")
        
        # Load dataset
        try:
            data, image_annotations, images_with_elements = load_dataset_with_categories(
                ann_file, target_categories
            )
        except Exception as e:
            logger.error(f"Error loading {split} annotations: {e}")
            continue
        
        # Limit images for testing
        if max_images:
            images_with_elements = images_with_elements[:max_images]
            logger.info(f"Limited to first {len(images_with_elements)} images for testing")
        
        # Process images
        image_mask_pairs = []
        successful_images = 0
        total_masks_by_category = {}
        
        with tqdm(images_with_elements, desc=f"Processing {split} images") as pbar:
            for img_info in pbar:
                image_path = images_path / img_info['file_name']
                
                # Get category annotations for this image
                category_annotations = image_annotations.get(img_info['id'], {})
                
                # Generate masks for all categories
                category_masks = process_single_image_multi_category(
                    image_path, category_annotations, mask_generator
                )
                
                if category_masks:
                    image_mask_pairs.append((str(image_path), category_masks))
                    successful_images += 1
                    
                    # Update totals
                    for cat_name, masks in category_masks.items():
                        total_masks_by_category[cat_name] = total_masks_by_category.get(cat_name, 0) + len(masks)
                    
                    total_masks = sum(len(masks) for masks in category_masks.values())
                    pbar.set_postfix({
                        'categories': len(category_masks),
                        'masks': total_masks,
                        'success_rate': f"{successful_images}/{len(image_mask_pairs)}"
                    })
        
        logger.info(f"✅ {split} processing complete:")
        logger.info(f"   - Successful images: {successful_images}/{len(images_with_elements)}")
        logger.info(f"   - Total masks by category:")
        for cat_name, count in total_masks_by_category.items():
            logger.info(f"     • {cat_name}: {count} masks")
        
        total_masks = sum(total_masks_by_category.values())
        logger.info(f"   - Total masks: {total_masks}")
        if successful_images > 0:
            logger.info(f"   - Average masks per image: {total_masks/successful_images:.1f}")
        
        if image_mask_pairs:
            # Create enhanced dataset with multi-category masks
            enhanced_ann_file = output_path / f"{split}_enriched_with_multi_masks.json"
            
            logger.info(f"💾 Creating new dataset with integrated masks: {enhanced_ann_file}")
            
            # Create a completely new dataset instead of adding to existing
            new_dataset = {
                "info": {
                    "description": f"{split.capitalize()} dataset with integrated masks",
                    "version": "2.0",
                    "year": 2024,
                    "contributor": "Enhanced mask generation system",
                    "date_created": "2024-12-19",
                    "enhanced_with_masks": True,
                    "mask_categories": target_categories
                },
                "licenses": [],
                "categories": data['categories'],
                "images": [],
                "annotations": []
            }
            
            # Create filename to image mapping
            filename_to_id = {img['file_name']: img for img in data['images']}
            
            # Process each image with masks
            annotation_id = 1
            total_masks_added = 0
            
            for image_path, category_masks in image_mask_pairs:
                image_path = Path(image_path)
                filename = image_path.name
                
                # Find corresponding image info
                if filename in filename_to_id:
                    image_info = filename_to_id[filename].copy()
                    image_id = image_info['id']
                    
                    # Add image to new dataset
                    new_dataset['images'].append(image_info)
                    
                    # Get original annotations for this image
                    original_annotations = []
                    for ann in data['annotations']:
                        if ann['image_id'] == image_id:
                            original_annotations.append(ann)
                    
                    # Add original annotations (these will be updated with masks if available)
                    for ann in original_annotations:
                        new_ann = ann.copy()
                        new_ann['id'] = annotation_id
                        new_ann['image_id'] = image_id
                        
                        # Check if we have a mask for this annotation
                        has_mask = False
                        for cat_name, masks in category_masks.items():
                            if cat_name in target_categories:
                                # For now, just add the original annotation
                                # The mask will be added as a separate annotation
                                pass
                        
                        new_dataset['annotations'].append(new_ann)
                        annotation_id += 1
                    
                    # Add mask annotations as separate entries
                    for category_name, masks in category_masks.items():
                        if category_name not in target_categories:
                            continue
                        
                        for mask in masks:
                            try:
                                mask_annotation = converter.create_annotation_for_category(
                                    mask, annotation_id, image_id, category_name
                                )
                                
                                # Only add if segmentation is not empty
                                if mask_annotation['segmentation'] and len(mask_annotation['segmentation']) > 0:
                                    new_dataset['annotations'].append(mask_annotation)
                                    annotation_id += 1
                                    total_masks_added += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to create mask annotation for {category_name}: {e}")
                    
                    logger.debug(f"Processed {filename}: {len(category_masks.get('data-point', []))} data-point masks")
            
            # Save the new dataset
            with open(enhanced_ann_file, 'w') as f:
                json.dump(new_dataset, f, indent=2)
            
            logger.info(f"✅ Saved {split} dataset with integrated masks: {enhanced_ann_file}")
            logger.info(f"📊 Total masks added: {total_masks_added}")
        else:
            logger.warning(f"No masks generated for {split} split")
    
    logger.info("🎉 Multi-category mask generation complete!")
    logger.info(f"📁 Enhanced datasets saved in: {output_path}")


def main():
    """Main function with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate masks for multiple chart element categories")
    parser.add_argument("--train_ann", type=str, 
                       default="legend_data/annotations_JSON_cleaned/train_enriched.json",
                       help="Path to training annotations")
    parser.add_argument("--val_ann", type=str,
                       default="legend_data/annotations_JSON_cleaned/val_enriched_with_info.json", 
                       help="Path to validation annotations")
    parser.add_argument("--images_dir", type=str,
                       default="legend_data/train/images",
                       help="Path to images directory")
    parser.add_argument("--output_dir", type=str,
                       default="legend_match_swin/mask_generation/enhanced_datasets",
                       help="Output directory for enhanced datasets")
    parser.add_argument("--categories", type=str, nargs='+',
                       default=['data-point', 'data-bar', 'boxplot'],
                       help="Target categories to generate masks for (data-line excluded)")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Limit number of images (for testing)")
    parser.add_argument("--test_run", action="store_true",
                       help="Run on first 5 images only (for testing)")
    
    args = parser.parse_args()
    
    # Override max_images for test run
    if args.test_run:
        args.max_images = 5
        logger.info("🧪 Test run mode: processing first 5 images only")
    
    try:
        generate_multi_category_masks(
            train_ann_file=args.train_ann,
            val_ann_file=args.val_ann,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            target_categories=args.categories,
            max_images=args.max_images
        )
        return 0
    except Exception as e:
        logger.error(f"❌ Multi-category batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 