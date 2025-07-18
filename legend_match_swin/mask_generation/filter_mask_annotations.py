#!/usr/bin/env python3
"""
Filter annotations to only include those with valid segmentation masks.
This is necessary for Mask R-CNN training since COCO evaluation crashes on empty segmentation arrays.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def has_valid_segmentation(annotation: Dict[str, Any]) -> bool:
    """Check if annotation has valid segmentation data."""
    segmentation = annotation.get('segmentation', [])
    
    # Check if segmentation is not empty
    if not segmentation:
        return False
    
    # Check if segmentation has valid polygon data
    if isinstance(segmentation, list):
        # Should be a list of polygon coordinates
        if len(segmentation) > 0:
            # Check if first polygon has coordinates
            if isinstance(segmentation[0], list) and len(segmentation[0]) >= 6:
                return True
    
    return False

def filter_annotations(input_path: Path, output_path: Path) -> None:
    """Filter annotations to only include those with valid masks."""
    logger.info(f"Loading annotations from {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['annotations'])
    logger.info(f"Original annotations: {original_count}")
    
    # Filter annotations with valid segmentation
    filtered_annotations = []
    categories_with_masks = set()
    
    for ann in data['annotations']:
        if has_valid_segmentation(ann):
            filtered_annotations.append(ann)
            categories_with_masks.add(ann['category_id'])
        else:
            logger.debug(f"Skipping annotation {ann['id']} - no valid segmentation")
    
    filtered_count = len(filtered_annotations)
    logger.info(f"Filtered annotations: {filtered_count}")
    logger.info(f"Removed {original_count - filtered_count} annotations without masks")
    logger.info(f"Categories with masks: {sorted(categories_with_masks)}")
    
    # Update the data
    data['annotations'] = filtered_annotations
    
    # Save filtered data
    logger.info(f"Saving filtered annotations to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✅ Filtered dataset saved: {filtered_count}/{original_count} annotations kept")

def main():
    """Filter both train and validation datasets."""
    base_dir = Path("legend_match_swin/mask_generation/enhanced_datasets")
    
    # Filter training data
    train_input = base_dir / "train_filtered_with_segmentation.json"
    train_output = base_dir / "train_filtered_with_masks_only.json"
    
    if train_input.exists():
        logger.info("=" * 50)
        logger.info("FILTERING TRAINING DATASET")
        logger.info("=" * 50)
        filter_annotations(train_input, train_output)
    else:
        logger.error(f"Training file not found: {train_input}")
    
    # Filter validation data
    val_input = base_dir / "val_enriched_with_multi_masks.json"
    val_output = base_dir / "val_enriched_with_masks_only.json"
    
    if val_input.exists():
        logger.info("=" * 50)
        logger.info("FILTERING VALIDATION DATASET")
        logger.info("=" * 50)
        filter_annotations(val_input, val_output)
    else:
        logger.error(f"Validation file not found: {val_input}")

if __name__ == "__main__":
    main() 