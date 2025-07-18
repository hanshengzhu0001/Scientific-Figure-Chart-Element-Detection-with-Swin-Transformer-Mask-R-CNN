#!/usr/bin/env python3
"""
Demo script for data point mask generation.

This script demonstrates the complete workflow:
1. Load existing COCO annotations with data point bboxes
2. Generate masks using multiple OpenCV methods
3. Convert masks to COCO format
4. Save the enhanced dataset with segmentation masks

Usage:
    python generate_masks_demo.py --input_image path/to/image.jpg --input_coco path/to/annotations.json
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import cv2
import numpy as np

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mask_generation import DataPointMaskGenerator, COCOMaskConverter
from mask_generation.utils import (
    visualize_masks, visualize_mask_comparison, analyze_mask_distribution,
    create_mask_summary_image, filter_valid_masks
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_data_point_bboxes(coco_file: Path, data_point_category_name: str = "data-point") -> dict:
    """
    Extract data point bounding boxes from existing COCO annotations.
    
    Args:
        coco_file: Path to COCO JSON file
        data_point_category_name: Name of the data point category
        
    Returns:
        Dictionary mapping image filenames to lists of bboxes
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find data point category ID
    data_point_cat_id = None
    for cat in coco_data['categories']:
        if cat['name'] == data_point_category_name:
            data_point_cat_id = cat['id']
            break
    
    if data_point_cat_id is None:
        logger.warning(f"No category found with name '{data_point_category_name}'")
        return {}
    
    # Create filename to image ID mapping
    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Extract data point bboxes
    image_bboxes = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] == data_point_cat_id:
            image_id = ann['image_id']
            filename = id_to_filename.get(image_id)
            if filename:
                if filename not in image_bboxes:
                    image_bboxes[filename] = []
                # Convert COCO bbox [x, y, w, h] to list
                bbox = ann['bbox']
                image_bboxes[filename].append([int(x) for x in bbox])
    
    logger.info(f"Extracted data point bboxes for {len(image_bboxes)} images")
    for filename, bboxes in image_bboxes.items():
        logger.info(f"  {filename}: {len(bboxes)} data points")
    
    return image_bboxes


def process_single_image(image_path: Path, 
                        bboxes: list,
                        mask_generator: DataPointMaskGenerator,
                        output_dir: Path,
                        show_visualizations: bool = True) -> list:
    """
    Process a single image to generate masks.
    
    Args:
        image_path: Path to the image
        bboxes: List of data point bounding boxes
        mask_generator: Initialized mask generator
        output_dir: Directory for saving outputs
        show_visualizations: Whether to show visualizations
        
    Returns:
        List of generated masks
    """
    logger.info(f"Processing image: {image_path.name}")
    logger.info(f"  Found {len(bboxes)} data point bboxes")
    
    # Generate masks using all methods
    mask_results = mask_generator.generate_masks(
        image_path, 
        bbox_list=bboxes,
        methods=['color', 'circles', 'contours']
    )
    
    # Load image for visualization
    image = cv2.imread(str(image_path))
    
    if show_visualizations:
        # Show comparison of methods
        visualize_mask_comparison(image, mask_results, 
                                save_path=output_dir / f"{image_path.stem}_method_comparison.png")
    
    # Combine all masks and filter by bbox overlap
    all_masks = []
    for method_masks in mask_results.values():
        all_masks.extend(method_masks)
    
    if not all_masks:
        logger.warning("No masks generated for this image")
        return []
    
    # Filter masks to only those that overlap with data point bboxes (more permissive)
    filtered_masks = mask_generator.filter_masks_by_bbox(all_masks, bboxes, min_overlap=0.2)
    final_masks = [mask for mask, _ in filtered_masks]
    
    # Further filter for validity (lower thresholds)
    final_masks = filter_valid_masks(final_masks, min_area=2, max_area=5000)
    
    logger.info(f"  Generated {len(all_masks)} total masks")
    logger.info(f"  Filtered to {len(final_masks)} valid masks")
    
    if final_masks and show_visualizations:
        # Import clear visualization functions
        from clear_mask_visualizer import visualize_masks_clear, visualize_individual_masks
        
        # Create clear visualizations instead of standard ones
        print("🎨 Creating clear mask visualizations...")
        
        # 1. Clear multi-view display
        visualize_masks_clear(
            image, final_masks, bboxes,
            save_path=output_dir / f"{image_path.stem}_clear_masks.png",
            show_options=['overlay', 'masks_only', 'outlines']
        )
        
        # 2. Individual mask inspection (limit to 12 for clarity)
        if len(final_masks) > 12:
            print(f"📝 Showing first 12 of {len(final_masks)} masks for individual inspection")
            inspect_masks = final_masks[:12]
        else:
            inspect_masks = final_masks
            
        visualize_individual_masks(
            image, inspect_masks,
            save_dir=output_dir / f"{image_path.stem}_individual_masks"
        )
        
        # 3. Original analysis (keep for statistics)
        analysis = analyze_mask_distribution(final_masks)
        
        # Create summary visualization
        create_mask_summary_image(image, final_masks, analysis,
                                save_path=output_dir / f"{image_path.stem}_mask_summary.png")
    
    return final_masks


def main():
    parser = argparse.ArgumentParser(description="Generate data point masks from chart images")
    parser.add_argument("--input_image", type=str, required=False,
                       help="Path to input image")
    parser.add_argument("--input_coco", type=str, required=False,
                       default="legend_data/annotations_JSON_cleaned/train_enriched.json",
                       help="Path to input COCO annotations file")
    parser.add_argument("--output_dir", type=str, default="mask_generation_output",
                       help="Output directory for results")
    parser.add_argument("--data_point_category", type=str, default="data-point",
                       help="Name of data point category in COCO file")
    parser.add_argument("--auto_select_image", action="store_true",
                       help="Automatically select an image with data points from the COCO file")
    parser.add_argument("--no_visualizations", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--color_tolerance", type=int, default=40,
                       help="Color tolerance for mask generation (more permissive)")
    parser.add_argument("--min_area", type=int, default=3,
                       help="Minimum mask area (lower threshold)")
    parser.add_argument("--max_area", type=int, default=2000,
                       help="Maximum mask area")
    
    args = parser.parse_args()
    
    # Setup paths
    coco_path = Path(args.input_coco)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate COCO file
    if not coco_path.exists():
        logger.error(f"❌ COCO file not found: {coco_path}")
        logger.info(f"📁 Current working directory: {Path.cwd()}")
        logger.info(f"💡 Please check the path or run from the correct directory")
        return 1
    
    # Extract data point bboxes
    image_bboxes = extract_data_point_bboxes(coco_path, args.data_point_category)
    
    if not image_bboxes:
        logger.error(f"No data point annotations found in COCO file")
        logger.info(f"💡 Make sure the COCO file contains '{args.data_point_category}' category")
        return 1
    
    # Handle image selection
    if args.auto_select_image or not args.input_image:
        # Select first image with data points
        image_filename = list(image_bboxes.keys())[0]
        # Find the full image path
        possible_image_dirs = [
            Path("legend_data/train/images"),
            Path("images"),
            Path("."),
        ]
        
        image_path = None
        for img_dir in possible_image_dirs:
            candidate_path = img_dir / image_filename
            if candidate_path.exists():
                image_path = candidate_path
                break
        
        if image_path is None:
            logger.error(f"❌ Could not find image file: {image_filename}")
            logger.info(f"📁 Searched in: {[str(d) for d in possible_image_dirs]}")
            return 1
            
        logger.info(f"🎯 Auto-selected image: {image_path}")
    else:
        image_path = Path(args.input_image)
        
        if not image_path.exists():
            logger.error(f"❌ Image file not found: {image_path}")
            logger.info(f"📁 Current working directory: {Path.cwd()}")
            logger.info(f"💡 Please check the path or run from the correct directory")
            return 1
    
    bboxes = image_bboxes[image_path.name]
    
    # Initialize mask generator
    mask_generator = DataPointMaskGenerator(
        color_tolerance=args.color_tolerance,
        min_contour_area=args.min_area,
        max_contour_area=args.max_area
    )
    
    # Process the image
    masks = process_single_image(
        image_path, 
        bboxes, 
        mask_generator, 
        output_dir,
        show_visualizations=not args.no_visualizations
    )
    
    if not masks:
        logger.warning("No valid masks generated")
        return 1
    
    # Convert to COCO format (optimized for small data points)
    converter = COCOMaskConverter(
        data_point_category_id=11,
        simplify_tolerance=0.1,    # Very low tolerance to preserve small data points
        min_polygon_points=3       # Allow triangles (simplified circles)
    )
    
    # Create new COCO dataset with masks
    image_mask_pairs = [(image_path, masks)]
    coco_dataset = converter.create_coco_dataset(image_mask_pairs)
    
    # Save the new dataset
    output_coco_path = output_dir / f"{image_path.stem}_with_masks.json"
    converter.save_coco_dataset(coco_dataset, output_coco_path)
    
    # Also add to existing COCO file
    enhanced_coco_path = output_dir / f"enhanced_{coco_path.name}"
    converter.add_masks_to_existing_coco(coco_path, image_mask_pairs, enhanced_coco_path)
    
    logger.info(f"✅ Mask generation complete!")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎭 Generated {len(masks)} masks")
    logger.info(f"📋 New COCO dataset: {output_coco_path}")
    logger.info(f"📋 Enhanced COCO dataset: {enhanced_coco_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 