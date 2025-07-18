"""
Utility functions for mask generation and validation.

Provides visualization tools and validation functions for generated masks.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def visualize_masks(image: np.ndarray, 
                   masks: List[np.ndarray],
                   bbox_list: Optional[List[List[int]]] = None,
                   title: str = "Generated Masks",
                   save_path: Optional[Union[str, Path]] = None,
                   alpha: float = 0.5) -> None:
    """
    Visualize masks overlaid on the original image.
    
    Args:
        image: Original image (BGR format)
        masks: List of binary masks
        bbox_list: Optional bounding boxes to overlay
        title: Plot title
        save_path: Optional path to save the visualization
        alpha: Transparency for mask overlay
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Create random colors for each mask
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    
    for i, mask in enumerate(masks):
        # Convert mask to RGB
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[mask > 0] = colors[i][:3]
        
        # Overlay mask
        plt.imshow(mask_rgb, alpha=alpha)
        
    # Draw bounding boxes if provided
    if bbox_list:
        for bbox in bbox_list:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
    
    plt.title(f"{title} ({len(masks)} masks)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_mask_comparison(image: np.ndarray,
                            mask_results: Dict[str, List[np.ndarray]],
                            save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Compare masks from different generation methods side by side.
    
    Args:
        image: Original image (BGR format)
        mask_results: Dictionary mapping method names to mask lists
        save_path: Optional path to save the comparison
    """
    num_methods = len(mask_results)
    if num_methods == 0:
        return
        
    fig, axes = plt.subplots(1, num_methods + 1, figsize=(5 * (num_methods + 1), 5))
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Show original image
    if num_methods == 1:
        axes = [axes[0], axes[1]]
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show masks for each method
    for i, (method_name, masks) in enumerate(mask_results.items()):
        ax = axes[i + 1]
        ax.imshow(image_rgb)
        
        # Create random colors for masks
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for j, mask in enumerate(masks):
            mask_rgb = np.zeros((*mask.shape, 3))
            mask_rgb[mask > 0] = colors[j][:3]
            ax.imshow(mask_rgb, alpha=0.6)
            
        ax.set_title(f"{method_name.title()} ({len(masks)} masks)")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved comparison to {save_path}")
    
    plt.show()


def validate_masks(masks: List[np.ndarray],
                  min_area: int = 10,
                  max_area: int = 10000,
                  image_shape: Optional[Tuple[int, int]] = None) -> List[bool]:
    """
    Validate masks based on various criteria.
    
    Args:
        masks: List of binary masks
        min_area: Minimum acceptable area
        max_area: Maximum acceptable area
        image_shape: (height, width) to check if masks are within bounds
        
    Returns:
        List of boolean values indicating which masks are valid
    """
    valid_masks = []
    
    for i, mask in enumerate(masks):
        is_valid = True
        
        # Check area
        area = np.sum(mask > 0)
        if area < min_area or area > max_area:
            logger.debug(f"Mask {i} invalid area: {area}")
            is_valid = False
            
        # Check if mask is within image bounds
        if image_shape:
            h, w = image_shape
            if mask.shape[0] != h or mask.shape[1] != w:
                logger.debug(f"Mask {i} shape mismatch: {mask.shape} vs {image_shape}")
                is_valid = False
                
        # Check if mask has any content
        if not np.any(mask > 0):
            logger.debug(f"Mask {i} is empty")
            is_valid = False
            
        valid_masks.append(is_valid)
        
    valid_count = sum(valid_masks)
    logger.info(f"Validated {valid_count}/{len(masks)} masks")
    
    return valid_masks


def filter_valid_masks(masks: List[np.ndarray],
                      min_area: int = 10,
                      max_area: int = 10000,
                      image_shape: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    Filter masks to only return valid ones.
    
    Args:
        masks: List of binary masks
        min_area: Minimum acceptable area
        max_area: Maximum acceptable area
        image_shape: (height, width) to check if masks are within bounds
        
    Returns:
        List of valid masks
    """
    valid_flags = validate_masks(masks, min_area, max_area, image_shape)
    valid_masks = [mask for mask, is_valid in zip(masks, valid_flags) if is_valid]
    
    return valid_masks


def analyze_mask_distribution(masks: List[np.ndarray]) -> Dict[str, Any]:
    """
    Analyze the distribution of mask properties.
    
    Args:
        masks: List of binary masks
        
    Returns:
        Dictionary with analysis results
    """
    if not masks:
        return {"num_masks": 0}
        
    areas = [np.sum(mask > 0) for mask in masks]
    
    # Calculate bounding box sizes
    bbox_areas = []
    aspect_ratios = []
    
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_areas.append(w * h)
            aspect_ratios.append(w / h if h > 0 else 1.0)
    
    analysis = {
        "num_masks": len(masks),
        "area_stats": {
            "min": min(areas),
            "max": max(areas),
            "mean": np.mean(areas),
            "std": np.std(areas),
            "median": np.median(areas)
        },
        "bbox_area_stats": {
            "min": min(bbox_areas) if bbox_areas else 0,
            "max": max(bbox_areas) if bbox_areas else 0,
            "mean": np.mean(bbox_areas) if bbox_areas else 0,
        },
        "aspect_ratio_stats": {
            "min": min(aspect_ratios) if aspect_ratios else 0,
            "max": max(aspect_ratios) if aspect_ratios else 0,
            "mean": np.mean(aspect_ratios) if aspect_ratios else 0,
        }
    }
    
    return analysis


def save_masks_as_images(masks: List[np.ndarray],
                        output_dir: Union[str, Path],
                        prefix: str = "mask") -> List[Path]:
    """
    Save individual masks as image files.
    
    Args:
        masks: List of binary masks
        output_dir: Directory to save mask images
        prefix: Filename prefix for saved masks
        
    Returns:
        List of paths to saved mask files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, mask in enumerate(masks):
        filename = f"{prefix}_{i:04d}.png"
        file_path = output_dir / filename
        
        cv2.imwrite(str(file_path), mask)
        saved_paths.append(file_path)
        
    logger.info(f"Saved {len(masks)} mask images to {output_dir}")
    
    return saved_paths


def create_mask_summary_image(image: np.ndarray,
                            masks: List[np.ndarray],
                            analysis: Dict[str, Any],
                            save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create a summary visualization with image, masks, and statistics.
    
    Args:
        image: Original image (BGR format)
        masks: List of binary masks
        analysis: Analysis results from analyze_mask_distribution
        save_path: Optional path to save the summary
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Masks overlay
    axes[0, 1].imshow(image_rgb)
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    for i, mask in enumerate(masks):
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[mask > 0] = colors[i][:3]
        axes[0, 1].imshow(mask_rgb, alpha=0.6)
    axes[0, 1].set_title(f"Generated Masks ({len(masks)})")
    axes[0, 1].axis('off')
    
    # Area distribution
    if masks:
        areas = [np.sum(mask > 0) for mask in masks]
        axes[1, 0].hist(areas, bins=min(20, len(masks)), alpha=0.7)
        axes[1, 0].set_title("Mask Area Distribution")
        axes[1, 0].set_xlabel("Area (pixels)")
        axes[1, 0].set_ylabel("Count")
    
    # Statistics text
    stats_text = f"""
Mask Generation Summary
─────────────────────
Number of masks: {analysis['num_masks']}

Area Statistics:
  Mean: {analysis['area_stats']['mean']:.1f} px
  Median: {analysis['area_stats']['median']:.1f} px
  Min: {analysis['area_stats']['min']} px
  Max: {analysis['area_stats']['max']} px
  Std: {analysis['area_stats']['std']:.1f} px

Aspect Ratio:
  Mean: {analysis['aspect_ratio_stats']['mean']:.2f}
  Min: {analysis['aspect_ratio_stats']['min']:.2f}
  Max: {analysis['aspect_ratio_stats']['max']:.2f}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved summary to {save_path}")
    
    plt.show()


def load_and_validate_coco_annotations(coco_path: Union[str, Path]) -> bool:
    """
    Load and validate COCO annotations file.
    
    Args:
        coco_path: Path to COCO JSON file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
            
        # Check required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Validate data point category exists
        data_point_exists = any(cat['name'] == 'data-point' for cat in coco_data['categories'])
        if not data_point_exists:
            logger.warning("No 'data-point' category found in COCO file")
            
        # Count segmentation annotations
        seg_count = sum(1 for ann in coco_data['annotations'] if 'segmentation' in ann)
        
        logger.info(f"COCO validation passed:")
        logger.info(f"  - {len(coco_data['images'])} images")
        logger.info(f"  - {len(coco_data['annotations'])} annotations")
        logger.info(f"  - {seg_count} with segmentation")
        logger.info(f"  - {len(coco_data['categories'])} categories")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating COCO file: {e}")
        return False 