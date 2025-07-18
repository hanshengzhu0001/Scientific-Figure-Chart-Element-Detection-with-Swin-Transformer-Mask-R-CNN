#!/usr/bin/env python3
"""
Clear mask visualization with multiple display options.

Provides various ways to visualize masks clearly:
- High contrast masks
- Side-by-side comparisons
- Individual mask display
- Mask outlines only
- Different color schemes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json

def visualize_masks_clear(image: np.ndarray, 
                         masks: List[np.ndarray],
                         bbox_list: Optional[List[List[int]]] = None,
                         save_path: Optional[Union[str, Path]] = None,
                         show_options: List[str] = ['overlay', 'masks_only', 'outlines']) -> None:
    """
    Create multiple clear visualizations of masks.
    
    Args:
        image: Original image (BGR format)
        masks: List of binary masks
        bbox_list: Optional bounding boxes
        save_path: Optional path to save (will create multiple files)
        show_options: List of visualization types to show
    """
    if not masks:
        print("No masks to visualize!")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create high-contrast colors for masks
    colors = [
        [1.0, 0.0, 0.0],  # Bright Red
        [0.0, 1.0, 0.0],  # Bright Green  
        [0.0, 0.0, 1.0],  # Bright Blue
        [1.0, 1.0, 0.0],  # Bright Yellow
        [1.0, 0.0, 1.0],  # Bright Magenta
        [0.0, 1.0, 1.0],  # Bright Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
    ]
    
    # Repeat colors if we have more masks
    while len(colors) < len(masks):
        colors.extend(colors)
    
    num_plots = len(show_options)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Option 1: Overlay with reduced transparency
    if 'overlay' in show_options:
        ax = axes[plot_idx]
        ax.imshow(image_rgb)
        
        for i, mask in enumerate(masks):
            mask_colored = np.zeros((*mask.shape, 3))
            mask_colored[mask > 0] = colors[i % len(colors)]
            ax.imshow(mask_colored, alpha=0.7)  # Less transparent
        
        # Draw bounding boxes
        if bbox_list:
            for bbox in bbox_list:
                x, y, w, h = bbox
                rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='white', facecolor='none', linestyle='--')
                ax.add_patch(rect)
        
        ax.set_title(f"Mask Overlay ({len(masks)} masks)")
        ax.axis('off')
        plot_idx += 1
    
    # Option 2: Masks only (no background image)
    if 'masks_only' in show_options:
        ax = axes[plot_idx]
        
        # Create black background
        mask_display = np.zeros_like(image_rgb)
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_display[mask > 0] = [int(c * 255) for c in color]
        
        ax.imshow(mask_display)
        ax.set_title(f"Masks Only ({len(masks)} masks)")
        ax.axis('off')
        plot_idx += 1
    
    # Option 3: Mask outlines only
    if 'outlines' in show_options:
        ax = axes[plot_idx]
        ax.imshow(image_rgb)
        
        for i, mask in enumerate(masks):
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Convert contour to matplotlib format
                contour = contour.reshape(-1, 2)
                color = colors[i % len(colors)]
                ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=3)
        
        ax.set_title(f"Mask Outlines ({len(masks)} masks)")
        ax.axis('off')
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        base_path = Path(save_path)
        save_file = base_path.parent / f"{base_path.stem}_clear_masks{base_path.suffix}"
        plt.savefig(save_file, bbox_inches='tight', dpi=200)
        print(f"💾 Saved clear mask visualization: {save_file}")
    
    plt.show()


def visualize_individual_masks(image: np.ndarray, 
                              masks: List[np.ndarray],
                              save_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Show each mask individually for detailed inspection.
    
    Args:
        image: Original image (BGR format)  
        masks: List of binary masks
        save_dir: Optional directory to save individual mask images
    """
    if not masks:
        print("No masks to visualize!")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate grid size
    n_masks = len(masks)
    cols = min(4, n_masks)
    rows = (n_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_masks))
    
    for i, mask in enumerate(masks):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Show original image
        ax.imshow(image_rgb, alpha=0.3)  # Very faded background
        
        # Show mask in bright color
        mask_colored = np.zeros((*mask.shape, 3))
        mask_colored[mask > 0] = colors[i][:3]
        ax.imshow(mask_colored, alpha=0.9)  # Very opaque mask
        
        # Calculate mask statistics
        area = np.sum(mask > 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = cv2.boundingRect(contours[0]) if contours else (0, 0, 0, 0)
        
        ax.set_title(f"Mask {i+1}\nArea: {area}px\nBBox: {bbox[2]}×{bbox[3]}")
        ax.axis('off')
        
        # Save individual mask if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save just the mask
            mask_file = save_dir / f"mask_{i+1:03d}.png"
            cv2.imwrite(str(mask_file), mask * 255)
            
            # Save mask with color overlay
            overlay_file = save_dir / f"mask_{i+1:03d}_overlay.png"
            overlay = image_rgb.copy()
            overlay[mask > 0] = [int(c * 255) for c in colors[i][:3]]
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_file), overlay_rgb)
    
    # Hide unused subplots
    for i in range(n_masks, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        summary_file = save_dir / "individual_masks_summary.png"
        plt.savefig(summary_file, bbox_inches='tight', dpi=150)
        print(f"💾 Saved individual masks: {save_dir}")
        print(f"💾 Summary image: {summary_file}")
    
    plt.show()


def create_mask_comparison_grid(image: np.ndarray,
                               mask_results: dict,
                               save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create a clear comparison grid showing different mask generation methods.
    
    Args:
        image: Original image (BGR format)
        mask_results: Dictionary mapping method names to mask lists  
        save_path: Optional path to save the comparison
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    methods = list(mask_results.keys())
    num_methods = len(methods)
    
    if num_methods == 0:
        print("No mask results to compare!")
        return
    
    # Create 2 rows: top row shows overlays, bottom row shows masks only
    fig, axes = plt.subplots(2, num_methods + 1, figsize=(5 * (num_methods + 1), 10))
    
    # Original image (both rows)
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image_rgb)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')
    
    # Show each method
    for i, (method_name, masks) in enumerate(mask_results.items()):
        col = i + 1
        
        # Create high-contrast colors
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(masks))))
        
        # Top row: Overlay
        axes[0, col].imshow(image_rgb)
        for j, mask in enumerate(masks):
            mask_colored = np.zeros((*mask.shape, 3))
            mask_colored[mask > 0] = colors[j][:3]
            axes[0, col].imshow(mask_colored, alpha=0.8)
        axes[0, col].set_title(f"{method_name.title()}\n({len(masks)} masks)")
        axes[0, col].axis('off')
        
        # Bottom row: Masks only (black background)
        mask_display = np.zeros_like(image_rgb)
        for j, mask in enumerate(masks):
            color = colors[j][:3]
            mask_display[mask > 0] = [int(c * 255) for c in color]
        
        axes[1, col].imshow(mask_display)
        axes[1, col].set_title(f"{method_name.title()}\n(Masks Only)")
        axes[1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"💾 Saved comparison grid: {save_path}")
    
    plt.show()


def generate_clear_mask_demo(image_path: str, 
                           bbox_list: List[List[int]] = None,
                           output_dir: str = "clear_mask_output") -> None:
    """
    Generate clear mask visualizations using all available methods.
    
    Args:
        image_path: Path to input image
        bbox_list: Optional list of bounding boxes  
        output_dir: Output directory for visualizations
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from mask_generation import DataPointMaskGenerator
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"🖼️ Processing image: {Path(image_path).name}")
    print(f"📐 Image size: {image.shape}")
    
    # Initialize mask generator
    generator = DataPointMaskGenerator(
        color_tolerance=30,
        min_contour_area=5,
        max_contour_area=2000
    )
    
    # Generate masks using all methods
    print("🎭 Generating masks...")
    mask_results = generator.generate_masks(
        image_path,
        bbox_list=bbox_list,
        methods=['color', 'circles', 'contours', 'shapes']
    )
    
    # Combine all masks
    all_masks = []
    for method_masks in mask_results.values():
        all_masks.extend(method_masks)
    
    if not all_masks:
        print("❌ No masks generated!")
        return
    
    print(f"✅ Generated {len(all_masks)} total masks")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print("🎨 Creating visualizations...")
    
    # 1. Clear overlay visualization
    visualize_masks_clear(
        image, all_masks, bbox_list,
        save_path=output_path / "clear_masks.png",
        show_options=['overlay', 'masks_only', 'outlines']
    )
    
    # 2. Individual mask inspection
    visualize_individual_masks(
        image, all_masks[:12],  # Limit to first 12 for clarity
        save_dir=output_path / "individual_masks"
    )
    
    # 3. Method comparison
    create_mask_comparison_grid(
        image, mask_results,
        save_path=output_path / "method_comparison.png"
    )
    
    print(f"🎉 All visualizations saved to: {output_path}")
    print(f"📁 Check these files for clear mask views:")
    print(f"   - clear_masks.png (overlay + masks only + outlines)")
    print(f"   - individual_masks/ (each mask separately)")
    print(f"   - method_comparison.png (comparison of all methods)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate clear mask visualizations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="clear_mask_output", help="Output directory")
    
    args = parser.parse_args()
    
    generate_clear_mask_demo(args.image, output_dir=args.output) 