# Chart Element Mask Generation System

This module provides a comprehensive solution for generating instance segmentation masks for chart elements in scientific charts using semi-automatic OpenCV methods and training Mask R-CNN models.

## Overview

The system consists of three main components:

1. **Semi-automatic mask generation** using OpenCV (color thresholding, circle detection, contour analysis)
2. **COCO format conversion** for compatibility with MMDetection training
3. **Mask R-CNN training configuration** optimized for chart element segmentation

### Supported Elements

- **Data points**: Small geometric shapes (circles, squares, triangles, etc.) using multiple detection methods
- **Data bars**: Simple rectangular masks (no complex shape detection needed)
- **Boxplots**: Box-and-whisker plot elements with whisker detection
- **Other visual elements**: Generic bbox-based masks

**Note**: Data lines are excluded as they're better handled as line annotations rather than segmentation masks.

## Quick Start

### 1. Generate Masks from Chart Images

```bash
# Basic usage - generate masks for a single image
python generate_masks_demo.py \
    --input_image ../science2.jpg \
    --input_coco ../test/annotations.json \
    --output_dir mask_output

# With custom parameters
python generate_masks_demo.py \
    --input_image ../science2.jpg \
    --input_coco ../test/annotations.json \
    --output_dir mask_output \
    --color_tolerance 25 \
    --min_area 5 \
    --max_area 1000 \
    --no_visualizations
```

### 2. Use the API Programmatically

```python
from mask_generation import DataPointMaskGenerator, COCOMaskConverter
from mask_generation.utils import visualize_masks, analyze_mask_distribution

# Initialize mask generator
generator = DataPointMaskGenerator(
    color_tolerance=30,
    min_contour_area=10,
    max_contour_area=2000
)

# Generate masks using multiple methods
mask_results = generator.generate_masks(
    image_path="path/to/chart.jpg",
    bbox_list=[[x1, y1, w1, h1], [x2, y2, w2, h2]],  # Data point bboxes
    methods=['color', 'circles', 'contours']
)

# Convert to COCO format
converter = COCOMaskConverter()
coco_dataset = converter.create_coco_dataset([
    ("path/to/chart.jpg", mask_results['color'])
])
converter.save_coco_dataset(coco_dataset, "output_with_masks.json")
```

## Components

### DataPointMaskGenerator

Semi-automatic mask generation using multiple OpenCV strategies:

- **Color thresholding**: Detects data points based on consistent colors
- **Circle detection**: Uses HoughCircles for circular/round markers
- **Contour detection**: General shape detection with area filtering

**Key Parameters:**
- `color_tolerance`: Color variance tolerance (default: 30)
- `min_contour_area`: Minimum pixel area for valid masks (default: 10) 
- `max_contour_area`: Maximum pixel area for valid masks (default: 2000)
- `circle_min_radius`: Minimum circle radius for HoughCircles (default: 3)
- `circle_max_radius`: Maximum circle radius for HoughCircles (default: 50)

### COCOMaskConverter

Converts OpenCV masks to COCO format with segmentation annotations:

- **Polygon format**: Simplified contour polygons (default)
- **RLE format**: Run-length encoding for complex masks
- **COCO validation**: Ensures proper annotation structure

**Key Parameters:**
- `simplify_tolerance`: Polygon simplification tolerance (default: 1.0)
- `min_polygon_points`: Minimum points for valid polygon (default: 6)
- `use_rle`: Use RLE instead of polygons (default: False)

### Utility Functions

- `visualize_masks()`: Overlay masks on original images
- `visualize_mask_comparison()`: Compare different generation methods
- `analyze_mask_distribution()`: Statistics on mask properties
- `filter_valid_masks()`: Remove invalid/corrupted masks
- `create_mask_summary_image()`: Comprehensive visualization with stats

## Mask R-CNN Training

### Configuration

The provided `mask_rcnn_swin_datapoint.py` config is optimized for data point segmentation:

- **Backbone**: Swin Transformer Base (pretrained)
- **Single class**: Only "data-point" category
- **Small object optimization**: Adjusted anchor scales and NMS thresholds
- **Mask-specific losses**: Optimized for tiny object segmentation

### Training Setup

1. **Prepare your dataset** with generated masks:
```bash
data/
├── images/
│   ├── chart1.jpg
│   ├── chart2.jpg
│   └── ...
└── annotations/
    ├── train_with_masks.json
    └── val_with_masks.json
```

2. **Update the config** paths in `mask_rcnn_swin_datapoint.py`:
```python
data_root = 'path/to/your/data/'
ann_file = 'annotations/train_with_masks.json'
```

3. **Train the model**:
```bash
cd ../..  # Go to mmdetection root
python tools/train.py \
    legend_match_swin/mask_generation/mask_rcnn_swin_datapoint.py \
    --work-dir work_dirs/mask_rcnn_datapoint
```

### Expected Performance

For data point instance segmentation:
- **Box mAP**: 0.6-0.8 (depending on data quality)
- **Mask mAP**: 0.5-0.7 (typically 10-15% lower than box mAP)
- **Training time**: ~20-30 epochs for convergence
- **Inference speed**: ~50-100ms per image (depending on data point density)

## Advanced Usage

### Batch Processing Multiple Images

```python
from pathlib import Path
import json

# Load COCO annotations
with open('annotations.json', 'r') as f:
    coco_data = json.load(f)

# Extract all images with data points
image_bboxes = extract_data_point_bboxes(Path('annotations.json'))

# Process all images
generator = DataPointMaskGenerator()
converter = COCOMaskConverter()

image_mask_pairs = []
for filename, bboxes in image_bboxes.items():
    image_path = Path('images') / filename
    if image_path.exists():
        mask_results = generator.generate_masks(image_path, bboxes)
        
        # Use the best method (you can customize this logic)
        best_masks = mask_results.get('color', [])
        if not best_masks:
            best_masks = mask_results.get('circles', [])
        
        image_mask_pairs.append((image_path, best_masks))

# Create enhanced dataset
coco_with_masks = converter.create_coco_dataset(image_mask_pairs)
converter.save_coco_dataset(coco_with_masks, 'dataset_with_masks.json')
```

### Custom Mask Generation Strategy

```python
class CustomDataPointGenerator(DataPointMaskGenerator):
    def custom_method(self, image, bbox_list):
        """Implement your own mask generation logic."""
        masks = []
        
        # Your custom algorithm here
        # For example, template matching, ML-based segmentation, etc.
        
        return masks
    
    def generate_masks(self, image_path, bbox_list=None, methods=None):
        results = super().generate_masks(image_path, bbox_list, methods)
        
        # Add your custom method
        if bbox_list:
            image = cv2.imread(str(image_path))
            results['custom'] = self.custom_method(image, bbox_list)
            
        return results
```

### Integration with Existing Pipeline

To integrate mask generation with your existing training pipeline:

```python
# After running your Cascade R-CNN to get data point detections
cascade_results = run_cascade_inference(image_path)
data_point_bboxes = extract_data_point_boxes(cascade_results)

# Generate masks for these detections
mask_generator = DataPointMaskGenerator()
masks = mask_generator.generate_masks(image_path, data_point_bboxes)

# Run mask model for final segmentation
mask_model = init_mask_model('work_dirs/mask_rcnn_datapoint/latest.pth')
final_results = run_mask_inference(mask_model, image_path, data_point_bboxes)

# Combine bbox + mask results
combined_results = merge_bbox_and_mask_results(cascade_results, final_results)
```

## Dependencies

Required packages (add to your `requirements.txt`):

```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pycocotools>=2.0.2
mmdet>=3.0.0
mmcv>=2.0.0
```

## Troubleshooting

### Common Issues

1. **No masks generated**:
   - Check if data point bboxes are provided correctly
   - Adjust `color_tolerance` and area thresholds
   - Try different detection methods

2. **Poor mask quality**:
   - Increase `simplify_tolerance` for smoother polygons
   - Filter masks with `min_overlap` parameter
   - Use RLE format for complex shapes

3. **Training convergence issues**:
   - Ensure sufficient mask annotations (>100 per class minimum)
   - Check mask quality with visualization tools
   - Adjust learning rate and training schedule

4. **Memory issues during training**:
   - Reduce batch size in config
   - Use gradient checkpointing
   - Consider using smaller input resolution

### Performance Tips

- **For speed**: Use only 'circles' method for simple round markers
- **For accuracy**: Combine all three methods and filter by bbox overlap
- **For memory**: Process images in batches rather than all at once
- **For quality**: Manually review and filter generated masks before training

## Contributing

To extend this system:

1. Add new mask generation methods to `DataPointMaskGenerator`
2. Implement custom visualization functions in `utils.py`
3. Create specialized COCO converters for your use case
4. Optimize the Mask R-CNN config for your specific chart types

## License

This module is part of the larger chart analysis system and follows the same license terms. 