# Scientific Figure Chart Element Detection with Swin Transformer + Mask R-CNN

## Introduction

This project provides a robust pipeline for detecting and segmenting chart elements in scientific figures, with a special focus on small, densely packed data points. It leverages the Swin Transformer (Base) backbone for strong local and global feature extraction, combined with Mask R-CNN for instance segmentation and Cascade R-CNN for multi-stage bounding box refinement.

## Demos

| Demo 1 | Demo 2 |
|--------|--------|
| ![Demo 1](media/demo%201.gif) | ![Demo 2](media/demo%202.gif) |
| *Data bar detection and segmentation* | *Chart element selection* |

| Demo 3 | Demo 4 |
|--------|--------|
| ![Demo 3](media/demo%203.gif) | ![Demo 4](media/demo%204.gif) |
| *Data point detection and user selection workflow* | *Dense captioning and grounding* |

---

## Dataset Preparation

The datasets in `legend_data` are prepared through a multi-step process, as detailed in the `Swin_MaskRCNN_Training.ipynb` notebook and supporting scripts:

1. **Raw Data Collection:**
   - Raw chart images and their corresponding annotation JSON files are collected in `legend_data/train/images/` and `legend_data/train/annotations/`.

2. **Merging and Splitting:**
   - The script merges all valid image/annotation pairs, filters out invalid or missing data, and splits the dataset into training and validation sets (typically 70/30 split).

3. **COCO-Style Conversion & Enrichment:**
   - Each annotation is converted to a COCO-style format, with enhanced metadata for each image (including chart type, plot bounding box, data series, axes info, and visual element statistics).
   - All chart elements (data points, lines, bars, boxplots, etc.) are assigned unique category IDs and bounding boxes.
   - Additional metadata is added for text roles, axis ticks, and other chart-specific features.

4. **Mask Generation:**
   - For data point segmentation, instance masks are generated and added to the annotation files using scripts in `legend_match_swin/mask_generation/`.
   - The mask generation process ensures that each data point is annotated with a pixel-precise mask, which is critical for training Mask R-CNN.

5. **Final Dataset Structure:**
   - The final datasets are saved as enhanced COCO-style JSON files (e.g., `train_enriched_with_info.json`, `val_enriched_with_info.json`) in `legend_data/annotations_JSON_cleaned/`.
   - These files include all necessary metadata, bounding boxes, and masks for robust training and evaluation.

6. **Quality Checks:**
   - The notebook and scripts include setup checkers and validation steps to ensure all images, annotations, and masks are present and correctly formatted before training.

**You can download the final, ready-to-train dataset here:**  
[Final Dataset (Google Drive)](https://drive.google.com/drive/folders/1Lv-LTWG0C9JgoD8DhJkyUuTRVH4lasrq)

---

## Model Architecture

- **Backbone:** Swin Transformer (Base)
  - Hierarchical, window-based self-attention for efficient and effective small object detection.
- **Data Point Detection:** Mask R-CNN
  - Provides both bounding boxes and pixel-precise instance masks for data points.
- **Other Element Detection:** Cascade R-CNN
  - Multi-stage refinement for accurate localization of axes, legends, titles, tick labels, and more.

## Custom Models and Configuration (`legend_match_swin`)

- **Custom Model Registration:**
  - All custom models, heads, datasets, and hooks are registered in `legend_match_swin/custom_models/register.py`.
  - Includes `CustomCascadeWithMeta`, `SquareFCNMaskHead`, `ChartDataset`, and custom hooks for loss, error handling, and chart-type filtering.
- **Custom Dataset:**
  - `ChartDataset` supports enhanced metadata, chart-type-specific filtering, and flexible annotation loading.
  - Chart-type filtering ensures only relevant elements are detected for each chart type (e.g., only data points in scatter plots).
- **Custom Heads:**
  - Specialized heads (e.g., `DataSeriesHead`) for advanced tasks like data series prediction, with attention mechanisms for coordinates and axis-relative positions.

## Mask R-CNN + Swin Transformer for Data Point Detection

- **Configuration:**
  - See `legend_match_swin/mask_rcnn_swin_meta.py` and `mask_generation/mask_rcnn_swin_datapoint.py`.
  - **Backbone:** Swin Transformer Base (pretrained on ImageNet-22K)
  - **Image Size:** All images are resized to **1120×672**. This is mathematically optimized for 14×14 data points and 7×7 Swin windows, ensuring that after 4× downsampling, data points are ~4×4 pixels in feature maps—providing optimal coverage for the Swin attention mechanism.
  - **Mask Head:** 14×14 RoI size, matching data point size.
  - **Class Weighting:** Data-point class receives 10× loss weight to prioritize its detection.
  - **Losses:** Progressive loss strategies, with CIoU loss for better aspect ratio and center alignment.
  - **Augmentation:** Enhanced test-time augmentation and mask-specific optimizations.

- **Why This Setup?**
  - Swin-Base provides a strong balance of capacity and efficiency, and its windowed attention is ideal for small, dense objects.
  - Mask R-CNN enables both detection and instance segmentation, which is crucial for scientific figures where pixel-level accuracy is needed.

## Cascade R-CNN + Swin Transformer for Other Elements

- **Configuration:**
  - See `legend_match_swin/cascade_rcnn_r50_fpn_meta.py`.
  - **Backbone:** Swin Transformer Base
  - **Cascade Structure:** Multi-stage refinement for bounding boxes, especially beneficial for large or ambiguous elements.
  - **Class Weighting:** Data-point class is still prioritized, but cascade structure helps with complex elements.
  - **Preprocessing:** Same 1120×672 resizing for all images, ensuring consistent scale and receptive field.

- **Why This Setup?**
  - Cascade R-CNN’s multi-stage refinement reduces false positives and improves localization for complex elements.
  - Uniform preprocessing ensures anchor sizes and receptive fields are appropriate for all element types.

## Training Workflow

- **Custom Configs:**
  - All training is driven by custom config files in `legend_match_swin/`, which import and register the custom models, datasets, and hooks.
  - Training scripts (e.g., `Swin_MaskRCNN_Training.ipynb`) use these configs to produce the final `chart_datapoint.pth` model.
- **Data:**
  - Enhanced COCO-style annotations with segmentation masks for data points.
  - All images and annotations are preprocessed and filtered according to chart type.
- **Hooks and Losses:**
  - Progressive loss hooks, chart-type distribution hooks, and nan-recovery hooks are used for robust training.

## User Workflow and Dense Captioning

- **Element Selection:**
  - Users can select pre-identified elements (e.g., “legend”, “data points”) during voiceover or interactive sessions, reducing manual work and errors.
- **Dense Captioning:**
  - Accurate detection and segmentation enable precise grounding of spoken references in dense captioning workflows, improving accessibility and automated reporting.

## Performance

| Category        | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|-----------------|-------|--------|--------|-------|-------|-------|
| data-point      | 0.485 | 0.687  | 0.581  | 0.487 | 0.05  | nan   |
| title           | 0.837 | 0.988  | 0.957  | 0.283 | 0.775 | 0.897 |
| x-axis          | 0.382 | 0.860  | 0.261  | 0.382 | nan   | nan   |
| y-axis          | 0.475 | 0.949  | 0.404  | 0.475 | nan   | nan   |
| x-tick-label    | 0.807 | 0.975  | 0.891  | 0.796 | 0.835 | 0.830 |
| y-tick-label    | 0.785 | 0.976  | 0.893  | 0.786 | 0.632 | nan   |
| data-line       | 0.759 | 0.986  | 0.916  | nan   | 0.492 | 0.760 |
| data-bar        | 0.080 | 0.206  | 0.049  | 0.080 | nan   | nan   |
| axis-title      | 0.818 | 0.988  | 0.935  | 0.826 | 0.811 | 0.492 |
| plot-area       | 0.976 | 0.996  | 0.993  | nan   | nan   | 0.976 |

- **High mAP** for title, axis-title, plot-area, tick-labels: Swin-Base + Cascade R-CNN is highly effective for these.
- **Moderate mAP** for x/y-axis, data-line, data-point: These are challenging due to ambiguity or small size.
- **Low mAP** for data-bar: Likely due to limited data or high variability.
- **Many classes** (e.g., subtitle, legend, data-area, grid-line, data-label, legend-text): No detections or not enough ground truth for evaluation.

## How to Train

1. **Prepare Data:**
   - Place images and enhanced COCO-style annotations (with masks) in the expected directories.
2. **Configure Training:**
   - Edit the config files in `legend_match_swin/` as needed for your data paths and training preferences.
3. **Run Training:**
   - Use the provided notebook (`Swin_MaskRCNN_Training.ipynb`) or run:
     ```bash
     python tools/train.py legend_match_swin/mask_rcnn_swin_meta.py --work-dir work_dirs/mask_rcnn_swin
     ```
   - The final model will be saved as `chart_datapoint.pth`.

## References

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [Cascade R-CNN](https://arxiv.org/abs/1712.00726)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

## Acknowledgements

This project builds on the open-source MMDetection framework and the Swin Transformer backbone, with extensive customizations for scientific figure analysis. 
