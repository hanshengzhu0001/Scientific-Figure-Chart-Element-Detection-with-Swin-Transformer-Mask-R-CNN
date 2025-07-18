import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project paths for imports
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
except NameError:
    project_root = '/content/CHART-DeMatch'
    if not os.path.exists(project_root):
        project_root = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch'
    if not os.path.exists(project_root):
        project_root = os.getcwd()
    sys.path.insert(0, project_root)

sys.path.insert(0, '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch')
sys.path.insert(0, '/content/CHART-DeMatch')

print("🚀 DATA POINT PRECISION ANALYSIS (SCATTER & LINE CHARTS)")
print("=" * 60)
print("🎯 Focus: Scatter and line charts only")
print("📋 Conditions: chart-type in ['scatter', 'line']")

# Import MMDetection modules
try:
    # Add mmdetection to path if it exists
    mmdetection_paths = [
        '/content/mmdetection',
        './mmdetection',
        '../mmdetection',
        os.path.join(os.getcwd(), 'mmdetection')
    ]
    
    for path in mmdetection_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✅ Added mmdetection path: {path}")
            break
    
    from mmdet.utils import register_all_modules
    from mmengine.registry import MODELS
    from mmengine.config import Config
    from mmdet.apis import inference_detector, init_detector
    from mmengine.model.utils import revert_sync_batchnorm
    register_all_modules()
    try:
        from legend_match_swin.custom_models.register import register_all_modules as register_custom_modules
        register_custom_modules()
        print("✅ Custom models from legend_match_swin registered successfully")
    except ImportError as e:
        print(f"⚠️  Could not import custom models: {e}")
    MMDET_AVAILABLE = True
    print("✅ MMDetection available")
except ImportError as e:
    print(f"❌ MMDetection not available: {e}")
    print("💡 Please ensure mmdetection is installed or in your Python path")
    MMDET_AVAILABLE = False
    # Don't exit in notebook environment
    if 'ipykernel' not in sys.modules and 'google.colab' not in sys.modules:
        sys.exit(1)

ENHANCED_CLASS_NAMES = [
    'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
    'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item',
    'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
    'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
]
DATA_POINT_CLASS_ID = ENHANCED_CLASS_NAMES.index('data-point')
print(f"📊 Data-point class ID: {DATA_POINT_CLASS_ID} ('{ENHANCED_CLASS_NAMES[DATA_POINT_CLASS_ID]}')")

DISPLAY_CONFIG = {
    'grid_columns': 4,
    'max_images_to_process': None,
    'iou_threshold': 0.5,
    'score_threshold': 0.1,
    'show_detailed_view': True,
    'save_files': True,
}

# Path configuration for Colab

def find_project_files():
    potential_roots = [os.getcwd(), project_root if 'project_root' in globals() else None]
    potential_roots = [p for p in potential_roots if p is not None]
    found_root = None
    found_checkpoint = None
    
    # First, let's scan for the file in common locations
    search_locations = [
        '/content',
        '/content/drive/MyDrive',
        '/content/drive/.shortcut-targets-by-id',
        os.getcwd(),
        '.',
        '..'
    ]
    
    print("🔍 Scanning for chart_datapoint.pth...")
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        try:
            # Walk through the directory tree (limited depth)
            for root, dirs, files in os.walk(location):
                for file in files:
                    if file == 'chart_datapoint.pth':
                        full_path = os.path.join(root, file)
                        found_root = root
                        found_checkpoint = full_path
                        print(f"✅ Found chart_datapoint.pth at: {full_path}")
                        break
                
                # Limit depth to avoid going too deep
                if root.count(os.sep) - location.count(os.sep) > 3:
                    dirs[:] = []  # Don't go deeper
                    
                if found_checkpoint:
                    break
                    
        except PermissionError:
            continue
        except Exception as e:
            print(f"⚠️  Error scanning {location}: {e}")
            
        if found_checkpoint:
            break
    
    # If not found, also check the original simple locations
    if not found_checkpoint:
        for root in potential_roots:
            checkpoint_path = os.path.join(root, 'chart_datapoint.pth')
            if os.path.exists(checkpoint_path):
                found_root = root
                found_checkpoint = checkpoint_path
                print(f"✅ Found chart_datapoint.pth at: {checkpoint_path}")
                break
    
    if not found_checkpoint:
        print("❌ Could not find chart_datapoint.pth in any common location.")
        print("\n💡 Please ensure chart_datapoint.pth is uploaded to Colab.")
        print("💡 Common upload locations:")
        print("   - /content/ (root of Colab)")
        print("   - /content/drive/MyDrive/ (Google Drive)")
        print("   - Current working directory")
        
        # Show what .pth files ARE available
        print("\n🔍 Available .pth files found:")
        for location in search_locations[:4]:  # Check main locations only
            if not os.path.exists(location):
                continue
            try:
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if file.endswith('.pth'):
                            full_path = os.path.join(root, file)
                            print(f"   📄 {full_path}")
                    if root.count(os.sep) - location.count(os.sep) > 2:
                        dirs[:] = []
            except:
                continue
        return None, None, None, None, None
    
    # If checkpoint is in /content but we're working in Google Drive, use the working directory
    if found_root == '/content' and os.getcwd() != '/content':
        print(f"📁 Checkpoint found in /content, but working in: {os.getcwd()}")
        print(f"📁 Using working directory as root for validation data")
        found_root = os.getcwd()
    
    print(f"\n📁 Using found_root: {found_root}")
    val_ann_candidates = [
        os.path.join(found_root, 'legend_data/annotations_JSON_cleaned/val_enriched_with_info.json'),
        os.path.join(found_root, 'val_enriched_with_info.json')
    ]
    val_img_candidates = [
        os.path.join(found_root, 'legend_data/train/images/'),
        os.path.join(found_root, 'images/')
    ]
    found_val_ann = None
    found_val_img = None
    print(f"\n🔍 Searching for validation annotations...")
    for path in val_ann_candidates:
        if os.path.exists(path):
            found_val_ann = path
            print(f"✅ Found validation annotations at: {path}")
            break
        else:
            print(f"   ❌ Not found: {path}")
    
    print(f"\n🔍 Searching for validation images...")
    for path in val_img_candidates:
        if os.path.exists(path):
            found_val_img = path
            print(f"✅ Found validation images at: {path}")
            break
        else:
            print(f"   ❌ Not found: {path}")
    output_dir = os.path.join(found_root, 'scatter_plot_data_point_analysis')
    return found_root, found_checkpoint, found_val_ann, found_val_img, output_dir

PROJECT_ROOT, CHECKPOINT_PATH, VAL_ANN_PATH, VAL_IMG_DIR, OUTPUT_DIR = find_project_files()

# Model config for chart_datapoint.pth

def create_swin_cascade_config():
    return dict(
        type='MaskRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            pad_mask=True,
            mask_pad_value=0,
        ),
        backbone=dict(
            type='SwinTransformer',
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth'
            )
        ),
        neck=dict(
            type='FPN',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=5,
            start_level=0,
            add_extra_convs='on_input'
        ),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[1, 2, 4, 8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=21,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0
                ),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=(14, 14), sampling_ratio=0, aligned=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=21,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0
                )
            )
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.1,
                nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.1, method='gaussian', sigma=0.5),
                max_per_img=500))
    )

def safe_torch_load(checkpoint_path: str):
    """Safely load checkpoint with fallback handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        return checkpoint
    except Exception as e1:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e2:
            print(f"❌ Failed to load checkpoint: {e1}, {e2}")
            return None

def load_model():
    """Load the chart_datapoint.pth model"""

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: {CHECKPOINT_PATH} not found!")
        print(f"💡 Make sure chart_datapoint.pth is in the project root directory")
        return None, None

    print(f"📦 Loading model from: {CHECKPOINT_PATH}")

    try:
        # Use the actual mask_rcnn_swin_meta.py config file
        config_path = "legend_match_swin/mask_rcnn_swin_meta.py"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            print("💡 Make sure mask_rcnn_swin_meta.py is in the legend_match_swin folder")
            return None, None

        print(f"📄 Using config file: {config_path}")

        # Initialize model
        print("🔧 Building model...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🎮 Using device: {device}")

        model = init_detector(config_path, CHECKPOINT_PATH, device=device)

        # Convert SyncBatchNorm to BatchNorm for inference
        model = revert_sync_batchnorm(model)
        model.eval()

        print("✅ Model loaded successfully!")
        
        # Load config separately for reference
        config = Config.fromfile(config_path)
        return model, config

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def load_validation_data():
    """Load validation dataset annotations and filter for scatter and line charts with data points"""

    # Check if paths are None first
    if VAL_ANN_PATH is None:
        print(f"❌ Validation annotations path is None - no validation data found")
        return None, None

    if VAL_IMG_DIR is None:
        print(f"❌ Validation images directory path is None - no validation data found")
        return None, None

    if not os.path.exists(VAL_ANN_PATH):
        print(f"❌ Validation annotations not found: {VAL_ANN_PATH}")
        return None, None

    if not os.path.exists(VAL_IMG_DIR):
        print(f"❌ Validation images directory not found: {VAL_IMG_DIR}")
        return None, None

    print(f"📊 Loading validation data from: {VAL_ANN_PATH}")

    try:
        with open(VAL_ANN_PATH, 'r') as f:
            val_data = json.load(f)

        print(f"✅ Loaded {len(val_data['images'])} validation images")
        print(f"✅ Loaded {len(val_data['annotations'])} validation annotations")

        # Filter for scatter and line charts
        print(f"\n🎯 FILTERING FOR SCATTER AND LINE CHARTS")
        print(f"📋 Conditions: chart-type in ['scatter', 'line']")
        
        # Create image_id to annotations mapping
        img_to_anns = defaultdict(list)
        data_point_annotations = []
        for ann in val_data['annotations']:
            if ann['category_id'] == 11:  # data-point category ID
                img_to_anns[ann['image_id']].append(ann)
                data_point_annotations.append(ann)
        
        print(f"📊 Found {len(data_point_annotations)} data-point annotations across all images")
        
        # Debug: Check category IDs in annotations
        category_ids = set()
        for ann in val_data['annotations']:
            category_ids.add(ann['category_id'])
        print(f"📊 Category IDs found in annotations: {sorted(category_ids)}")
        print(f"📊 Looking for data-point category ID: 11")
        
        # Debug: Check which chart types have data-point annotations
        chart_type_with_data_points = defaultdict(int)
        for img in val_data['images']:
            img_id = img['id']
            if len(img_to_anns[img_id]) > 0:
                chart_type = img.get('chart_type', 'unknown')
                chart_type_with_data_points[chart_type] += 1
        
        print(f"📊 Images with data-point annotations by chart type:")
        for chart_type, count in sorted(chart_type_with_data_points.items()):
            print(f"      {chart_type}: {count} images")
        
        # Filter images that are scatter plots (no data-point requirement)
        scatter_line_images = []
        
        # Debug: Check a few sample images to understand the structure
        print(f"\n🔍 DEBUGGING: Checking sample images for chart type structure...")
        sample_count = 0
        for img in val_data['images'][:5]:  # Check first 5 images
            sample_count += 1
            print(f"   Sample {sample_count}: {img['file_name']}")
            print(f"      Keys: {list(img.keys())}")
            if 'chart_type' in img:
                print(f"      chart_type: {img['chart_type']}")
            elif 'chart-type' in img:
                print(f"      chart-type: {img['chart-type']}")
            elif 'metadata' in img:
                print(f"      metadata keys: {list(img['metadata'].keys())}")
                if 'chart-type' in img['metadata']:
                    print(f"      metadata.chart-type: {img['metadata']['chart-type']}")
            else:
                print(f"      No chart_type, chart-type, or metadata found")
        
        for img in val_data['images']:
            img_id = img['id']
            
            # Check if image is a scatter plot
            is_scatter = False
            if 'chart_type' in img:
                is_scatter = img['chart_type'] == 'scatter'
            elif 'chart-type' in img:
                is_scatter = img['chart-type'] == 'scatter'
            elif 'metadata' in img and 'chart-type' in img['metadata']:
                is_scatter = img['metadata']['chart-type'] == 'scatter'
            
            if is_scatter:
                scatter_line_images.append(img)
        
        print(f"📊 Found {len(scatter_line_images)} scatter/line charts")
        
        # Debug: Count all chart types in the dataset
        print(f"\n📊 DEBUGGING: Chart type distribution in dataset...")
        chart_type_counts = {}
        for img in val_data['images']:
            chart_type = None
            if 'chart_type' in img:
                chart_type = img['chart_type']
            elif 'chart-type' in img:
                chart_type = img['chart-type']
            elif 'metadata' in img and 'chart-type' in img['metadata']:
                chart_type = img['metadata']['chart-type']
            
            if chart_type:
                chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1
        
        print(f"   Chart type distribution:")
        for chart_type, count in sorted(chart_type_counts.items()):
            print(f"      {chart_type}: {count} images")
        
        # Create filtered dataset
        # Only keep annotations for the filtered images
        filtered_image_ids = {img['id'] for img in scatter_line_images}
        filtered_annotations = [ann for ann in val_data['annotations'] if ann['image_id'] in filtered_image_ids]
        
        filtered_data = {
            'images': scatter_line_images,
            'annotations': filtered_annotations,  # Only annotations for filtered images
            'categories': val_data['categories']
        }
        
        print(f"📊 Filtered dataset: {len(scatter_line_images)} images, {len(filtered_annotations)} annotations")
        
        # Show sample of filtered images
        if len(scatter_line_images) > 0:
            print(f"📋 Sample scatter/line charts:")
            for i, img in enumerate(scatter_line_images[:5]):
                chart_type = img.get('chart_type', img.get('chart-type', img.get('metadata', {}).get('chart-type', 'unknown')))
                data_point_count = len(img_to_anns[img['id']])
                print(f"   {i+1}. {img['file_name']} (chart-type: {chart_type}, data-points: {data_point_count})")
            if len(scatter_line_images) > 5:
                print(f"   ... and {len(scatter_line_images) - 5} more")
        
        # Debug: Check what annotations scatter plots actually have
        print(f"\n🔍 DEBUGGING: Annotation analysis for scatter/line charts...")
        scatter_annotation_counts = defaultdict(int)
        scatter_annotation_types = defaultdict(int)
        
        for img in scatter_line_images:
            img_id = img['id']
            img_annotations = [ann for ann in val_data['annotations'] if ann['image_id'] == img_id]
            
            # Count total annotations per scatter plot
            scatter_annotation_counts[len(img_annotations)] += 1
            
            # Count annotation types
            for ann in img_annotations:
                category_id = ann['category_id']
                scatter_annotation_types[category_id] += 1
        
        print(f"📊 Scatter plot annotation distribution:")
        print(f"   Scatter plots by annotation count:")
        for count, num_plots in sorted(scatter_annotation_counts.items()):
            print(f"      {count} annotations: {num_plots} scatter plots")
        
        print(f"   Annotation types in scatter plots:")
        for category_id, count in sorted(scatter_annotation_types.items()):
            print(f"      Category ID {category_id}: {count} annotations")
        
        # Show sample annotations for first few scatter plots
        print(f"\n📋 Sample annotations for first 3 scatter plots:")
        for i, img in enumerate(scatter_line_images[:3]):
            img_id = img['id']
            img_annotations = [ann for ann in val_data['annotations'] if ann['image_id'] == img_id]
            print(f"   {img['file_name']}: {len(img_annotations)} annotations")
            for j, ann in enumerate(img_annotations[:5]):  # Show first 5 annotations
                print(f"      {j+1}. Category ID {ann['category_id']}: {ann.get('bbox', 'no bbox')}")
            if len(img_annotations) > 5:
                print(f"      ... and {len(img_annotations) - 5} more annotations")
        
        return filtered_data, VAL_IMG_DIR
    except Exception as e:
        print(f"❌ Error loading validation data: {str(e)}")
        return None, None

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def evaluate_data_point_precision(model, val_data, val_img_dir, iou_threshold=0.5, score_threshold=0.1, max_images=None):
    """
    Evaluate data-point detection precision for each image using point-in-box matching.

    For data-point detection, precision is calculated as:
    - Ground truth: actual point coordinates (center of GT bounding boxes)
    - Predictions: bounding boxes from the model
    - True Positive: predicted box contains at least one ground truth point
    - Precision = TP / (TP + FP) = correct_predictions / total_predictions

    This is more appropriate than IoU-based matching for point detection tasks.
    """

    print(f"\n🎯 Evaluating data-point precision using POINT-IN-BOX matching (Score≥{score_threshold})")
    print(f"💡 Precision = predicted_boxes_containing_GT_points / total_predicted_boxes")

    # Create image_id to annotations mapping
    img_to_anns = defaultdict(list)
    for ann in val_data['annotations']:
        if ann['category_id'] == 11:  # data-point category ID
            img_to_anns[ann['image_id']].append(ann)

    # Create image_id to image info mapping (use filtered images)
    img_id_to_info = {img['id']: img for img in val_data['images']}

    results = []

    # Process all scatter plots (already filtered in load_validation_data)
    all_scatter_image_ids = [img['id'] for img in val_data['images']]
    print(f"📊 Found {len(all_scatter_image_ids)} scatter/line charts in filtered dataset")
    
    # Check how many have data-point annotations
    scatter_with_data_points = [img_id for img_id in all_scatter_image_ids if len(img_to_anns[img_id]) > 0]
    print(f"📊 Among these, {len(scatter_with_data_points)} have data-point annotations")

    # Ensure we only process image IDs that exist in both annotations and filtered images
    available_image_ids = set(img_id_to_info.keys())
    all_scatter_image_ids = [img_id for img_id in all_scatter_image_ids if img_id in available_image_ids]
    print(f"📊 After filtering for available images: {len(all_scatter_image_ids)} scatter/line charts")

    # Limit number of images for faster processing (if max_images is specified)
    if max_images is None:
        images_to_process = all_scatter_image_ids
        print(f"🎯 Processing ALL {len(images_to_process)} scatter/line charts")
    else:
        images_to_process = all_scatter_image_ids[:max_images]
        print(f"🎯 Processing {len(images_to_process)} scatter/line charts (limited to {max_images} for speed)")

    for img_id in tqdm(images_to_process, desc="Evaluating images"):

        img_info = img_id_to_info[img_id]
        img_path = os.path.join(val_img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue

        try:
            # Run inference
            result = inference_detector(model, img_path)

            # Get predictions for data-point class
            if hasattr(result, 'pred_instances'):
                pred_boxes = result.pred_instances.bboxes.cpu().numpy()
                pred_scores = result.pred_instances.scores.cpu().numpy()
                pred_labels = result.pred_instances.labels.cpu().numpy()
            else:
                # Handle older format
                pred_boxes = result[DATA_POINT_CLASS_ID][:, :4]
                pred_scores = result[DATA_POINT_CLASS_ID][:, 4]
                pred_labels = np.full(len(pred_boxes), DATA_POINT_CLASS_ID)

            # Filter predictions for data-point class and score threshold
            data_point_mask = (pred_labels == 11) & (pred_scores >= score_threshold)  # data-point category ID
            pred_boxes_filtered = pred_boxes[data_point_mask]
            pred_scores_filtered = pred_scores[data_point_mask]

            # Get ground truth data-point coordinates (center of bbox)
            gt_points = []
            gt_boxes = []  # Keep for visualization
            for ann in img_to_anns[img_id]:
                x, y, w, h = ann['bbox']
                # Data point is the center of the bounding box
                center_x = x + w / 2
                center_y = y + h / 2
                gt_points.append([center_x, center_y])
                gt_boxes.append([x, y, x + w, y + h])  # Keep for visualization

            # Calculate precision: how many predicted boxes contain ground truth points
            if len(pred_boxes_filtered) == 0:
                precision = 1.0 if len(gt_points) == 0 else 0.0  # Perfect if no GT and no predictions
                tp = 0
                fp = len(pred_boxes_filtered)
            else:
                tp = 0  # True positives: predicted boxes that contain at least one GT point
                matched_gt_points = set()

                for pred_idx, pred_box in enumerate(pred_boxes_filtered):
                    x1, y1, x2, y2 = pred_box
                    contains_point = False

                    # Check if this predicted box contains any unmatched ground truth points
                    for gt_idx, (gt_x, gt_y) in enumerate(gt_points):
                        if gt_idx in matched_gt_points:
                            continue

                        # Check if point is inside the predicted box
                        if x1 <= gt_x <= x2 and y1 <= gt_y <= y2:
                            contains_point = True
                            matched_gt_points.add(gt_idx)
                            break  # One point per predicted box is enough for TP

                    if contains_point:
                        tp += 1

                fp = len(pred_boxes_filtered) - tp
                precision = tp / len(pred_boxes_filtered) if len(pred_boxes_filtered) > 0 else 0.0

            results.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'img_path': img_path,
                'precision': precision,
                'tp': tp,
                'fp': fp,
                'num_predictions': len(pred_boxes_filtered),
                'num_gt': len(gt_points),
                'pred_boxes': pred_boxes_filtered.tolist(),
                'pred_scores': pred_scores_filtered.tolist(),
                'gt_boxes': gt_boxes,  # For visualization
                'gt_points': gt_points  # Actual data point coordinates
            })

        except Exception as e:
            print(f"⚠️  Error processing {img_info['file_name']}: {str(e)}")
            continue

    print(f"✅ Evaluated {len(results)} images")
    return results

def visualize_image_results(img_path, pred_boxes, pred_scores, gt_boxes, precision, title_suffix="", gt_points=None):
    """Visualize predictions vs ground truth for an image"""

    # Load image
    img = plt.imread(img_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    # Draw ground truth boxes (green, semi-transparent)
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        ax.add_patch(rect)

    # Draw ground truth points (green dots) - the actual points being evaluated
    if gt_points:
        for gt_x, gt_y in gt_points:
            ax.plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
                   markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)

    # Draw prediction boxes (red)
    for pred_box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = pred_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_title(f'{os.path.basename(img_path)} - Data-Point Precision: {precision:.3f}{title_suffix}',
                fontsize=14, weight='bold')
    ax.axis('off')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, alpha=0.6, label=f'GT Bboxes ({len(gt_boxes)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markeredgecolor='darkgreen', markersize=8, label=f'GT Points ({len(gt_points) if gt_points else 0})'),
        Line2D([0], [0], color='red', lw=2, label=f'Predictions ({len(pred_boxes)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    return fig

def visualize_gt_only(img_path, gt_boxes, gt_points=None, title_suffix=""):
    """Visualize only the ground truth (GT) boxes and points for an image."""
    img = plt.imread(img_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    # Draw ground truth boxes (green, semi-transparent)
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        ax.add_patch(rect)
    # Draw ground truth points (green dots)
    if gt_points:
        for gt_x, gt_y in gt_points:
            ax.plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
                   markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
    ax.set_title(f'{os.path.basename(img_path)} - Ground Truth{title_suffix}',
                fontsize=14, weight='bold')
    ax.axis('off')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, alpha=0.6, label=f'GT Bboxes ({len(gt_boxes)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markeredgecolor='darkgreen', markersize=8, label=f'GT Points ({len(gt_points) if gt_points else 0})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    return fig

def display_images_grid(results, title, max_cols=4, total_results=None):
    """Display a grid of images with their precision scores"""

    n_images = len(results)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Ensure axes is always 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Load and display image
        img = plt.imread(result['img_path'])
        ax.imshow(img)

        # Draw ground truth boxes (green, semi-transparent)
        for gt_box in result['gt_boxes']:
            x1, y1, x2, y2 = gt_box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor='green', facecolor='none', alpha=0.6)
            ax.add_patch(rect)

        # Draw ground truth points (green dots) - the actual points being evaluated
        if 'gt_points' in result and result['gt_points']:
            for gt_x, gt_y in result['gt_points']:
                ax.plot(gt_x, gt_y, 'go', markersize=6, markerfacecolor='lime',
                       markeredgecolor='darkgreen', markeredgewidth=1.5, alpha=0.9)

        # Draw prediction boxes (red)
        for pred_box, score in zip(result['pred_boxes'], result['pred_scores']):
            x1, y1, x2, y2 = pred_box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            # Add score text
            ax.text(x1, y1-3, f'{score:.2f}', color='red', fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

        # Set title with precision info
        if title.startswith("🏆"):
            rank = i + 1  # Top 20: ranks 1-20
        else:
            # Bottom 20: ranks from (total-19) to total
            rank = total_results - 20 + i + 1 if total_results else len(results) + i + 1

        ax.set_title(f'#{rank}: {os.path.basename(result["file_name"])}\n'
                    f'Precision: {result["precision"]:.3f} | GT:{result["num_gt"]} | Pred:{result["num_predictions"]}',
                    fontsize=10, weight='bold')
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def display_detailed_image(result, rank, category=""):
    """Display a single image with detailed annotations: GT-only and prediction overlay side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    # GT only
    img = plt.imread(result['img_path'])
    axes[0].imshow(img)
    for gt_box in result['gt_boxes']:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        axes[0].add_patch(rect)
    if result.get('gt_points'):
        for gt_x, gt_y in result['gt_points']:
            axes[0].plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
                   markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
    axes[0].set_title(f'GT: {os.path.basename(result["file_name"])}', fontsize=14, weight='bold', color='green')
    axes[0].axis('off')
    # Prediction overlay
    axes[1].imshow(img)
    for gt_box in result['gt_boxes']:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        axes[1].add_patch(rect)
    if result.get('gt_points'):
        for gt_x, gt_y in result['gt_points']:
            axes[1].plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
                   markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
    for pred_box, score in zip(result['pred_boxes'], result['pred_scores']):
        x1, y1, x2, y2 = pred_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    axes[1].set_title(f'Predictions: {os.path.basename(result["file_name"])}\nPrecision: {result["precision"]:.3f}', fontsize=14, weight='bold', color='red')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def save_and_display_analysis_results(results, output_dir=OUTPUT_DIR):
    """Save and display best and worst performing images with analysis"""

    os.makedirs(output_dir, exist_ok=True)

    # Sort by precision
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)

    # Get top 20 and bottom 20
    top_20 = results_sorted[:20]
    bottom_20 = results_sorted[-20:]

    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"Total images evaluated: {len(results)}")
    if len(results) > 0:
        print(f"Average precision: {np.mean([r['precision'] for r in results]):.3f}")
        print(f"Best precision: {results_sorted[0]['precision']:.3f}")
        print(f"Worst precision: {results_sorted[-1]['precision']:.3f}")
        print(f"Median precision: {np.median([r['precision'] for r in results]):.3f}")

    # DISPLAY TOP 20 IMAGES IN GRID
    print(f"\n🏆 DISPLAYING TOP 20 DATA-POINT PRECISION IMAGES:")
    display_images_grid(top_20, "🏆 TOP 20 - BEST Data-Point Detection Precision",
                       max_cols=DISPLAY_CONFIG['grid_columns'], total_results=len(results))

    # DISPLAY BOTTOM 20 IMAGES IN GRID
    print(f"\n📉 DISPLAYING BOTTOM 20 DATA-POINT PRECISION IMAGES:")
    display_images_grid(bottom_20, "📉 BOTTOM 20 - WORST Data-Point Detection Precision",
                       max_cols=DISPLAY_CONFIG['grid_columns'], total_results=len(results))

    # DISPLAY DETAILED VIEW OF TOP 5 AND BOTTOM 5
    if DISPLAY_CONFIG['show_detailed_view']:
        print(f"\n🔍 DETAILED VIEW - TOP 5 PERFORMERS:")
        for i, result in enumerate(top_20[:5]):
            print(f"\n📸 Displaying #{i+1}: {result['file_name']} (Precision: {result['precision']:.3f})")
            display_detailed_image(result, i+1, " - TOP PERFORMER")

        print(f"\n🔍 DETAILED VIEW - BOTTOM 5 PERFORMERS:")
        for i, result in enumerate(bottom_20[-5:]):
            rank = len(results) - 5 + i + 1
            print(f"\n📸 Displaying #{rank}: {result['file_name']} (Precision: {result['precision']:.3f})")
            display_detailed_image(result, rank, " - BOTTOM PERFORMER")
    else:
        print(f"\n⏭️  Skipping detailed view (disabled in config)")

    # Save files (existing functionality)
    if DISPLAY_CONFIG['save_files']:
        print(f"\n💾 SAVING FILES...")

        # Save top 20 visualizations
        print(f"🏆 Saving TOP 20 images...")
        top_dir = os.path.join(output_dir, 'top_20_precision')
        os.makedirs(top_dir, exist_ok=True)

        for i, result in enumerate(top_20):
            fig = visualize_image_results(
                result['img_path'],
                result['pred_boxes'],
                result['pred_scores'],
                result['gt_boxes'],
                result['precision'],
                f" (Rank {i+1})",
                result.get('gt_points', [])
            )
            fig.savefig(os.path.join(top_dir, f'rank_{i+1:02d}_{os.path.basename(result["file_name"])}'),
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # Save bottom 20 visualizations
        print(f"📉 Saving BOTTOM 20 images...")
        bottom_dir = os.path.join(output_dir, 'bottom_20_precision')
        os.makedirs(bottom_dir, exist_ok=True)

        for i, result in enumerate(bottom_20):
            fig = visualize_image_results(
                result['img_path'],
                result['pred_boxes'],
                result['pred_scores'],
                result['gt_boxes'],
                result['precision'],
                f" (Rank {len(results)-20+i+1})",
                result.get('gt_points', [])
            )
            fig.savefig(os.path.join(bottom_dir, f'rank_{len(results)-20+i+1:02d}_{os.path.basename(result["file_name"])}'),
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # Save detailed CSV results
        df_results = pd.DataFrame([{
            'rank': i+1,
            'file_name': r['file_name'],
            'precision': r['precision'],
            'tp': r['tp'],
            'fp': r['fp'],
            'num_predictions': r['num_predictions'],
            'num_gt': r['num_gt']
        } for i, r in enumerate(results_sorted)])

        csv_path = os.path.join(output_dir, 'scatter_plot_data_point_precision_results.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"📄 Saved detailed results: {csv_path}")
    else:
        print(f"\n⏭️  Skipping file saving (disabled in config)")
        csv_path = None

    # Print summaries
    print(f"\n🏆 TOP 20 DATA-POINT PRECISION:")
    for i, result in enumerate(top_20):
        print(f"{i+1:2d}. {result['file_name']:<30} | Precision: {result['precision']:.3f} | TP: {result['tp']}, FP: {result['fp']}, GT: {result['num_gt']}")

    print(f"\n📉 BOTTOM 20 DATA-POINT PRECISION:")
    for i, result in enumerate(bottom_20):
        print(f"{len(results)-20+i+1:2d}. {result['file_name']:<30} | Precision: {result['precision']:.3f} | TP: {result['tp']}, FP: {result['fp']}, GT: {result['num_gt']}")

    return top_20, bottom_20, csv_path

def show_highest_lowest_precision_scatter_plots(results, output_dir=OUTPUT_DIR):
    """Display the highest and lowest precision scatter plots with detailed annotations"""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Sort by precision
    results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
    
    print(f"\n🎯 HIGHEST AND LOWEST PRECISION SCATTER PLOTS ANALYSIS")
    print(f"=" * 60)
    
    # Get the best and worst performers
    best_result = results_sorted[0]
    worst_result = results_sorted[-1]
    
    print(f"\n🏆 HIGHEST PRECISION SCATTER PLOT:")
    print(f"   📄 File: {best_result['file_name']}")
    print(f"   🎯 Precision: {best_result['precision']:.3f}")
    print(f"   ✅ True Positives: {best_result['tp']}")
    print(f"   ❌ False Positives: {best_result['fp']}")
    print(f"   🎯 Ground Truth Points: {best_result['num_gt']}")
    print(f"   🔍 Predictions: {best_result['num_predictions']}")
    
    print(f"\n📉 LOWEST PRECISION SCATTER PLOT:")
    print(f"   📄 File: {worst_result['file_name']}")
    print(f"   🎯 Precision: {worst_result['precision']:.3f}")
    print(f"   ✅ True Positives: {worst_result['tp']}")
    print(f"   ❌ False Positives: {worst_result['fp']}")
    print(f"   🎯 Ground Truth Points: {worst_result['num_gt']}")
    print(f"   🔍 Predictions: {worst_result['num_predictions']}")
    
    # Display the best and worst images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Best image
    img1 = plt.imread(best_result['img_path'])
    ax1.imshow(img1)
    
    # Draw ground truth boxes and points for best image
    for gt_box in best_result['gt_boxes']:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        ax1.add_patch(rect)
    
    for gt_x, gt_y in best_result['gt_points']:
        ax1.plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
               markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
    
    # Draw prediction boxes for best image
    for pred_box, score in zip(best_result['pred_boxes'], best_result['pred_scores']):
        x1, y1, x2, y2 = pred_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax1.set_title(f'🏆 HIGHEST PRECISION: {best_result["precision"]:.3f}\n{best_result["file_name"]}', 
                  fontsize=14, weight='bold', color='green')
    ax1.axis('off')
    
    # Worst image
    img2 = plt.imread(worst_result['img_path'])
    ax2.imshow(img2)
    
    # Draw ground truth boxes and points for worst image
    for gt_box in worst_result['gt_boxes']:
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
        ax2.add_patch(rect)
    
    for gt_x, gt_y in worst_result['gt_points']:
        ax2.plot(gt_x, gt_y, 'go', markersize=8, markerfacecolor='lime',
               markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.9)
    
    # Draw prediction boxes for worst image
    for pred_box, score in zip(worst_result['pred_boxes'], worst_result['pred_scores']):
        x1, y1, x2, y2 = pred_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax2.set_title(f'📉 LOWEST PRECISION: {worst_result["precision"]:.3f}\n{worst_result["file_name"]}', 
                  fontsize=14, weight='bold', color='red')
    ax2.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, alpha=0.6, label='Ground Truth Boxes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markeredgecolor='darkgreen', markersize=8, label='Ground Truth Points'),
        Line2D([0], [0], color='red', lw=2, label='Predictions')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Save the comparison
    if DISPLAY_CONFIG['save_files']:
        comparison_path = os.path.join(output_dir, 'highest_vs_lowest_precision_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved comparison: {comparison_path}")
    
    # Show top 10 and bottom 10 in a table format
    print(f"\n📊 TOP 10 HIGHEST PRECISION SCATTER PLOTS:")
    print(f"{'Rank':<4} {'Precision':<10} {'TP':<4} {'FP':<4} {'GT':<4} {'File Name':<30}")
    print("-" * 70)
    for i, result in enumerate(results_sorted[:10]):
        print(f"{i+1:<4} {result['precision']:<10.3f} {result['tp']:<4} {result['fp']:<4} {result['num_gt']:<4} {result['file_name']:<30}")
    
    print(f"\n📊 BOTTOM 10 LOWEST PRECISION SCATTER PLOTS:")
    print(f"{'Rank':<4} {'Precision':<10} {'TP':<4} {'FP':<4} {'GT':<4} {'File Name':<30}")
    print("-" * 70)
    for i, result in enumerate(results_sorted[-10:]):
        rank = len(results_sorted) - 10 + i + 1
        print(f"{rank:<4} {result['precision']:<10.3f} {result['tp']:<4} {result['fp']:<4} {result['num_gt']:<4} {result['file_name']:<30}")
    
    return best_result, worst_result

def main():
    """Main execution function"""

    print("🚀 Starting Data Point Precision Analysis (SCATTER PLOTS ONLY)")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🎯 Focus: Scatter plots only")

    try:
        print(f"📁 Script location: {os.path.dirname(os.path.abspath(__file__))}")
    except NameError:
        print("📁 Running in notebook mode (no __file__ available)")

    if not MMDET_AVAILABLE:
        print("❌ MMDetection not available. Exiting.")
        return

    # Load model
    model, config = load_model()
    if model is None:
        print("❌ Failed to load model.")
        return

    # Load validation data
    val_data, val_img_dir = load_validation_data()
    if val_data is None:
        print("❌ Failed to load validation data.")
        print("\n💡 GOOD NEWS: Your model loaded successfully!")
        print("💡 To run the full analysis, you need to upload validation data.")
        print("\n📋 NEXT STEPS:")
        print("   1. Upload your validation annotation file (val_enriched_with_info.json)")
        print("   2. Upload your validation images directory")
        print("   3. Or copy them to the Google Drive CHART-DeMatch folder")
        return

    # Evaluate data-point precision
    results = evaluate_data_point_precision(
        model, val_data, val_img_dir,
        iou_threshold=DISPLAY_CONFIG['iou_threshold'],
        score_threshold=DISPLAY_CONFIG['score_threshold'],
        max_images=DISPLAY_CONFIG['max_images_to_process']
    )

    if len(results) == 0:
        print("❌ No valid results obtained. Exiting.")
        return

    # Save and display analysis results
    top_20, bottom_20, csv_path = save_and_display_analysis_results(results)
    
    # Show highest and lowest precision scatter plots with annotations
    best_result, worst_result = show_highest_lowest_precision_scatter_plots(results)

    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    print(f"🏆 Top 20 images: {os.path.join(OUTPUT_DIR, 'top_20_precision/')}")
    print(f"📉 Bottom 20 images: {os.path.join(OUTPUT_DIR, 'bottom_20_precision/')}")
    print(f"📄 Detailed CSV: {csv_path}")
    print(f"🆚 Comparison: {os.path.join(OUTPUT_DIR, 'highest_vs_lowest_precision_comparison.png')}")

    return results

def quick_test_model():
    """Quick test to see if the model loads and can make predictions"""
    print("\n🚀 QUICK MODEL TEST")
    print("=" * 40)

    # Try to load just the model
    model, config = load_model()
    if model is None:
        print("❌ Model loading failed")
        return

    print("✅ Model loaded successfully!")
    print("🎯 Model details:")
    print(f"   • Architecture: {model.__class__.__name__}")
    print(f"   • Device: {next(model.parameters()).device}")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n💡 The model is ready for inference!")
    print("💡 To test on an image, use: inference_detector(model, 'your_image.jpg')")

    return model

def test_model_on_image(image_path):
    """Test the loaded model on a specific image"""
    print(f"\n🧪 TESTING MODEL ON: {image_path}")
    print("=" * 50)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load model
    model, config = load_model()
    if model is None:
        print("❌ Model loading failed")
        return

    try:
        # Run inference
        print("🔄 Running inference...")
        result = inference_detector(model, image_path)

        # Process results
        if hasattr(result, 'pred_instances'):
            pred_boxes = result.pred_instances.bboxes.cpu().numpy()
            pred_scores = result.pred_instances.scores.cpu().numpy()
            pred_labels = result.pred_instances.labels.cpu().numpy()
        else:
            # Handle older format
            pred_boxes = []
            pred_scores = []
            pred_labels = []
            for class_id, detections in enumerate(result):
                if len(detections) > 0:
                    pred_boxes.extend(detections[:, :4])
                    pred_scores.extend(detections[:, 4])
                    pred_labels.extend([class_id] * len(detections))
            pred_boxes = np.array(pred_boxes) if pred_boxes else np.empty((0, 4))
            pred_scores = np.array(pred_scores) if pred_scores else np.empty(0)
            pred_labels = np.array(pred_labels) if pred_labels else np.empty(0)

        # Filter for data-point class
        data_point_mask = pred_labels == DATA_POINT_CLASS_ID
        data_point_boxes = pred_boxes[data_point_mask]
        data_point_scores = pred_scores[data_point_mask]

        # Show results
        print(f"✅ Inference completed!")
        print(f"📊 Total detections: {len(pred_boxes)}")
        print(f"🎯 Data-point detections: {len(data_point_boxes)}")

        if len(data_point_boxes) > 0:
            print(f"📈 Data-point scores: {data_point_scores}")
            print(f"💡 Highest scoring data-point: {data_point_scores.max():.3f}")

        # Simple visualization
        img = plt.imread(image_path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)

        # Draw data-point detections
        for i, (box, score) in enumerate(zip(data_point_boxes, data_point_scores)):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=10, weight='bold')

        ax.set_title(f'Data-Point Detection Test\n{os.path.basename(image_path)} - {len(data_point_boxes)} data points found')
        ax.axis('off')
        plt.show()

        return result

    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def setup_mmdetection():
    """Setup mmdetection for Colab environment"""
    print("🔧 SETTING UP MMDETECTION FOR COLAB")
    print("=" * 50)
    
    # Check if mmdetection is already in path
    try:
        import mmdet
        print("✅ mmdet already available")
        return True
    except ImportError:
        pass
    
    # Try to find mmdetection directory
    mmdetection_paths = [
        '/content/mmdetection',
        './mmdetection',
        '../mmdetection',
        os.path.join(os.getcwd(), 'mmdetection')
    ]
    
    for path in mmdetection_paths:
        if os.path.exists(path):
            print(f"✅ Found mmdetection at: {path}")
            sys.path.insert(0, path)
            
            # Try to install if needed
            try:
                import subprocess
                print("📦 Installing mmdetection...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", path])
                print("✅ mmdetection installed successfully")
                return True
            except Exception as e:
                print(f"⚠️  Installation failed: {e}")
                print("💡 Trying to use without installation...")
                return True
    
    print("❌ mmdetection not found in common locations")
    print("💡 Please ensure mmdetection is available in your environment")
    return False

def set_checkpoint_path(path):
    """Manually set the checkpoint path"""
    global CHECKPOINT_PATH
    if os.path.exists(path):
        CHECKPOINT_PATH = path
        print(f"✅ Set checkpoint path to: {CHECKPOINT_PATH}")
        return True
    else:
        print(f"❌ File not found: {path}")
        return False

def list_available_files():
    """List all available files in common locations"""
    print("🔍 SCANNING FOR AVAILABLE FILES")
    print("=" * 50)
    
    search_locations = [
        '/content',
        '/content/drive/MyDrive',
        os.getcwd(),
        '.'
    ]
    
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        print(f"\n📁 {location}:")
        try:
            files = os.listdir(location)
            for file in sorted(files):
                if file.endswith(('.pth', '.json', '.py')) or 'legend' in file.lower():
                    full_path = os.path.join(location, file)
                    if os.path.isdir(full_path):
                        print(f"   📁 {file}/")
                    else:
                        size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                        print(f"   📄 {file} ({size:,} bytes)")
        except Exception as e:
            print(f"   ❌ Error reading {location}: {e}")
    
    print("\n💡 If you don't see chart_datapoint.pth, please upload it to Colab.")

def find_validation_data():
    """Find validation data files"""
    print("🔍 SEARCHING FOR VALIDATION DATA")
    print("=" * 50)
    
    # Check the specific location you mentioned
    enhanced_datasets_path = "legend_match_swin/mask_generation/enhanced_datasets"
    
    if os.path.exists(enhanced_datasets_path):
        print(f"✅ Found enhanced_datasets directory: {enhanced_datasets_path}")
        
        try:
            files = os.listdir(enhanced_datasets_path)
            print(f"📁 Contents of {enhanced_datasets_path}:")
            for file in sorted(files):
                full_path = os.path.join(enhanced_datasets_path, file)
                if os.path.isdir(full_path):
                    print(f"   📁 {file}/")
                    # Check subdirectory contents
                    try:
                        subfiles = os.listdir(full_path)
                        for subfile in subfiles[:10]:  # Show first 10
                            print(f"      📄 {subfile}")
                        if len(subfiles) > 10:
                            print(f"      ... and {len(subfiles)-10} more files")
                    except:
                        pass
                else:
                    size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    print(f"   📄 {file} ({size:,} bytes)")
        except Exception as e:
            print(f"   ❌ Error reading directory: {e}")
    else:
        print(f"❌ Enhanced datasets directory not found: {enhanced_datasets_path}")
    
    # Also check for the specific validation file
    val_file = "legend_data/annotations_JSON_cleaned/val_enriched_with_info.json"
    if os.path.exists(val_file):
        print(f"✅ Found validation file: {val_file}")
        size = os.path.getsize(val_file)
        print(f"   Size: {size:,} bytes")
    else:
        print(f"❌ Validation file not found: {val_file}")
        print(f"💡 Expected path: {os.path.abspath(val_file)}")
    
    # Check for images directory in legend_data/train/images
    images_dir = "legend_data/train/images"
    if os.path.exists(images_dir):
        print(f"✅ Found images directory: {images_dir}")
        try:
            img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"   Contains {len(img_files)} image files")
            if img_files:
                print(f"   Sample images: {img_files[:5]}")
        except Exception as e:
            print(f"   ❌ Error reading images: {e}")
    else:
        print(f"❌ Images directory not found: {images_dir}")
        print("💡 Please ensure legend_data/train/images/ directory exists with your image files")
    
    return enhanced_datasets_path

# Add a simple function to run the analysis
def run_scatter_plot_analysis():
    """Run the scatter plot data point precision analysis"""
    print("🚀 RUNNING SCATTER PLOT DATA POINT PRECISION ANALYSIS")
    print("=" * 60)
    print("🎯 This will analyze scatter plots from val_enriched_with_info.json")
    print("📊 Showing highest and lowest precision data point detection")
    print("=" * 60)
    
    return main()

if __name__ == "__main__":
    main()

# Auto-run when executed in notebook (helpful for Colab)
try:
    # Check if we're in a notebook environment
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("\n💡 COLAB USAGE OPTIONS:")
        print("   find_validation_data()    - Find validation data files")
        print("   list_available_files()    - Show all available files")
        print("   set_checkpoint_path(path) - Manually set checkpoint path")
        print("   setup_mmdetection()       - Setup mmdetection for Colab")
        print("   main()                    - Full analysis (needs validation data)")
        print("   run_scatter_plot_analysis() - Run scatter plot analysis specifically")
        print("   quick_test_model()        - Test model loading only")
        print("   test_model_on_image(path) - Test inference on single image")
        print("\n🎯 If you can't find validation data, try:")
        print("   find_validation_data()    # Look for validation files")
        print("   list_available_files()    # See what files are available")
        print("   run_scatter_plot_analysis() # Run the analysis")
except:
    pass 