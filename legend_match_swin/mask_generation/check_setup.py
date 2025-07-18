#!/usr/bin/env python3
"""
Check mask generation setup and validate paths for the updated annotation system.
"""

import json
import sys
from pathlib import Path

def check_annotation_files():
    """Check if the required annotation files exist with correct structure."""
    print("🔍 Checking annotation files...")
    
    base_dir = Path("legend_data/annotations_JSON_cleaned")
    required_files = [
        "train_enriched.json",
        "val_enriched_with_info.json"
    ]
    
    if not base_dir.exists():
        print(f"❌ Annotations directory not found: {base_dir}")
        print("💡 Make sure you're running from the project root directory")
        return False
    
    missing_files = []
    valid_files = []
    
    for filename in required_files:
        file_path = base_dir / filename
        if file_path.exists():
            # Check if it has the expected structure
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Validate structure
                required_keys = ['info', 'categories', 'images', 'annotations']
                if all(key in data for key in required_keys):
                    # Check for data-point category (ID 11)
                    data_point_cat = None
                    for cat in data['categories']:
                        if cat['name'] == 'data-point' and cat['id'] == 11:
                            data_point_cat = cat
                            break
                    
                    if data_point_cat:
                        # Count data-point annotations
                        data_point_annotations = [
                            ann for ann in data['annotations'] 
                            if ann['category_id'] == 11
                        ]
                        
                        print(f"✅ {filename}: {len(data['images'])} images, "
                              f"{len(data_point_annotations)} data-point annotations")
                        valid_files.append(filename)
                    else:
                        print(f"⚠️ {filename}: Missing 'data-point' category with ID 11")
                else:
                    print(f"⚠️ {filename}: Invalid COCO structure")
                    
            except Exception as e:
                print(f"❌ {filename}: Error reading file - {e}")
        else:
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
    
    if missing_files:
        print(f"\n💡 Missing files: {missing_files}")
        print("   Run the merge and metadata scripts first:")
        print("   1. python merge_and_split_benetech.py")
        print("   2. python legend_match_swin/add_metadata_to_annotations.py")
        return False
    
    return len(valid_files) == len(required_files)

def check_image_directory():
    """Check if the image directory exists."""
    print("\n🖼️ Checking image directory...")
    
    img_dir = Path("legend_data/train/images")
    if img_dir.exists():
        image_count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"✅ Images directory found: {image_count} images")
        return True
    else:
        print(f"❌ Images directory not found: {img_dir}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'sklearn', 'pycocotools'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def print_usage_examples():
    """Print usage examples for the mask generation system."""
    print("\n🚀 Mask Generation Usage Examples:")
    print("="*50)
    
    print("\n1. Auto-select image and generate masks:")
    print("   python legend_match_swin/mask_generation/generate_masks_demo.py --auto_select_image")
    
    print("\n2. Process specific image:")
    print("   python legend_match_swin/mask_generation/generate_masks_demo.py \\")
    print("       --input_image legend_data/train/images/your_image.jpg")
    
    print("\n3. Train Mask R-CNN (after generating masks):")
    print("   # Update config file paths first, then:")
    print("   python mmdetection/tools/train.py \\")
    print("       legend_match_swin/mask_generation/mask_rcnn_swin_datapoint.py")

def main():
    """Main setup checker."""
    print("🔧 Mask Generation Setup Checker")
    print("="*40)
    
    # Check annotation files
    annotations_ok = check_annotation_files()
    
    # Check image directory
    images_ok = check_image_directory()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Overall status
    print("\n📋 Setup Status:")
    print("="*20)
    
    if annotations_ok and images_ok and deps_ok:
        print("🎉 All checks passed! Ready for mask generation.")
        print_usage_examples()
        return 0
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 