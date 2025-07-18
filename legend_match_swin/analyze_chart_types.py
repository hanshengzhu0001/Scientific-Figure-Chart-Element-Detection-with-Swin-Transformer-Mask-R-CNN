#!/usr/bin/env python3
"""
Chart Type Analysis Script
Analyzes all chart types present in annotation files to understand dataset distribution.
"""

import json
import os
from collections import Counter, defaultdict
import argparse

def analyze_chart_types(annotation_file):
    """Analyze chart types in annotation file."""
    
    print(f"📊 Analyzing chart types in: {annotation_file}")
    print("=" * 60)
    
    # Load annotation file
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found - {annotation_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON - {e}")
        return
    
    # Extract chart types from images
    chart_types = []
    chart_type_examples = defaultdict(list)
    missing_chart_type = []
    
    for img in data.get('images', []):
        chart_type = img.get('chart_type', '')
        img_filename = img.get('file_name', f"image_{img.get('id', 'unknown')}")
        
        if chart_type:
            chart_types.append(chart_type.lower())
            # Store first few examples for each chart type
            if len(chart_type_examples[chart_type.lower()]) < 3:
                chart_type_examples[chart_type.lower()].append(img_filename)
        else:
            missing_chart_type.append(img_filename)
    
    # Count and display results
    chart_type_counts = Counter(chart_types)
    total_images = len(data.get('images', []))
    
    print(f"📈 CHART TYPE DISTRIBUTION:")
    print(f"Total images: {total_images}")
    print(f"Images with chart_type: {len(chart_types)}")
    print(f"Images missing chart_type: {len(missing_chart_type)}")
    print()
    
    # Display chart type statistics
    print(f"📊 CHART TYPE BREAKDOWN:")
    for chart_type, count in chart_type_counts.most_common():
        percentage = (count / total_images) * 100
        print(f"   • {chart_type:<15}: {count:4d} images ({percentage:5.1f}%)")
    
    print()
    
    # Show examples for each chart type
    print(f"📁 EXAMPLES BY CHART TYPE:")
    for chart_type, count in chart_type_counts.most_common():
        examples = chart_type_examples[chart_type]
        examples_str = ", ".join(examples)
        print(f"   • {chart_type:<15}: {examples_str}")
    
    print()
    
    # Check against our chart-type filtering
    print(f"🎯 CHART-TYPE FILTERING COVERAGE:")
    
    # Our defined chart types from the filtering system (based on your dataset)
    DEFINED_CHART_TYPES = {
        'line',                     # data-line (41.9% - 1710 images)
        'scatter', 'dot',           # data-point (27.4% - 1116 images total)
        'vertical_bar', 'horizontal_bar',  # data-bar (30.7% - 1255 images total)
    }
    
    found_chart_types = set(chart_type_counts.keys())
    
    # Chart types covered by our filtering
    covered = found_chart_types & DEFINED_CHART_TYPES
    # Chart types not covered by our filtering
    not_covered = found_chart_types - DEFINED_CHART_TYPES
    # Defined types not found in data
    not_found = DEFINED_CHART_TYPES - found_chart_types
    
    print(f"   ✅ Covered by filtering ({len(covered)}): {sorted(covered)}")
    if not_covered:
        print(f"   ⚠️  NOT covered by filtering ({len(not_covered)}): {sorted(not_covered)}")
    if not_found:
        print(f"   📝 Defined but not found ({len(not_found)}): {sorted(not_found)}")
    
    print()
    
    # Show missing chart type examples
    if missing_chart_type:
        print(f"⚠️  IMAGES MISSING CHART_TYPE ({len(missing_chart_type)}):")
        for i, filename in enumerate(missing_chart_type[:10]):  # Show first 10
            print(f"   • {filename}")
        if len(missing_chart_type) > 10:
            print(f"   ... and {len(missing_chart_type) - 10} more")
    
    print()
    
    # Data element analysis
    print(f"📊 DATA ELEMENT ANALYSIS:")
    analyze_data_elements(data, chart_type_counts)
    
    return chart_type_counts

def analyze_data_elements(data, chart_type_counts):
    """Analyze data elements (data-point, data-line, data-bar) by chart type."""
    
    # Map category IDs to names (assuming standard order)
    CLASSES = [
        'title', 'subtitle', 'x-axis', 'y-axis', 'x-axis-label', 'y-axis-label',
        'x-tick-label', 'y-tick-label', 'legend', 'legend-title', 'legend-item', 
        'data-point', 'data-line', 'data-bar', 'data-area', 'grid-line',
        'axis-title', 'tick-label', 'data-label', 'legend-text', 'plot-area'
    ]
    
    # Map image IDs to chart types
    img_id_to_chart_type = {}
    for img in data.get('images', []):
        chart_type = img.get('chart_type', '').lower()
        img_id_to_chart_type[img['id']] = chart_type
    
    # Count data elements by chart type
    data_elements = {'data-point', 'data-line', 'data-bar', 'data-area'}
    chart_type_data_elements = defaultdict(lambda: defaultdict(int))
    
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        category_id = ann['category_id'] 
        
        # Convert to 0-indexed (COCO is 1-indexed)
        class_idx = category_id - 1
        if 0 <= class_idx < len(CLASSES):
            class_name = CLASSES[class_idx]
            if class_name in data_elements:
                chart_type = img_id_to_chart_type.get(img_id, 'unknown')
                chart_type_data_elements[chart_type][class_name] += 1
    
    # Display results
    for chart_type in sorted(chart_type_data_elements.keys()):
        if chart_type == 'unknown' or chart_type == '':
            continue
            
        elements = chart_type_data_elements[chart_type]
        if elements:
            element_str = ", ".join([f"{elem}: {count}" for elem, count in elements.items()])
            print(f"   • {chart_type:<15}: {element_str}")
    
    # Check for problematic combinations
    print()
    print(f"🚫 PROBLEMATIC COMBINATIONS (should be filtered):")
    problems_found = False
    
    for chart_type, elements in chart_type_data_elements.items():
        if chart_type in ['scatter', 'dot'] and ('data-line' in elements or 'data-bar' in elements):
            problems_found = True
            print(f"   ⚠️  {chart_type} has data-line/data-bar: {dict(elements)}")
        elif chart_type == 'line' and ('data-point' in elements or 'data-bar' in elements):
            problems_found = True
            print(f"   ⚠️  {chart_type} has data-point/data-bar: {dict(elements)}")
        elif chart_type in ['bar', 'vertical_bar', 'horizontal_bar'] and ('data-point' in elements or 'data-line' in elements):
            problems_found = True
            print(f"   ⚠️  {chart_type} has data-point/data-line: {dict(elements)}")
    
    if not problems_found:
        print(f"   ✅ No problematic combinations found!")

def main():
    """Main function with command line argument support."""
    
    parser = argparse.ArgumentParser(description='Analyze chart types in annotation files')
    parser.add_argument('--ann-file', type=str, 
                       help='Path to annotation file (if not provided, will search common locations)')
    parser.add_argument('--data-root', type=str, default='.',
                       help='Data root directory (default: current directory)')
    
    # Handle Colab/Jupyter kernel arguments
    import sys
    # Filter out Jupyter/Colab specific arguments (-f and the following argument)
    filtered_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == '-f':
            skip_next = True  # Skip both -f and the next argument
            continue
        if arg.startswith('-f'):
            continue  # Skip -f=value format
        filtered_args.append(arg)
    
    args = parser.parse_args(filtered_args)
    
    # Find annotation files if not specified
    if args.ann_file:
        annotation_files = [args.ann_file]
    else:
        # Search common annotation file locations
        search_paths = [
            'legend_data/annotations_JSON_cleaned/train_enriched_with_info.json',
            'legend_data/annotations_JSON_cleaned/val_enriched_with_info.json',
            'legend_data/annotations_JSON_cleaned/train_enriched.json', 
            'legend_data/annotations_JSON_cleaned/val_enriched.json',
            'legend_data/annotations_JSON_cleaned/train.json',
            'legend_data/annotations_JSON_cleaned/val.json',
            'annotations_JSON/train_enriched_with_info.json',
            'annotations_JSON/val_enriched_with_info.json',
            'train_enriched_with_info.json',
            'val_enriched_with_info.json'
        ]
        
        annotation_files = []
        for path in search_paths:
            full_path = os.path.join(args.data_root, path)
            if os.path.exists(full_path):
                annotation_files.append(full_path)
        
        if not annotation_files:
            print("❌ No annotation files found!")
            print("💡 Try specifying --ann-file path/to/annotations.json")
            print("💡 Or run from the correct data directory")
            return
    
    # Analyze each annotation file
    all_chart_types = Counter()
    
    for ann_file in annotation_files:
        chart_types = analyze_chart_types(ann_file)
        if chart_types:
            all_chart_types.update(chart_types)
        print()
    
    # Combined summary if multiple files
    if len(annotation_files) > 1:
        print("🎯 COMBINED SUMMARY:")
        print("=" * 40)
        total_images = sum(all_chart_types.values())
        print(f"Total images across all files: {total_images}")
        
        for chart_type, count in all_chart_types.most_common():
            percentage = (count / total_images) * 100
            print(f"   • {chart_type:<15}: {count:4d} images ({percentage:5.1f}%)")

# Colab-friendly function (no command line args)
def run_analysis(data_root=".", ann_file=None):
    """
    Colab-friendly function to run chart type analysis.
    
    Args:
        data_root: Root directory to search for annotation files
        ann_file: Specific annotation file path (optional)
    """
    if ann_file:
        annotation_files = [ann_file]
    else:
        # Search common annotation file locations
        search_paths = [
            'legend_data/annotations_JSON_cleaned/train_enriched_with_info.json',
            'legend_data/annotations_JSON_cleaned/val_enriched_with_info.json',
            'legend_data/annotations_JSON_cleaned/train_enriched.json', 
            'legend_data/annotations_JSON_cleaned/val_enriched.json',
            'legend_data/annotations_JSON_cleaned/train.json',
            'legend_data/annotations_JSON_cleaned/val.json',
            'annotations_JSON/train_enriched_with_info.json',
            'annotations_JSON/val_enriched_with_info.json',
            'train_enriched_with_info.json',
            'val_enriched_with_info.json'
        ]
        
        annotation_files = []
        for path in search_paths:
            full_path = os.path.join(data_root, path)
            if os.path.exists(full_path):
                annotation_files.append(full_path)
        
        if not annotation_files:
            print("❌ No annotation files found!")
            print("💡 Try specifying ann_file parameter")
            print("💡 Or set data_root to the correct directory")
            return
    
    # Analyze each annotation file
    all_chart_types = Counter()
    
    for ann_file in annotation_files:
        chart_types = analyze_chart_types(ann_file)
        if chart_types:
            all_chart_types.update(chart_types)
        print()
    
    # Combined summary if multiple files
    if len(annotation_files) > 1:
        print("🎯 COMBINED SUMMARY:")
        print("=" * 40)
        total_images = sum(all_chart_types.values())
        print(f"Total images across all files: {total_images}")
        
        for chart_type, count in all_chart_types.most_common():
            percentage = (count / total_images) * 100
            print(f"   • {chart_type:<15}: {count:4d} images ({percentage:5.1f}%)")
    
    return all_chart_types

if __name__ == "__main__":
    main()

# Colab function using exact paths from your config
def analyze_your_annotations():
    """
    Analyze chart types using the exact paths from your training config.
    """
    # Your exact paths from cascade_rcnn_r50_fpn_meta.py
    base_path = "/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch"
    
    # Your exact annotation files
    train_ann = f"{base_path}/legend_data/annotations_JSON_cleaned/train_enriched.json"
    val_ann = f"{base_path}/legend_data/annotations_JSON_cleaned/val_enriched_with_info.json"
    
    print("🎯 Analyzing YOUR annotation files from training config:")
    print(f"📁 Base path: {base_path}")
    print()
    
    all_chart_types = Counter()
    
    # Analyze training annotations
    if os.path.exists(train_ann):
        print("=" * 60)
        print("📊 TRAINING SET ANALYSIS")
        print("=" * 60)
        train_types = analyze_chart_types(train_ann)
        if train_types:
            all_chart_types.update(train_types)
    else:
        print(f"❌ Training annotations not found: {train_ann}")
    
    print()
    
    # Analyze validation annotations  
    if os.path.exists(val_ann):
        print("=" * 60)
        print("📊 VALIDATION SET ANALYSIS")
        print("=" * 60)
        val_types = analyze_chart_types(val_ann)
        if val_types:
            all_chart_types.update(val_types)
    else:
        print(f"❌ Validation annotations not found: {val_ann}")
    
    # Combined summary
    if all_chart_types:
        print()
        print("🎯 COMBINED TRAIN + VAL SUMMARY:")
        print("=" * 40)
        total_images = sum(all_chart_types.values())
        print(f"Total images across both sets: {total_images}")
        
        for chart_type, count in all_chart_types.most_common():
            percentage = (count / total_images) * 100
            print(f"   • {chart_type:<15}: {count:4d} images ({percentage:5.1f}%)")
    
    return all_chart_types

# For Colab users - you can call:
# analyze_your_annotations()  # Uses your exact config paths
# run_analysis()  # Auto-detect files
# run_analysis("/content/drive/MyDrive/your/data/path")  # Specify data root
# run_analysis(ann_file="/path/to/specific/file.json")  # Specific file 