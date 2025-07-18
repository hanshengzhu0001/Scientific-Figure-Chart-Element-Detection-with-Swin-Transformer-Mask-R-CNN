"""
Semi-automatic mask generation for data points in scientific charts.

This module implements OpenCV-based contour detection and polygon approximation
to generate instance segmentation masks for various data point shapes including
circles, squares, triangles, crosses, diamonds, stars, and other markers.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPointMaskGenerator:
    """
    Generates masks for data points using semi-automatic OpenCV methods.
    
    Supports multiple detection strategies for various marker shapes:
    - Circles: HoughCircles detection
    - Squares/Rectangles: Corner detection and aspect ratio analysis
    - Triangles: Polygon approximation and vertex counting
    - Crosses/Plus signs: Line intersection detection
    - Diamonds: Rotated square detection
    - Stars: Complex polygon analysis
    - Color-based thresholding for consistent chart colors
    - Contour-based detection with shape classification
    """
    
    def __init__(self, 
                 color_tolerance: int = 30,
                 min_contour_area: int = 10,
                 max_contour_area: int = 2000,
                 circle_dp: float = 1.0,
                 circle_min_dist: int = 20,
                 circle_param1: int = 50,
                 circle_param2: int = 30,
                 circle_min_radius: int = 3,
                 circle_max_radius: int = 50):
        """
        Initialize the mask generator with detection parameters.
        
        Args:
            color_tolerance: Color tolerance for thresholding (0-255)
            min_contour_area: Minimum contour area to consider
            max_contour_area: Maximum contour area to consider
            circle_dp: Inverse ratio of accumulator resolution for HoughCircles
            circle_min_dist: Minimum distance between circle centers
            circle_param1: Upper threshold for edge detection in HoughCircles
            circle_param2: Accumulator threshold for center detection
            circle_min_radius: Minimum circle radius
            circle_max_radius: Maximum circle radius
        """
        self.color_tolerance = color_tolerance
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        
        # HoughCircles parameters
        self.circle_dp = circle_dp
        self.circle_min_dist = circle_min_dist
        self.circle_param1 = circle_param1
        self.circle_param2 = circle_param2
        self.circle_min_radius = circle_min_radius
        self.circle_max_radius = circle_max_radius
        
    def create_marker_templates(self, size: int = 20) -> Dict[str, np.ndarray]:
        """
        Create template images for different marker shapes.
        
        Args:
            size: Template size (size x size pixels)
            
        Returns:
            Dictionary mapping shape names to template images
        """
        templates = {}
        center = size // 2
        
        # Circle template
        circle_template = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(circle_template, (center, center), size//3, 255, -1)
        templates['circle'] = circle_template
        
        # Square template
        square_template = np.zeros((size, size), dtype=np.uint8)
        margin = size // 4
        cv2.rectangle(square_template, (margin, margin), (size-margin, size-margin), 255, -1)
        templates['square'] = square_template
        
        # Triangle (up) template
        triangle_template = np.zeros((size, size), dtype=np.uint8)
        pts = np.array([[center, margin], [margin, size-margin], [size-margin, size-margin]], np.int32)
        cv2.fillPoly(triangle_template, [pts], 255)
        templates['triangle_up'] = triangle_template
        
        # Triangle (down) template
        triangle_down = np.zeros((size, size), dtype=np.uint8)
        pts = np.array([[center, size-margin], [margin, margin], [size-margin, margin]], np.int32)
        cv2.fillPoly(triangle_down, [pts], 255)
        templates['triangle_down'] = triangle_down
        
        # Diamond template
        diamond_template = np.zeros((size, size), dtype=np.uint8)
        pts = np.array([[center, margin], [size-margin, center], [center, size-margin], [margin, center]], np.int32)
        cv2.fillPoly(diamond_template, [pts], 255)
        templates['diamond'] = diamond_template
        
        # Cross/Plus template
        cross_template = np.zeros((size, size), dtype=np.uint8)
        thickness = max(2, size // 8)
        cv2.line(cross_template, (center, margin), (center, size-margin), 255, thickness)
        cv2.line(cross_template, (margin, center), (size-margin, center), 255, thickness)
        templates['cross'] = cross_template
        
        # X template
        x_template = np.zeros((size, size), dtype=np.uint8)
        cv2.line(x_template, (margin, margin), (size-margin, size-margin), 255, thickness)
        cv2.line(x_template, (size-margin, margin), (margin, size-margin), 255, thickness)
        templates['x_mark'] = x_template
        
        # Star template (simple 5-pointed)
        star_template = np.zeros((size, size), dtype=np.uint8)
        outer_radius = size // 3
        inner_radius = size // 6
        angles = np.linspace(0, 2*np.pi, 11)  # 10 points (5 outer, 5 inner)
        star_points = []
        for i, angle in enumerate(angles[:-1]):
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = int(center + radius * np.cos(angle - np.pi/2))
            y = int(center + radius * np.sin(angle - np.pi/2))
            star_points.append([x, y])
        star_pts = np.array(star_points, np.int32)
        cv2.fillPoly(star_template, [star_pts], 255)
        templates['star'] = star_template
        
        return templates
        
    def classify_contour_shape(self, contour: np.ndarray) -> Tuple[str, float]:
        """
        Classify a contour by its geometric properties.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Tuple of (shape_name, confidence_score)
        """
        # Calculate basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0 or perimeter == 0:
            return "unknown", 0.0
            
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Calculate additional properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Aspect ratio from bounding rectangle
        rect = cv2.boundingRect(contour)
        aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0
        
        # Extent (area vs bounding rectangle area)
        rect_area = rect[2] * rect[3]
        extent = area / rect_area if rect_area > 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Classification logic
        confidence = 0.0
        
        # Circle detection
        if circularity > 0.7 and solidity > 0.8 and 0.8 < aspect_ratio < 1.2:
            return "circle", min(0.95, circularity)
            
        # Square/Rectangle detection
        if vertices == 4 and solidity > 0.85:
            if 0.8 < aspect_ratio < 1.2:
                return "square", min(0.9, solidity)
            else:
                return "rectangle", min(0.85, solidity)
                
        # Triangle detection
        if vertices == 3 and solidity > 0.8:
            return "triangle", min(0.85, solidity)
            
        # Diamond detection (rotated square)
        if vertices == 4 and 0.5 < extent < 0.8 and solidity > 0.7:
            return "diamond", min(0.8, solidity)
            
        # Cross/Plus detection (low solidity, specific extent)
        if solidity < 0.6 and 0.3 < extent < 0.7:
            return "cross", min(0.7, 1 - solidity)
            
        # Star detection (many vertices, medium solidity)
        if vertices > 6 and 0.5 < solidity < 0.8:
            return "star", min(0.75, solidity)
            
        # Default to generic marker
        return "marker", 0.5

    def generate_masks_by_shape_detection(self, image: np.ndarray, 
                                        bbox_list: Optional[List[List]] = None) -> Dict[str, List[Tuple[np.ndarray, str, float]]]:
        """
        Generate masks using shape-specific detection methods.
        
        Args:
            image: Input image (BGR format)
            bbox_list: Optional list of bounding boxes to constrain search
            
        Returns:
            Dictionary mapping detection methods to lists of (mask, shape, confidence) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = {}
        
        # Method 1: Contour-based shape classification
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                
                # Filter by bbox if provided
                if bbox_list:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        inside_bbox = False
                        for bbox in bbox_list:
                            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            if x <= cx <= x + w and y <= cy <= y + h:
                                inside_bbox = True
                                break
                                
                        if not inside_bbox:
                            continue
                
                # Classify shape
                shape, confidence = self.classify_contour_shape(contour)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    contour_results.append((mask, shape, confidence))
        
        results['contour_classification'] = contour_results
        
        # Method 2: Template matching for specific shapes
        template_results = []
        templates = self.create_marker_templates(20)  # 20x20 templates
        
        for shape_name, template in templates.items():
            # Multi-scale template matching
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                    
                # Template matching
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.6)  # Threshold for template matching
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    w, h = scaled_template.shape[1], scaled_template.shape[0]
                    
                    # Check if within bbox constraints
                    if bbox_list:
                        cx, cy = x + w//2, y + h//2
                        inside_bbox = False
                        for bbox in bbox_list:
                            bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                                inside_bbox = True
                                break
                        if not inside_bbox:
                            continue
                    
                    # Create mask
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    mask_template = cv2.resize(template, (w, h))
                    mask[y:y+h, x:x+w] = mask_template
                    
                    confidence = result[y, x]
                    template_results.append((mask, shape_name, confidence))
        
        results['template_matching'] = template_results
        
        # Method 3: Specialized circle detection (existing HoughCircles)
        circle_results = []
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=self.circle_min_dist,
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=self.circle_min_radius,
            maxRadius=self.circle_max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Check bbox constraints
                if bbox_list:
                    inside_bbox = False
                    for bbox in bbox_list:
                        bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        if bx <= x <= bx + bw and by <= y <= by + bh:
                            inside_bbox = True
                            break
                    if not inside_bbox:
                        continue
                
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                circle_results.append((mask, "circle", 0.9))  # High confidence for HoughCircles
        
        results['hough_circles'] = circle_results
        
        # Log results
        total_detections = sum(len(method_results) for method_results in results.values())
        logger.info(f"Shape detection found {total_detections} potential data points:")
        for method, method_results in results.items():
            shapes = {}
            for _, shape, conf in method_results:
                shapes[shape] = shapes.get(shape, 0) + 1
            logger.info(f"  {method}: {shapes}")
        
        return results

    def detect_data_point_colors(self, image: np.ndarray, bbox_list: List[List]) -> List[Tuple[int, int, int]]:
        """
        Analyze data point bboxes to extract dominant colors for thresholding.
        
        Args:
            image: Input image (BGR format)
            bbox_list: List of bounding boxes [x, y, w, h] containing data points
            
        Returns:
            List of dominant BGR colors found in the bboxes
        """
        colors = []
        
        for bbox in bbox_list:
            x, y, w, h = bbox
            # Convert float coordinates to integers for array slicing
            x, y, w, h = int(x), int(y), int(w), int(h)
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
                
            # Convert to RGB for easier color analysis
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Reshape to get all pixels
            pixels = roi_rgb.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            
            try:
                # Find 3 dominant colors
                kmeans = KMeans(n_clusters=min(3, len(pixels)), random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get the most frequent color (largest cluster)
                labels = kmeans.labels_
                unique, counts = np.unique(labels, return_counts=True)
                dominant_idx = unique[np.argmax(counts)]
                dominant_color = kmeans.cluster_centers_[dominant_idx]
                
                # Convert back to BGR
                bgr_color = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))
                colors.append(bgr_color)
                
            except Exception as e:
                logger.warning(f"Could not extract color from bbox {bbox}: {e}")
                
        # Remove duplicate colors (within tolerance)
        unique_colors = []
        for color in colors:
            is_unique = True
            for existing in unique_colors:
                if all(abs(c1 - c2) < self.color_tolerance for c1, c2 in zip(color, existing)):
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(color)
                
        logger.info(f"Detected {len(unique_colors)} unique data point colors: {unique_colors}")
        return unique_colors
        
    def generate_masks_by_color(self, image: np.ndarray, target_colors: List[Tuple[int, int, int]]) -> List[np.ndarray]:
        """
        Generate masks by color thresholding.
        
        Args:
            image: Input image (BGR format)
            target_colors: List of BGR colors to threshold
            
        Returns:
            List of binary masks for each detected data point
        """
        masks = []
        
        for color in target_colors:
            # Create color range
            lower = np.array([max(0, c - self.color_tolerance) for c in color])
            upper = np.array([min(255, c + self.color_tolerance) for c in color])
            
            # Create mask
            mask = cv2.inRange(image, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area <= area <= self.max_contour_area:
                    # Create individual mask for this contour
                    individual_mask = np.zeros_like(mask)
                    cv2.fillPoly(individual_mask, [contour], 255)
                    masks.append(individual_mask)
                    
        logger.info(f"Generated {len(masks)} masks using color thresholding")
        return masks
        
    def generate_masks_by_circles(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate masks using HoughCircles detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of binary masks for detected circles
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=self.circle_min_dist,
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=self.circle_min_radius,
            maxRadius=self.circle_max_radius
        )
        
        masks = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                masks.append(mask)
                
        logger.info(f"Generated {len(masks)} masks using circle detection")
        return masks
        
    def generate_masks_by_contours(self, image: np.ndarray, bbox_list: Optional[List[List]] = None) -> List[np.ndarray]:
        """
        Generate masks using general contour detection.
        
        Args:
            image: Input image (BGR format)
            bbox_list: Optional list of bounding boxes to constrain search
            
        Returns:
            List of binary masks for detected contours
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                
                # If bboxes provided, check if contour is within any bbox
                if bbox_list:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        inside_bbox = False
                        for bbox in bbox_list:
                            x, y, w, h = bbox
                            # Convert float coordinates to integers
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            if x <= cx <= x + w and y <= cy <= y + h:
                                inside_bbox = True
                                break
                                
                        if not inside_bbox:
                            continue
                
                # Create mask for this contour
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                masks.append(mask)
                
        logger.info(f"Generated {len(masks)} masks using contour detection")
        return masks
        
    def generate_masks(self, 
                      image_path: Union[str, Path], 
                      bbox_list: Optional[List[List]] = None,
                      methods: List[str] = ['shapes', 'color', 'circles', 'contours']) -> Dict[str, List[np.ndarray]]:
        """
        Generate masks using multiple methods and return all results.
        
        Args:
            image_path: Path to the input image
            bbox_list: Optional list of data point bounding boxes [x, y, w, h]
            methods: List of methods to use ('shapes', 'color', 'circles', 'contours')
            
        Returns:
            Dictionary mapping method names to lists of masks
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        results = {}
        
        # Enhanced shape detection (NEW)
        if 'shapes' in methods:
            shape_results = self.generate_masks_by_shape_detection(image, bbox_list)
            # Flatten shape detection results into simple mask lists
            all_shape_masks = []
            for method_name, method_results in shape_results.items():
                all_shape_masks.extend([mask for mask, shape, conf in method_results])
            results['shapes'] = all_shape_masks
        
        if 'color' in methods and bbox_list:
            colors = self.detect_data_point_colors(image, bbox_list)
            if colors:
                results['color'] = self.generate_masks_by_color(image, colors)
            else:
                results['color'] = []
                
        if 'circles' in methods:
            results['circles'] = self.generate_masks_by_circles(image)
            
        if 'contours' in methods:
            results['contours'] = self.generate_masks_by_contours(image, bbox_list)
            
        total_masks = sum(len(masks) for masks in results.values())
        logger.info(f"Generated {total_masks} total masks across {len(results)} methods")
        
        return results
        
    def filter_masks_by_bbox(self, masks: List[np.ndarray], bbox_list: List[List], 
                            min_overlap: float = 0.5) -> List[Tuple[np.ndarray, int]]:
        """
        Filter masks to only include those that significantly overlap with provided bboxes.
        
        Args:
            masks: List of binary masks
            bbox_list: List of bounding boxes [x, y, w, h]
            min_overlap: Minimum IoU overlap required
            
        Returns:
            List of (mask, bbox_index) tuples for masks that overlap with bboxes
        """
        filtered = []
        
        for mask in masks:
            best_overlap = 0
            best_bbox_idx = -1
            
            # Get mask bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            mask_bbox = cv2.boundingRect(contours[0])
            mx, my, mw, mh = mask_bbox
            
            for i, bbox in enumerate(bbox_list):
                x, y, w, h = bbox
                # Convert float coordinates to integers
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Calculate IoU
                x1 = max(mx, x)
                y1 = max(my, y)
                x2 = min(mx + mw, x + w)
                y2 = min(my + mh, y + h)
                
                if x2 <= x1 or y2 <= y1:
                    overlap = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = mw * mh + w * h - intersection
                    overlap = intersection / union if union > 0 else 0
                    
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_bbox_idx = i
                    
            if best_overlap >= min_overlap:
                filtered.append((mask, best_bbox_idx))
                
        logger.info(f"Filtered {len(filtered)} masks with >{min_overlap} bbox overlap from {len(masks)} total")
        return filtered 