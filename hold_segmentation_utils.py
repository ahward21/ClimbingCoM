import cv2 as cv
import numpy as np
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Polygon
from scipy.stats import linregress

# Parameters adapted from the original repo
STROKE_COLOR = (0, 255, 0)
STROKE_THICKNESS = 5

def dist(p1, p2):
    """Return the Euclidean distance between p1 and p2."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1/2)

def contour_to_list(c):
    """Convert a contour to a list of (x, y) tuples."""
    l = []
    for j in c:
        l.append(j.tolist()[0])
    return l

def filter_straight_contours(contours, max_avg_error=5):
    """
    Filter out contours that are too close to lines.
    Useful for removing tape or wall edges.
    """
    contours = list(contours)

    def error_function(a, b, c):
        error = 0
        points = contour_to_list(c)
        if not points: return 0
        for x, y in points:
            error += (a * x + b - y) ** 2
        return error / len(c)

    to_remove = []
    for i, c in enumerate(contours):
        points = np.asarray(contour_to_list(c))
        if len(points) < 2:
            to_remove.append(i)
            continue

        x = points[:,0]
        y = points[:,1]

        # Check if vertical (slope potential infinite)
        if np.std(x) < 1e-5: 
             # Almost vertical line
             to_remove.append(i)
             continue

        try:
            result = linregress(x, y)
            a, b = result.slope, result.intercept
            error = error_function(a, b, c)
            if error < max_avg_error:
                to_remove.append(i)
        except:
             # If regression fails, keep it or remove? safer to remove if ambiguous straight lines
             pass

    for r in reversed(to_remove):
        del contours[r]

    return contours

def filter_size_contours(contours, min_points=3, min_bb_area=125):
    """
    Filter out contours based on the number of their points and their bounding box area.
    """
    contours = list(contours)

    to_remove = []
    for i, c in enumerate(contours):
        if len(c) < min_points:
            to_remove.append(i)
            continue

        points = [j.tolist()[0] for j in c]
        if len(points) < 3: # Need at least 3 points for a polygon
             to_remove.append(i)
             continue
             
        try:
            p = Polygon(points)
            if not p.is_valid:
                 # Try cleaning it? or just skip
                 to_remove.append(i)
                 continue

            xl, yl, xh, yh = p.bounds
            w = abs(xl - xh)
            h = abs(yl - yh)

            if w * h < min_bb_area:
                to_remove.append(i)
        except Exception:
            to_remove.append(i)

    for r in reversed(to_remove):
        del contours[r]

    return contours

def gaussian_blur(img, size=13):
    return cv.GaussianBlur(img, (size, size), 0)

def canny(img, parameters=(20, 25)):
    return cv.Canny(img, *parameters)

def find_contours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def simplify_contours(contours, epsilon=0.005):
    simplified = []
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon * peri, True)
        simplified.append(approx)
    return simplified

def threshold(img, start=0, end=255):
    _, t = cv.threshold(img, start, end, cv.THRESH_BINARY)
    return t

def detect_blobs(img):
    """OpenCV simple blob detection."""
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100 
    params.maxArea = 5000 # drastically reduced from 1M to avoid giant blobs

    params.minThreshold = 1
    params.maxThreshold = 200
    params.thresholdStep = 10

    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv.SimpleBlobDetector_create(params)
    return detector.detect(img)

def get_nearby_contours(point, contours, distance):
    def is_close(p, c):
        for pc in contour_to_list(c):
            if dist(p, pc) < distance:
                return True
        return False

    close = []
    for c in contours:
        if is_close(point, c):
            close.append(c)
    return close

def get_closest_contour(point, contours):
    closest = None
    closest_distance = float('inf')
    for c in contours:
        for pc in contour_to_list(c):
            d = dist(point, pc)
            if d < closest_distance:
                closest_distance = d
                closest = c
    return closest

def point_to_segment_distance(a, b, p):
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)

    if np.array_equal(a, b):
        return np.linalg.norm(a - p)

    d = np.divide(b - a, np.linalg.norm(b - a))
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    h = np.maximum.reduce([s, t, 0])
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

def point_to_contour_distance(point, contour):
    cl = contour_to_list(contour)
    if not cl: return float('inf')

    min_d = float('inf')
    for i in range(len(cl) - 1):
        d = point_to_segment_distance(cl[i], cl[i + 1], point)
        if d < min_d:
            min_d = d
    # Close the loop
    if len(cl) > 1:
        d = point_to_segment_distance(cl[-1], cl[0], point)
        if d < min_d:
             min_d = d
    return min_d

def squared_contour_error(contours_from, contour_to):
    point_count = 0
    contour_error = 0
    for c in contours_from:
        point_count += len(c)
        for pc in contour_to_list(c):
            contour_error += point_to_contour_distance(pc, contour_to)
    
    if point_count == 0: return float('inf')
    return contour_error / point_count

def detect_holds_cv(img, keypoints, contours, threshold_step=10):
    """
    Detect holds by combining blob and edge detection.
    Returns a list of contours that represent the holds.
    """
    blur = gaussian_blur(img)
    
    # Pre-compute edge contours at various thresholds
    thresholds = {}
    # Reduced range/step for performance optimization if needed, but original used 0-255 step 5
    # Step 10 is faster
    for i in range(0, 255, threshold_step):
        t = threshold(blur, i, 255)
        thresholds[i] = []
        
        # Process each channel? Or just grayscale? 
        # The original code loops over channels of t: t[:,:,0] etc
        # If t is result of cv.threshold on a color, it behaves differently. 
        # Let's assume img is BGR.
        
        # Check if image is grayscale or color
        if len(t.shape) == 3:
             channels = [t[:,:,0], t[:,:,1], t[:,:,2]]
        else:
             channels = [t]
             
        for c_t in channels:
            t_edges = canny(c_t)
            t_contours = find_contours(t_edges)
            # Filter?
            t_contours = filter_size_contours(t_contours, min_bb_area=50) # Added basic filtering
            t_contours = simplify_contours(t_contours)
            thresholds[i].append(t_contours)

    final_contours = []
    
    for k in keypoints:
        # Initial search radius based on blob size
        # Limit the search radius to avoid matching far-away contours (giant blobs)
        search_radius = min(max(k.size / 1.5, 10.0), 100.0)
        
        nearby_contours = get_nearby_contours(k.pt, contours, search_radius) # contours here are the 'base' contours passing in? 
        # Actually in the original code 'contours' arg seems to be edges from the original image?
        # Let's look at how it was called in demo.ipynb?
        # In demo:
        # edges = utils.canny(image)
        # contours = utils.find_contours(edges)
        # contours = utils.filter_straight_contours(contours)
        # contours = utils.filter_size_contours(contours)
        # keypoints = utils.detect_blobs(image)
        # holds = utils.detect_holds(image, keypoints, contours)
        
        # So 'contours' passed in are the canny edges of the raw image.
        
        if len(nearby_contours) == 0:
            # If no edges found near the blob, maybe just use the blob as a circle? 
            # for now, skip
            continue

        best_contour = None
        best_contour_error = float('inf')

        # Find best matching contour across thresholds
        for i in range(0, 255, threshold_step):
            for t_contours_list in thresholds[i]: # stored as list of lists (per channel)? No, I appended lists.
                 # Actually my implementation above adds t_contours (list) to thresholds[i].
                 # So thresholds[i] is a list of lists of contours.
                 
                 # Flatten this level? or iterate
                 flat_t_contours = t_contours_list # actually it is a list of contours
                 
                 # Optimization: find closest contour in this threshold set
                 closest_t_contour = get_closest_contour(k.pt, flat_t_contours)
                 
                 if closest_t_contour is None:
                     continue
                     
                 # original used squared_contour_error(nearby_contours, closest_t_contour)
                 # nearby_contours are from the raw image edges.
                 err = squared_contour_error(nearby_contours, closest_t_contour)
                 
                 if err < best_contour_error:
                     best_contour_error = err
                     best_contour = closest_t_contour

        if best_contour is not None:
            final_contours.append(best_contour)

    return final_contours
