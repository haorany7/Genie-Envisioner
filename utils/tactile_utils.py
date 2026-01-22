import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay

def find_tactile_dots(img_gray, threshold=65, min_area=5, max_area=200, margin=15, num_dots=63):
    """
    Find marker dots in a GelSight gray image and return a fixed number of dots.
    """
    # Thresholding to extract dark dots
    _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Apply hard margin
    if margin > 0:
        h, w = thresh.shape
        thresh[:margin, :] = 0
        thresh[h-margin:, :] = 0
        thresh[:, :margin] = 0
        thresh[:, w-margin:] = 0
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                dots.append((M["m10"]/M["m00"], M["m01"]/M["m00"]))
    
    if not dots:
        return np.zeros((num_dots, 2))

    dots = np.array(dots)
    # Sort dots by distance to center to ensure consistent ordering for the MLP
    h, w = img_gray.shape
    center = np.array([w/2, h/2])
    dist_sq = np.sum((dots - center)**2, axis=1)
    sorted_indices = np.argsort(dist_sq)
    dots = dots[sorted_indices]

    # Pad or truncate to fixed number of dots
    if len(dots) >= num_dots:
        return dots[:num_dots]
    else:
        padding = np.zeros((num_dots - len(dots), 2))
        return np.concatenate([dots, padding], axis=0)

def calculate_tri_areas(pts, simplices):
    """
    Calculate areas of triangles formed by dots.
    
    Args:
        pts: Dot positions [N, 2].
        simplices: Indices of points forming triangles [M, 3].
        
    Returns:
        np.ndarray: Areas of M triangles.
    """
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    # Cross product formula for area
    return 0.5 * np.abs(a[:,0]*(b[:,1]-c[:,1]) + b[:,0]*(c[:,1]-a[:,1]) + c[:,0]*(a[:,1]-b[:,1]))

def compute_force_xyz(ref_dots, curr_dots_raw, tri_simplices, ref_areas, match_threshold_sq=144):
    """
    Compute 3D force field from reference and current markers.
    
    Args:
        ref_dots: Reference dot positions [N, 2].
        curr_dots_raw: Unmatched dot positions from current frame [K, 2].
        tri_simplices: Delaunay simplices from reference dots [M, 3].
        ref_areas: Reference triangle areas [M].
        match_threshold_sq: Max squared distance for marker matching.
        
    Returns:
        force_xy: [N, 2] Horizontal displacement (shear).
        force_z: [N] Vertical pressure (based on area change).
        matched_dots: [N, 2] The matched current dot positions.
    """
    # 1. Marker Tracking (Nearest Neighbor)
    matched_dots = np.zeros_like(ref_dots)
    for i, r_p in enumerate(ref_dots):
        if len(curr_dots_raw) > 0:
            dists = np.sum((curr_dots_raw - r_p)**2, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < match_threshold_sq:
                matched_dots[i] = curr_dots_raw[idx]
            else:
                matched_dots[i] = r_p # Fallback to original position if lost
        else:
            matched_dots[i] = r_p

    # 2. XY Force (Displacement vector)
    force_xy = matched_dots - ref_dots
    
    # 3. Z Force (Normal pressure via area change)
    curr_areas = calculate_tri_areas(matched_dots, tri_simplices)
    # Area decrease (negative change) corresponds to compression (positive Z force)
    # Area increase (positive change) corresponds to release/tension
    force_z_tri = (curr_areas - ref_areas)
    
    # Redistribute triangle area change to vertices (dots)
    force_z = np.zeros(len(ref_dots))
    for i, simplex in enumerate(tri_simplices):
        force_z[simplex] += force_z_tri[i] / 3.0
        
    return force_xy, force_z, matched_dots

class TactileForceExtractor:
    """
    Helper class to maintain state for tactile force extraction over a sequence.
    """
    def __init__(self, ref_frame_gray, margin=15, num_dots=63):
        self.num_dots = num_dots
        self.ref_dots = find_tactile_dots(ref_frame_gray, margin=margin, num_dots=num_dots)
        
        # Valid dots are those that are not padded (not [0, 0])
        valid_mask = np.any(self.ref_dots != 0, axis=1)
        valid_dots = self.ref_dots[valid_mask]
        
        if len(valid_dots) < 4:
            # Fallback: if not enough dots, Delaunay will fail. Use zero force.
            self.tri_simplices = None
            self.ref_areas = None
        else:
            self.tri = Delaunay(valid_dots)
            self.tri_simplices = self.tri.simplices
            self.ref_areas = calculate_tri_areas(valid_dots, self.tri_simplices)
            self.valid_indices = np.where(valid_mask)[0]
        
    def process_frame(self, curr_frame_gray):
        # We don't apply margin to current frames to avoid losing matched dots at edges
        curr_dots_raw = find_tactile_dots(curr_frame_gray, margin=0, num_dots=200) # Get more candidates candidate
        
        if self.tri_simplices is None:
            # Return zero force if initialization failed
            return self.ref_dots, np.zeros_like(self.ref_dots), np.zeros(len(self.ref_dots))

        force_xy, force_z, matched_dots = compute_force_xyz(
            self.ref_dots, 
            curr_dots_raw, 
            self.tri_simplices, 
            self.ref_areas
        )
        return self.ref_dots, force_xy, force_z

def extract_force_field_from_video(video_tensor, num_dots=63, ref_gray=None):
    """
    video_tensor: [T, H, W] grayscale or [T, C, H, W] (C in {1, 3})
    Returns force_field: [T, num_dots * 3]
    """
    if len(video_tensor.shape) == 4:
        # Expect [T, C, H, W]
        if video_tensor.size(1) == 3:
            # Use green channel for higher marker contrast
            video_gray = (video_tensor[:, 1].cpu().numpy() * 255).astype(np.uint8)
        else:
            # Single-channel
            video_gray = (video_tensor[:, 0].cpu().numpy() * 255).astype(np.uint8)
    else:
        # [T, H, W] already grayscale
        video_gray = (video_tensor.cpu().numpy() * 255).astype(np.uint8)

    if ref_gray is None:
        print("[tactile_utils] Warning: ref_gray is None; falling back to first frame of video_gray sampling video chunk for force field reference.")
        ref_gray = video_gray[0]
    extractor = TactileForceExtractor(ref_gray, num_dots=num_dots)

    T = len(video_gray)
    force_field = np.zeros((T, num_dots, 3))

    for t in range(T):
        _, f_xy, f_z = extractor.process_frame(video_gray[t])
        force_field[t, :, :2] = f_xy
        force_field[t, :, 2] = f_z

    return torch.from_numpy(force_field.reshape(T, -1)).float()
