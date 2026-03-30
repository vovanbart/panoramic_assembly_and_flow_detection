#!/usr/bin/env python3

import sys
import os
import re
import json
import time
import gc
import logging
from collections import deque
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal.windows import tukey
import torch
import torch.nn as nn
from ultralytics import YOLO as _YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

GRID_ROWS = 10
GRID_COLS = 10
SIFT_FEATURES = 3000
LOWE_RATIO = 0.75
RANSAC_THRESH = 5.0
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.2
TEMPLATE_SIZE = 1800
SEARCH_RADIUS = 55
JPEG_QUALITY = 95
NUM_WORKERS = max(1, min(cpu_count(), 14))
MIN_DISPLACEMENT = 400
GRID_PERIOD = 117

YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
SHIFT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_model.pt")
YOLO_CROP_SIZE = 1024
YOLO_CROP_OVERLAP = 200
YOLO_CLAHE_CLIP = 12.0
YOLO_CONF = 0.25
YOLO_NMS_IOU = 0.3

def sobel_mag(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)

def subpix(result, my, mx):
    rh, rw = result.shape
    sub_x = float(mx)
    sub_y = float(my)

    if 1 <= mx < rw - 1:
        left = float(result[my, mx - 1])
        center = float(result[my, mx])
        right = float(result[my, mx + 1])
        denom = 2.0 * (2 * center - left - right)
        if abs(denom) > 1e-6:
            sub_x = mx + (left - right) / denom

    if 1 <= my < rh - 1:
        top = float(result[my - 1, mx])
        center = float(result[my, mx])
        bottom = float(result[my + 1, mx])
        denom = 2.0 * (2 * center - top - bottom)
        if abs(denom) > 1e-6:
            sub_y = my + (top - bottom) / denom

    return sub_x, sub_y

def ncc(a, b):
    a_z = a - a.mean()
    b_z = b - b.mean()
    denom = np.sqrt(np.sum(a_z ** 2) * np.sum(b_z ** 2))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(a_z * b_z) / denom)

def get_ordered_files(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    def tc_number(fname):
        m = re.search(r'TC(\d+)', fname, re.IGNORECASE)
        return int(m.group(1)) if m else 0
    files.sort(key=tc_number)
    if len(files) != GRID_ROWS * GRID_COLS:
        log.warning(f"Expected {GRID_ROWS * GRID_COLS} files, got {len(files)}")

    grid = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    for idx, fname in enumerate(files):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        if row % 2 == 1:
            col = GRID_COLS - 1 - col
        grid[row][col] = os.path.join(directory, fname)

    log.info(f"Grid: snake pattern, {len(files)} files")
    return grid

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp, des = sift.detectAndCompute(img_eq, None)
    kp_data = [(p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
               for p in kp]
    return image_path, kp_data, des

def deserialize_keypoints(kp_data):
    if kp_data is None:
        return []
    return [cv2.KeyPoint(x=d[0], y=d[1], size=d[2], angle=d[3],
                         response=d[4], octave=d[5], class_id=d[6])
            for d in kp_data]

def extract_all_features(grid):
    paths = [path for row in grid for path in row if path]
    log.info(f"Extracting SIFT features from {len(paths)} images ({NUM_WORKERS} workers)...")
    t0 = time.time()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(extract_features, paths)
    features = {}
    for path, kp_data, des in results:
        features[path] = (kp_data, des)
    log.info(f"Feature extraction done in {time.time() - t0:.1f}s")
    return features

def match_sift_pair(kp1_data, des1, kp2_data, des2,
                    min_inliers=MIN_INLIERS, min_inlier_ratio=MIN_INLIER_RATIO):
    if des1 is None or des2 is None:
        return None
    if len(des1) < min_inliers or len(des2) < min_inliers:
        return None
    kp1 = deserialize_keypoints(kp1_data)
    kp2 = deserialize_keypoints(kp2_data)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < LOWE_RATIO * n.distance:
                good.append(m)

    if len(good) < MIN_INLIERS:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=RANSAC_THRESH)
    if M is None or inliers is None:
        return None

    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / len(good)

    if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return None

    dx = M[0, 2]
    dy = M[1, 2]

    displacement = np.sqrt(dx * dx + dy * dy)
    if displacement < MIN_DISPLACEMENT:
        return None

    angle = np.arctan2(M[1, 0], M[0, 0])
    angle_deg = np.degrees(angle)

    return (dx, dy, n_inliers, angle_deg)

def match_template_constrained(path1, path2, expected_dx, expected_dy):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img1_eq = sobel_mag(clahe.apply(img1)).astype(np.uint8)
    img2_eq = sobel_mag(clahe.apply(img2)).astype(np.uint8)

    if abs(expected_dx) > abs(expected_dy):
        max_overlap = w - abs(expected_dx)
    else:
        max_overlap = h - abs(expected_dy)
    max_tmpl = max(200, int(max_overlap) - 150)
    tmpl_h = min(TEMPLATE_SIZE, h - 100, max_tmpl)
    tmpl_w = min(TEMPLATE_SIZE, w - 100, max_tmpl)

    if abs(expected_dx) > abs(expected_dy):
        if expected_dx < 0:
            tx = w - tmpl_w - 50
        else:
            tx = 50
        ty = max(50, (h - tmpl_h) // 2)
    else:
        if expected_dy < 0:
            ty = h - tmpl_h - 50
        else:
            ty = 50
        tx = max(50, (w - tmpl_w) // 2)

    template = img1_eq[ty:ty + tmpl_h, tx:tx + tmpl_w]

    search_cx = int(tx + expected_dx)
    search_cy = int(ty + expected_dy)

    sr = SEARCH_RADIUS
    s_x1 = max(0, search_cx - sr)
    s_y1 = max(0, search_cy - sr)
    s_x2 = min(w, search_cx + tmpl_w + sr)
    s_y2 = min(h, search_cy + tmpl_h + sr)

    if s_x2 - s_x1 < tmpl_w or s_y2 - s_y1 < tmpl_h:
        return None

    search_region = img2_eq[s_y1:s_y2, s_x1:s_x2]

    if search_region.shape[0] < tmpl_h or search_region.shape[1] < tmpl_w:
        return None

    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.3:
        return None

    my, mx = max_loc[1], max_loc[0]
    sub_x, sub_y = subpix(result, my, mx)

    found_x = s_x1 + sub_x
    found_y = s_y1 + sub_y

    dx = found_x - tx
    dy = found_y - ty

    confidence = max(1, int(max_val * 20))
    return (dx, dy, confidence, 0.0)

def match_pcm_multipeak(path1, path2, expected_dx, expected_dy):
    NUM_PEAKS = 5

    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img1_eq = clahe.apply(img1)
    img2_eq = clahe.apply(img2)

    edx, edy = int(round(expected_dx)), int(round(expected_dy))

    ox1 = max(0, -edx)
    ox2 = min(w, w - edx)
    oy1 = max(0, -edy)
    oy2 = min(h, h - edy)

    ow, oh = ox2 - ox1, oy2 - oy1
    if ow < 400 or oh < 400:
        return None

    margin = 100
    ox1 += margin
    ox2 -= margin
    oy1 += margin
    oy2 -= margin
    ow, oh = ox2 - ox1, oy2 - oy1
    if ow < 300 or oh < 300:
        return None

    max_dim = 1000
    if ow > max_dim:
        trim = (ow - max_dim) // 2
        ox1 += trim
        ox2 -= trim
        ow = ox2 - ox1
    if oh > max_dim:
        trim = (oh - max_dim) // 2
        oy1 += trim
        oy2 -= trim
        oh = oy2 - oy1

    region1 = img1_eq[oy1:oy2, ox1:ox2].astype(np.float32)

    r2_x1 = ox1 + edx
    r2_y1 = oy1 + edy
    if r2_x1 < 0 or r2_y1 < 0 or r2_x1 + ow > w or r2_y1 + oh > h:
        return None

    region2 = img2_eq[r2_y1:r2_y1 + oh, r2_x1:r2_x1 + ow].astype(np.float32)

    if region1.shape != region2.shape:
        return None

    hann = cv2.createHanningWindow((ow, oh), cv2.CV_32F)
    r1w = region1 * hann
    r2w = region2 * hann

    f1 = np.fft.fft2(r1w)
    f2 = np.fft.fft2(r2w)
    cross = f1 * np.conj(f2)
    cross_norm = cross / (np.abs(cross) + 1e-10)
    pcm = np.fft.ifft2(cross_norm).real

    sr = GRID_PERIOD
    pcm_h, pcm_w = pcm.shape

    valid = np.zeros((pcm_h, pcm_w), dtype=bool)
    valid[:sr, :sr] = True
    valid[:sr, -sr:] = True
    valid[-sr:, :sr] = True
    valid[-sr:, -sr:] = True

    pcm_masked = pcm.copy()
    pcm_masked[~valid] = 0

    candidates = []
    for _ in range(NUM_PEAKS):
        idx = np.argmax(pcm_masked)
        best_y, best_x = np.unravel_index(idx, pcm_masked.shape)
        best_val = pcm_masked[best_y, best_x]

        if best_val <= 0:
            break

        shift_x = best_x if best_x < pcm_w // 2 else best_x - pcm_w
        shift_y = best_y if best_y < pcm_h // 2 else best_y - pcm_h

        candidates.append((shift_x, shift_y, best_val))

        sup_r = 8
        for sy in range(best_y - sup_r, best_y + sup_r + 1):
            for sx in range(best_x - sup_r, best_x + sup_r + 1):
                pcm_masked[sy % pcm_h, sx % pcm_w] = 0

    if not candidates:
        return None

    best_ncc = -1
    best_dx, best_dy = 0, 0
    patch_size = 400

    cy1, cx1 = oh // 2, ow // 2
    ps = patch_size // 2
    p_y1 = max(0, cy1 - ps)
    p_y2 = min(oh, cy1 + ps)
    p_x1 = max(0, cx1 - ps)
    p_x2 = min(ow, cx1 + ps)
    patch1 = region1[p_y1:p_y2, p_x1:p_x2]

    for shift_x, shift_y, pcm_val in candidates:
        r2_py1 = p_y1 - shift_y
        r2_py2 = p_y2 - shift_y
        r2_px1 = p_x1 - shift_x
        r2_px2 = p_x2 - shift_x

        if r2_py1 < 0 or r2_px1 < 0 or r2_py2 > oh or r2_px2 > ow:
            continue

        patch2 = region2[r2_py1:r2_py2, r2_px1:r2_px2]

        if patch1.shape != patch2.shape:
            continue

        score = ncc(patch1, patch2)

        if score > best_ncc:
            best_ncc = score
            best_dx = expected_dx + shift_x
            best_dy = expected_dy + shift_y

    if best_ncc < 0.3:
        return None

    dev = np.sqrt((best_dx - expected_dx)**2 + (best_dy - expected_dy)**2)
    if dev > GRID_PERIOD:
        return None

    confidence = max(1, int(best_ncc * 30))
    return (best_dx, best_dy, confidence, 0.0)

def do_phase_match(args):
    (r1, c1), (r2, c2), direction, path1, path2, exp_dx, exp_dy = args
    result = match_pcm_multipeak(path1, path2, exp_dx, exp_dy)
    return (r1, c1), (r2, c2), direction, result

def get_neighbor_pairs(grid):
    pairs = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if c + 1 < GRID_COLS:
                pairs.append(((r, c), (r, c + 1), 'h'))
            if r + 1 < GRID_ROWS:
                pairs.append(((r, c), (r + 1, c), 'v'))
    return pairs

def do_sift_match(args):
    (r1, c1), (r2, c2), direction, feat1, feat2 = args
    kp1_data, des1 = feat1
    kp2_data, des2 = feat2
    result = match_sift_pair(kp1_data, des1, kp2_data, des2)
    return (r1, c1), (r2, c2), direction, result

def do_template_match(args):
    (r1, c1), (r2, c2), direction, path1, path2, exp_dx, exp_dy = args
    result = match_template_constrained(path1, path2, exp_dx, exp_dy)
    return (r1, c1), (r2, c2), direction, result

def tukey_phase_corr(path1, path2, alpha=0.5):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape
    wy = tukey(h, alpha=alpha).reshape(-1, 1)
    wx = tukey(w, alpha=alpha).reshape(1, -1)
    window = (wy * wx).astype(np.float64)
    f1 = np.fft.fft2(img1.astype(np.float64) * window)
    f2 = np.fft.fft2(img2.astype(np.float64) * window)
    cross = f1 * np.conj(f2)
    cross /= np.abs(cross) + 1e-8
    correlation = np.real(np.fft.ifft2(cross))
    y_idx = np.arange(-400, 401) % h
    correlation[np.ix_(y_idx, np.arange(min(401, w)))] = 0
    if w > 401:
        correlation[np.ix_(y_idx, np.arange(w - 400, w))] = 0
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    tx = peak_x if peak_x < w // 2 else peak_x - w
    ty = peak_y if peak_y < h // 2 else peak_y - h
    dx_sift = -tx
    dy_sift = -ty
    peak_val = float(correlation[peak_y, peak_x])
    return dx_sift, dy_sift, peak_val

def refine_periodic_h_step(grid, rough_h, rough_v, search_range=30):
    rough_dx, rough_dy = rough_h
    TMPL = 800
    pairs = []
    for r in range(1, 9):
        for c in range(1, 8):
            if c >= GRID_COLS - 1:
                continue
            if grid[r][c] and grid[r][c + 1]:
                pairs.append((grid[r][c], grid[r][c + 1]))

    if not pairs:
        return rough_h, rough_v
    log.info("Phase 1: Refining dx with NCC on gradient images...")
    dx_results = []
    for p1, p2 in pairs:
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        h, w = img1.shape
        grad1 = sobel_mag(img1)
        grad2 = sobel_mag(img2)
        overlap_start = int(-rough_dx)
        overlap_mid_x = (overlap_start + w) // 2
        overlap_mid_y = h // 2
        tx = max(0, overlap_mid_x - TMPL // 2)
        ty = max(0, overlap_mid_y - TMPL // 2)
        tx = min(tx, w - TMPL)
        ty = min(ty, h - TMPL)
        if tx < 0 or ty < 0:
            continue
        template = grad1[ty:ty + TMPL, tx:tx + TMPL]
        if template.shape[0] != TMPL or template.shape[1] != TMPL:
            continue

        expected_x = tx + int(round(rough_dx))
        expected_y = ty + int(round(rough_dy))

        sx = max(0, expected_x - search_range)
        sy = max(0, expected_y - search_range)
        ex = min(w, expected_x + TMPL + search_range)
        ey = min(h, expected_y + TMPL + search_range)

        if ex - sx < TMPL or ey - sy < TMPL:
            continue

        search_region = grad2[sy:ey, sx:ex]
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.05:
            continue

        found_x = sx + max_loc[0]
        pair_dx = found_x - tx
        dx_results.append(pair_dx)

    if dx_results:
        dx_arr = np.array(dx_results)
        refined_dx = float(np.median(dx_arr))
        log.info(f"    Median dx={refined_dx:.1f}, std={np.std(dx_arr):.1f}, n={len(dx_results)}/{len(pairs)}")
        if np.std(dx_arr) > GRID_PERIOD:
            refined_dx = rough_dx
    else:
        refined_dx = rough_dx

    log.info("  Phase 2: Refining dy with detrended row-profile NCC...")

    STRIP_W = 50
    SMOOTH_WIN = 350
    SEARCH_DY = 130

    test_dy = np.arange(-SEARCH_DY, SEARCH_DY + 1, 1)
    dy_scores = {dy: [] for dy in test_dy}

    idx_dx = int(round(refined_dx))

    for p1, p2 in pairs:
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        h, w = img1.shape
        overlap_w = w + idx_dx
        if overlap_w < 500:
            continue

        mid = overlap_w // 2

        x1_start = w - overlap_w + mid - STRIP_W // 2
        prof1 = np.mean(img1[:, x1_start:x1_start + STRIP_W].astype(np.float32), axis=1)
        x2_start = mid - STRIP_W // 2
        prof2 = np.mean(img2[:, x2_start:x2_start + STRIP_W].astype(np.float32), axis=1)

        prof1_dt = prof1 - uniform_filter1d(prof1, SMOOTH_WIN)
        prof2_dt = prof2 - uniform_filter1d(prof2, SMOOTH_WIN)

        for dy in test_dy:
            idy = int(round(dy))
            margin = 300
            if idy >= 0:
                p1_s = prof1_dt[margin:h - idy - margin]
                p2_s = prof2_dt[idy + margin:h - margin]
            else:
                p1_s = prof1_dt[-idy + margin:h - margin]
                p2_s = prof2_dt[margin:h + idy - margin]

            if len(p1_s) < 500 or len(p1_s) != len(p2_s):
                continue

            p1_z = p1_s - np.mean(p1_s)
            p2_z = p2_s - np.mean(p2_s)
            denom = np.sqrt(np.sum(p1_z ** 2) * np.sum(p2_z ** 2))
            if denom < 1e-10:
                continue
            score = float(np.sum(p1_z * p2_z) / denom)
            dy_scores[dy].append(score)

    dy_results = [(dy, np.mean(dy_scores[dy])) for dy in test_dy if dy_scores[dy]]

    if dy_results:
        best_dy_entry = max(dy_results, key=lambda x: x[1])
        refined_dy = float(best_dy_entry[0])
        log.info(f"    Best dy={refined_dy:.0f}, NCC={best_dy_entry[1]:.4f}")
    else:
        refined_dy = rough_dy

    refined_h = (refined_dx, refined_dy)
    log.info(f"  Refined H step: ({refined_h[0]:.1f}, {refined_h[1]:.1f})")

    log.info("  Phase 3: Independently refining V step...")

    v_pairs = []
    for r in range(1, 8):
        for c in range(1, 9):
            if c >= GRID_COLS:
                continue
            if grid[r][c] and grid[r + 1][c]:
                v_pairs.append((grid[r][c], grid[r + 1][c]))

    v_dx_init = rough_v[0]
    v_dy_init = rough_v[1]

    log.info("    Refining V dy with NCC on gradient images...")
    vdy_results = []
    TMPL_V = 800

    for p1, p2 in v_pairs:
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        h, w = img1.shape

        grad1 = sobel_mag(img1)
        grad2 = sobel_mag(img2)

        abs_vdy = abs(int(round(v_dy_init)))
        overlap_h = h - abs_vdy
        if overlap_h < TMPL_V + 2 * search_range:
            continue

        overlap_mid_y = abs_vdy + overlap_h // 2
        overlap_mid_x = w // 2

        tx = max(0, overlap_mid_x - TMPL_V // 2)
        ty = max(abs_vdy, overlap_mid_y - TMPL_V // 2)
        tx = min(tx, w - TMPL_V)
        ty = min(ty, h - TMPL_V)

        if tx < 0 or ty < 0:
            continue

        template = grad1[ty:ty + TMPL_V, tx:tx + TMPL_V]
        if template.shape[0] != TMPL_V or template.shape[1] != TMPL_V:
            continue

        expected_x = tx + int(round(v_dx_init))
        expected_y = ty + int(round(v_dy_init))

        sx = max(0, expected_x - search_range)
        sy = max(0, expected_y - search_range)
        ex = min(w, expected_x + TMPL_V + search_range)
        ey = min(h, expected_y + TMPL_V + search_range)

        if ex - sx < TMPL_V or ey - sy < TMPL_V:
            continue

        search_region = grad2[sy:ey, sx:ex]
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.05:
            continue

        found_y = sy + max_loc[1]
        pair_vdy = found_y - ty
        vdy_results.append(pair_vdy)

    h_derived_vdy = float(refined_h[0])

    if vdy_results:
        vdy_arr = np.array(vdy_results)
        ncc_vdy = float(np.median(vdy_arr))
        log.info(f"    V dy (NCC gradient): median={ncc_vdy:.1f}, "
                 f"std={np.std(vdy_arr):.1f}, n={len(vdy_results)}/{len(v_pairs)}")

        if abs(ncc_vdy - h_derived_vdy) < 15:
            refined_vdy = ncc_vdy
        elif np.std(vdy_arr) > GRID_PERIOD:
            refined_vdy = h_derived_vdy
        else:
            refined_vdy = h_derived_vdy
    else:
        refined_vdy = h_derived_vdy

    log.info("    Refining V dx with column-profile NCC...")
    test_vdx = np.arange(-SEARCH_DY, SEARCH_DY + 1, 1)
    vdx_scores = {dx: [] for dx in test_vdx}

    abs_vdy = abs(int(round(refined_vdy)))

    for p1, p2 in v_pairs:
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        h, w = img1.shape

        overlap_h = h - abs_vdy
        if overlap_h < 500:
            continue
        mid_y = overlap_h // 2

        for vdx in test_vdx:
            ivdx = int(round(vdx))
            if ivdx >= 0:
                x1s = 0
                x2s = ivdx
                ow = w - ivdx
            else:
                x1s = -ivdx
                x2s = 0
                ow = w + ivdx
            if ow < 500:
                continue

            margin = 200
            y1_center = abs_vdy + mid_y
            y2_center = mid_y
            sh = STRIP_W

            prof1 = np.mean(img1[y1_center - sh // 2:y1_center + sh // 2,
                            x1s:x1s + ow].astype(np.float32), axis=0)
            prof2 = np.mean(img2[y2_center - sh // 2:y2_center + sh // 2,
                            x2s:x2s + ow].astype(np.float32), axis=0)

            prof1_dt = prof1 - uniform_filter1d(prof1, SMOOTH_WIN)
            prof2_dt = prof2 - uniform_filter1d(prof2, SMOOTH_WIN)

            p1_s = prof1_dt[margin:-margin]
            p2_s = prof2_dt[margin:-margin]
            if len(p1_s) < 300 or len(p1_s) != len(p2_s):
                continue

            p1_z = p1_s - np.mean(p1_s)
            p2_z = p2_s - np.mean(p2_s)
            denom = np.sqrt(np.sum(p1_z ** 2) * np.sum(p2_z ** 2))
            if denom < 1e-10:
                continue
            score = float(np.sum(p1_z * p2_z) / denom)
            vdx_scores[vdx].append(score)

    vdx_results = [(dx, np.mean(vdx_scores[dx])) for dx in test_vdx if vdx_scores[dx]]
    if vdx_results:
        rot_vdx = -refined_h[1]
        peaks = []
        for i in range(1, len(vdx_results) - 1):
            if (vdx_results[i][1] > vdx_results[i - 1][1] and
                    vdx_results[i][1] > vdx_results[i + 1][1] and
                    vdx_results[i][1] > 0.01):
                peaks.append(vdx_results[i])

        if peaks:
            top_ncc = max(c[1] for c in peaks) * 0.5
            strong_peaks = [c for c in peaks if c[1] >= top_ncc]
            best_vdx_entry = min(strong_peaks, key=lambda x: abs(x[0] - rot_vdx))
        else:
            best_vdx_entry = max(vdx_results, key=lambda x: x[1])

        refined_vdx = float(best_vdx_entry[0])
        log.info(f"    V dx: best={refined_vdx:.0f}, NCC={best_vdx_entry[1]:.4f} "
                 f"(rotation estimate: {-refined_h[1]:.0f})")
    else:
        refined_vdx = v_dx_init

    refined_v = (refined_vdx, refined_vdy)
    log.info(f"  Refined V step: ({refined_v[0]:.1f}, {refined_v[1]:.1f})")

    gc.collect()

    return refined_h, refined_v

def refine_minor_axes(grid, h_step, v_step, scan_range=25):

    def overlap_diff_h(img1, img2, h_dx, h_dy):
        h, w = img1.shape
        abs_dx = abs(int(round(h_dx)))
        idy = int(round(h_dy))
        ov_w = w - abs_dx
        if ov_w < 300:
            return 1e9
        if idy >= 0:
            y1s, y2s, oh = 0, idy, h - idy
        else:
            y1s, y2s, oh = -idy, 0, h + idy
        if oh < 300:
            return 1e9
        mx = ov_w // 6
        my = oh // 6
        x1s = w - ov_w + mx
        x2s = mx
        ow = ov_w - 2 * mx
        y1_start = y1s + my
        y2_start = y2s + my
        oh_inner = oh - 2 * my
        if ow < 100 or oh_inner < 100:
            return 1e9
        p1 = img1[y1_start:y1_start + oh_inner, x1s:x1s + ow].astype(np.float32)
        p2 = img2[y2_start:y2_start + oh_inner, x2s:x2s + ow].astype(np.float32)
        g1y = cv2.Sobel(p1, cv2.CV_32F, 0, 1, ksize=3)
        g2y = cv2.Sobel(p2, cv2.CV_32F, 0, 1, ksize=3)
        edge_mask = (np.abs(g1y) > 5) | (np.abs(g2y) > 5)
        if np.sum(edge_mask) < 100:
            return 1e9
        diff = (g1y[edge_mask] - g2y[edge_mask]) ** 2
        return float(np.mean(diff))

    def overlap_diff_v(img1, img2, v_dx, v_dy):
        h, w = img1.shape
        abs_dy = abs(int(round(v_dy)))
        idx = int(round(v_dx))
        ov_h = h - abs_dy
        if ov_h < 300:
            return 1e9
        if idx >= 0:
            x1s, x2s, ov_w = 0, idx, w - idx
        else:
            x1s, x2s, ov_w = -idx, 0, w + idx
        if ov_w < 300:
            return 1e9
        mx = ov_w // 6
        my = ov_h // 6
        y1s = h - ov_h + my
        y2s = my
        oh = ov_h - 2 * my
        x1_start = x1s + mx
        x2_start = x2s + mx
        ow = ov_w - 2 * mx
        if ow < 100 or oh < 100:
            return 1e9
        p1 = img1[y1s:y1s + oh, x1_start:x1_start + ow].astype(np.float32)
        p2 = img2[y2s:y2s + oh, x2_start:x2_start + ow].astype(np.float32)
        g1x = cv2.Sobel(p1, cv2.CV_32F, 1, 0, ksize=3)
        g2x = cv2.Sobel(p2, cv2.CV_32F, 1, 0, ksize=3)
        edge_mask = (np.abs(g1x) > 5) | (np.abs(g2x) > 5)
        if np.sum(edge_mask) < 100:
            return 1e9
        diff = (g1x[edge_mask] - g2x[edge_mask]) ** 2
        return float(np.mean(diff))

    log.info("Refining minor axes with gradient overlap difference...")

    h_images = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            if grid[r][c] and grid[r][c + 1]:
                im1 = cv2.imread(grid[r][c], cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(grid[r][c + 1], cv2.IMREAD_GRAYSCALE)
                h_images.append((im1, im2))

    current_dy = h_step[1]
    dy_lo = int(round(current_dy)) - scan_range
    dy_hi = int(round(current_dy)) + scan_range

    log.info(f"  H dy scan: [{dy_lo}, {dy_hi}] around current {current_dy:.1f}")
    best_dy = int(round(current_dy))
    best_dy_score = 1e9

    for dy in range(dy_lo, dy_hi + 1):
        diffs = [overlap_diff_h(im1, im2, h_step[0], dy) for im1, im2 in h_images]
        avg = np.mean(diffs)
        if avg < best_dy_score:
            best_dy_score = avg
            best_dy = dy

    log.info(f"  H dy: {current_dy:.1f} -> {best_dy} (score={best_dy_score:.6f})")

    del h_images
    gc.collect()

    v_images = []
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS):
            if grid[r][c] and grid[r + 1][c]:
                im1 = cv2.imread(grid[r][c], cv2.IMREAD_GRAYSCALE)
                im2 = cv2.imread(grid[r + 1][c], cv2.IMREAD_GRAYSCALE)
                v_images.append((im1, im2))

    current_dx = v_step[0]
    dx_lo = int(round(current_dx)) - scan_range
    dx_hi = int(round(current_dx)) + scan_range

    log.info(f"  V dx scan: [{dx_lo}, {dx_hi}] around current {current_dx:.1f}")
    best_dx = int(round(current_dx))
    best_dx_score = 1e9

    for dx in range(dx_lo, dx_hi + 1):
        diffs = [overlap_diff_v(im1, im2, dx, v_step[1]) for im1, im2 in v_images]
        avg = np.mean(diffs)
        if avg < best_dx_score:
            best_dx_score = avg
            best_dx = dx

    log.info(f"  V dx: {current_dx:.1f} -> {best_dx} (score={best_dx_score:.6f})")

    del v_images
    gc.collect()

    refined_h = (h_step[0], float(best_dy))
    refined_v = (float(best_dx), v_step[1])

    log.info(f"  Refined step: H=({refined_h[0]:.1f}, {refined_h[1]:.1f}), "
             f"V=({refined_v[0]:.1f}, {refined_v[1]:.1f})")

    return refined_h, refined_v

def estimate_periodic_step(grid):
    log.info("Estimating step for periodic grid using Tukey phase correlation...")

    h_results = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            p1, p2 = grid[r][c], grid[r][c + 1]
            if p1 and p2:
                result = tukey_phase_corr(p1, p2)
                if result is not None:
                    dx, dy, val = result
                    if dx < -400 and abs(dx) > abs(dy):
                        h_results.append((dx, dy, val))

    v_results = []
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS):
            p1, p2 = grid[r][c], grid[r + 1][c]
            if p1 and p2:
                result = tukey_phase_corr(p1, p2)
                if result is not None:
                    dx, dy, val = result
                    if dy < -400 and abs(dy) > abs(dx):
                        v_results.append((dx, dy, val))

    log.info(f"Tukey phase corr: {len(h_results)}/90 H, {len(v_results)}/90 V passed sanity")

    if not h_results:
        return None, None

    h_dxs = np.array([r[0] for r in h_results])
    h_dys = np.array([r[1] for r in h_results])
    h_step = (float(np.median(h_dxs)), float(np.median(h_dys)))
    log.info(f"H step: ({h_step[0]:.1f}, {h_step[1]:.1f}), "
             f"std=({np.std(h_dxs):.1f}, {np.std(h_dys):.1f})")

    if v_results:
        v_dxs = np.array([r[0] for r in v_results])
        v_dys = np.array([r[1] for r in v_results])
        v_step = (float(np.median(v_dxs)), float(np.median(v_dys)))
    else:
        v_step = (-h_step[1], h_step[0])

    h_step, v_step = refine_periodic_h_step(grid, h_step, v_step)
    log.info(f"After profile NCC: H=({h_step[0]:.1f}, {h_step[1]:.1f}), "
             f"V=({v_step[0]:.1f}, {v_step[1]:.1f})")

    h_step, v_step = refine_minor_axes(grid, h_step, v_step, scan_range=25)

    expected_v_dx = h_step[1]
    if abs(v_step[0] - expected_v_dx) > 20:
        log.info(f"  V_dx correction: {v_step[0]:.1f} -> {expected_v_dx:.1f}")
        v_step = (expected_v_dx, v_step[1])

    log.info(f"Periodic grid step: H=({h_step[0]:.1f}, {h_step[1]:.1f}), "
             f"V=({v_step[0]:.1f}, {v_step[1]:.1f})")

    return h_step, v_step

def estimate_coarse_step(grid, features):
    h_candidates = []
    v_candidates = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            if grid[r][c] and grid[r][c + 1]:
                h_candidates.append((grid[r][c], grid[r][c + 1], r, c))
    for c in range(GRID_COLS):
        for r in range(GRID_ROWS - 1):
            if grid[r][c] and grid[r + 1][c]:
                v_candidates.append((grid[r][c], grid[r + 1][c], r, c))

    h_sift = []
    for p1, p2, r, c in h_candidates:
        if p1 in features and p2 in features:
            kp1, des1 = features[p1]
            kp2, des2 = features[p2]
            result = match_sift_pair(kp1, des1, kp2, des2)
            if result and abs(result[0]) > abs(result[1]) * 1.5:
                h_sift.append((result[0], result[1]))

    v_sift = []
    for p1, p2, r, c in v_candidates:
        if p1 in features and p2 in features:
            kp1, des1 = features[p1]
            kp2, des2 = features[p2]
            result = match_sift_pair(kp1, des1, kp2, des2)
            if result and abs(result[1]) > abs(result[0]) * 1.5:
                v_sift.append((result[0], result[1]))

    h_step, v_step = None, None
    if len(h_sift) >= 2:
        h_step = (float(np.median([s[0] for s in h_sift])),
                  float(np.median([s[1] for s in h_sift])))
        log.info(f"Coarse H from SIFT ({len(h_sift)}): ({h_step[0]:.0f}, {h_step[1]:.0f})")
    if len(v_sift) >= 2:
        v_step = (float(np.median([s[0] for s in v_sift])),
                  float(np.median([s[1] for s in v_sift])))
        log.info(f"Coarse V from SIFT ({len(v_sift)}): ({v_step[0]:.0f}, {v_step[1]:.0f})")
    if h_step and v_step:
        return h_step, v_step

    if not h_step or not v_step:
        log.info("SIFT coarse step incomplete, trying Tukey phase correlation...")
        tukey_h, tukey_v = estimate_periodic_step(grid)
        if tukey_h and not h_step:
            h_step = tukey_h
        if tukey_v and not v_step:
            v_step = tukey_v
        if h_step and v_step:
            return h_step, v_step

    log.info("Tukey incomplete, falling back to template matching...")
    SCALE = 4
    TMPL_4X = 200
    TMPL_FULL = 400
    VERIFY_MARGIN = 50

    def match_coarse_4x(path1, path2, direction):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        ih, iw = img1.shape
        sh, sw = ih // SCALE, iw // SCALE
        s1 = cv2.resize(img1, (sw, sh), interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(img2, (sw, sh), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        s1, s2 = clahe.apply(s1), clahe.apply(s2)
        if direction == 'h':
            tx, ty = sw - TMPL_4X - 10, (sh - TMPL_4X) // 2
        else:
            tx, ty = (sw - TMPL_4X) // 2, sh - TMPL_4X - 10
        if tx < 0 or ty < 0:
            return None
        template = s1[ty:ty + TMPL_4X, tx:tx + TMPL_4X]
        if direction == 'h':
            search_region = s2[:, :sw // 2 + TMPL_4X]
        else:
            search_region = s2[:sh // 2 + TMPL_4X, :]
        if search_region.shape[0] < TMPL_4X or search_region.shape[1] < TMPL_4X:
            return None
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.2:
            return None
        return ((max_loc[0] - tx) * SCALE, (max_loc[1] - ty) * SCALE, max_val)

    def match_coarse_fullres(path1, path2, direction, rough_dx, rough_dy):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        h, w = img1.shape
        ts = TMPL_FULL
        if direction == 'h':
            tx, ty = w - ts - 20, (h - ts) // 2
        else:
            tx, ty = (w - ts) // 2, h - ts - 20
        if tx < 0 or ty < 0:
            return None
        template = img1[ty:ty + ts, tx:tx + ts]
        sx, sy = int(tx + rough_dx), int(ty + rough_dy)
        rx0, ry0 = max(0, sx - VERIFY_MARGIN), max(0, sy - VERIFY_MARGIN)
        rx1, ry1 = min(w, sx + ts + VERIFY_MARGIN), min(h, sy + ts + VERIFY_MARGIN)
        if rx1 - rx0 < ts or ry1 - ry0 < ts:
            return None
        result = cv2.matchTemplate(img2[ry0:ry1, rx0:rx1], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.3:
            return None
        return (rx0 + max_loc[0] - tx, ry0 + max_loc[1] - ty, max_val)

    rough_h, rough_v, v_full = None, None, None

    if not h_step:
        h_res = [r for p1, p2, _, _ in h_candidates[:10] if (r := match_coarse_4x(p1, p2, 'h'))]
        if h_res:
            rough_h = (float(np.median([r[0] for r in h_res])),
                       float(np.median([r[1] for r in h_res])))

    if not v_step:
        v_res = [r for p1, p2, _, _ in v_candidates[:10] if (r := match_coarse_4x(p1, p2, 'v'))]
        if v_res:
            rough_v = (float(np.median([r[0] for r in v_res])),
                       float(np.median([r[1] for r in v_res])))

    if rough_h and not h_step:
        h_full = [r for p1, p2, _, _ in h_candidates[:10]
                  if (r := match_coarse_fullres(p1, p2, 'h', rough_h[0], rough_h[1]))]
        h_step = ((float(np.median([r[0] for r in h_full])),
                   float(np.median([r[1] for r in h_full]))) if h_full else rough_h)

    if rough_v and not v_step:
        v_full = [r for p1, p2, _, _ in v_candidates[:10]
                  if (r := match_coarse_fullres(p1, p2, 'v', rough_v[0], rough_v[1]))]
        v_step = ((float(np.median([r[0] for r in v_full])),
                   float(np.median([r[1] for r in v_full]))) if v_full else rough_v)

    if h_step and v_step and rough_v and not v_full:
        if abs(h_step[0]) > 100:
            tilt = h_step[1] / h_step[0]
            v_step = (-v_step[1] * tilt, v_step[1])

    return h_step, v_step

def compute_pairwise_transforms(grid, features):
    pairs = get_neighbor_pairs(grid)

    log.info(f"Tier A: SIFT matching on {len(pairs)} pairs...")
    t0 = time.time()

    sift_args = []
    for (r1, c1), (r2, c2), direction in pairs:
        path1, path2 = grid[r1][c1], grid[r2][c2]
        if path1 and path2 and path1 in features and path2 in features:
            sift_args.append(((r1, c1), (r2, c2), direction, features[path1], features[path2]))

    with Pool(NUM_WORKERS) as pool:
        sift_results = pool.map(do_sift_match, sift_args)

    transforms = {}
    succeeded = []
    failed_pairs = []
    for (r1, c1), (r2, c2), direction, result in sift_results:
        key = ((r1, c1), (r2, c2))
        if result is not None:
            transforms[key] = result
            succeeded.append((key, direction, result))
        else:
            failed_pairs.append(((r1, c1), (r2, c2), direction))

    log.info(f"Tier A: {len(succeeded)}/{len(pairs)} matched in {time.time() - t0:.1f}s")
    if not failed_pairs:
        return transforms, None

    h_dxs, h_dys, v_dxs, v_dys = [], [], [], []
    for key, direction, result in succeeded:
        dx, dy = result[0], result[1]
        if direction == 'h':
            h_dxs.append(dx); h_dys.append(dy)
        elif direction == 'v':
            v_dxs.append(dx); v_dys.append(dy)

    need_coarse = (len(h_dxs) < 3 or len(v_dxs) < 3)
    coarse_h, coarse_v = None, None
    if need_coarse:
        log.info("Too few SIFT matches, running coarse step estimation...")
        coarse_h, coarse_v = estimate_coarse_step(grid, features)

    if h_dxs:
        median_h_dx, median_h_dy = np.median(h_dxs), np.median(h_dys)
    elif coarse_h:
        median_h_dx, median_h_dy = coarse_h
    else:
        median_h_dx, median_h_dy = -cv2.imread(grid[0][0], cv2.IMREAD_GRAYSCALE).shape[1] * 0.4, 0

    if v_dxs:
        median_v_dx, median_v_dy = np.median(v_dxs), np.median(v_dys)
    elif coarse_v:
        median_v_dx, median_v_dy = coarse_v
    else:
        median_v_dx, median_v_dy = 0, -cv2.imread(grid[0][0], cv2.IMREAD_GRAYSCALE).shape[0] * 0.4

    log.info(f"Median steps: H=({median_h_dx:.1f}, {median_h_dy:.1f}), V=({median_v_dx:.1f}, {median_v_dy:.1f})")

    all_devs = []
    for key, direction, result in succeeded:
        dx, dy = result[0], result[1]
        if direction == 'h':
            dev = np.sqrt((dx - median_h_dx)**2 + (dy - median_h_dy)**2)
        elif direction == 'v':
            dev = np.sqrt((dx - median_v_dx)**2 + (dy - median_v_dy)**2)
        else:
            (r1, c1), (r2, c2) = key
            dc = c2 - c1
            exp_dx = median_h_dx * dc + median_v_dx
            exp_dy = median_h_dy * dc + median_v_dy
            dev = np.sqrt((dx - exp_dx)**2 + (dy - exp_dy)**2)
        all_devs.append(dev)

    if all_devs:
        dev_median = np.median(all_devs)
        dev_mad = np.median(np.abs(np.array(all_devs) - dev_median))
        dev_thresh = max(dev_median + 4 * dev_mad, 100.0)
    else:
        dev_thresh = 150.0

    sift_rejected = 0
    for key, direction, result in succeeded:
        dx, dy = result[0], result[1]
        (r1, c1), (r2, c2) = key
        if direction == 'h':
            exp_dx, exp_dy = median_h_dx, median_h_dy
        elif direction == 'v':
            exp_dx, exp_dy = median_v_dx, median_v_dy
        else:
            dc = c2 - c1
            exp_dx = median_h_dx * dc + median_v_dx
            exp_dy = median_h_dy * dc + median_v_dy
        dev = np.sqrt((dx - exp_dx)**2 + (dy - exp_dy)**2)
        if dev > dev_thresh:
            transforms[key] = (exp_dx, exp_dy, 1, 0.0)
            sift_rejected += 1
            failed_pairs.append((key[0], key[1], direction))

    if sift_rejected:
        log.info(f"  Rejected {sift_rejected} SIFT outliers (>{dev_thresh:.0f}px)")

    is_periodic = len(succeeded) == 0
    h_failed = [(a, b, d) for a, b, d in failed_pairs if d == 'h']
    v_failed = [(a, b, d) for a, b, d in failed_pairs if d == 'v']
    d_failed = [(a, b, d) for a, b, d in failed_pairs if d == 'd']
    t0 = time.time()
    tier_b_ok = 0
    tier_b_rejected = 0
    half_period = GRID_PERIOD / 2

    if is_periodic:
        log.info(f"Periodic grid: assigning uniform step to {len(failed_pairs)} pairs")
        for (r1, c1), (r2, c2), direction in failed_pairs:
            key = ((r1, c1), (r2, c2))
            if direction == 'h':
                transforms[key] = (median_h_dx, median_h_dy, 5, 0.0)
            elif direction == 'd':
                dc = c2 - c1
                transforms[key] = (median_h_dx * dc + median_v_dx,
                                   median_h_dy * dc + median_v_dy, 3, 0.0)
            else:
                transforms[key] = (median_v_dx, median_v_dy, 5, 0.0)
            tier_b_ok += 1
    else:
        if h_failed:
            h_args = [((r1, c1), (r2, c2), d, grid[r1][c1], grid[r2][c2], median_h_dx, median_h_dy)
                      for (r1, c1), (r2, c2), d in h_failed]
            with Pool(NUM_WORKERS) as pool:
                h_results = pool.map(do_template_match, h_args)
            for (r1, c1), (r2, c2), direction, result in h_results:
                key = ((r1, c1), (r2, c2))
                if result is not None:
                    if abs(result[0] - median_h_dx) > half_period or abs(result[1] - median_h_dy) > half_period:
                        result = None
                        tier_b_rejected += 1
                if result is not None:
                    transforms[key] = result
                    tier_b_ok += 1
                else:
                    transforms[key] = (median_h_dx, median_h_dy, 1, 0.0)

        refined_h_dx, refined_h_dy = median_h_dx, median_h_dy
        if len(h_dxs) >= 3:
            all_h_dys = list(h_dys)
            all_h_dxs = list(h_dxs)
            for (k1, k2), (dx, dy, conf, _) in transforms.items():
                if k1[0] == k2[0] and conf > 1:
                    all_h_dxs.append(dx)
                    all_h_dys.append(dy)
            refined_h_dx = np.median(all_h_dxs)
            refined_h_dy = np.median(all_h_dys)
            if abs(refined_h_dx) > 100:
                refined_tilt = refined_h_dy / refined_h_dx
                refined_v_dx = -median_v_dy * refined_tilt
                if abs(refined_v_dx - median_v_dx) > 3:
                    median_v_dx = refined_v_dx

        if v_failed:
            v_args = [((r1, c1), (r2, c2), d, grid[r1][c1], grid[r2][c2], median_v_dx, median_v_dy)
                      for (r1, c1), (r2, c2), d in v_failed]
            with Pool(NUM_WORKERS) as pool:
                v_results = pool.map(do_template_match, v_args)
            for (r1, c1), (r2, c2), direction, result in v_results:
                key = ((r1, c1), (r2, c2))
                if result is not None:
                    if abs(result[0] - median_v_dx) > half_period or abs(result[1] - median_v_dy) > half_period:
                        result = None
                        tier_b_rejected += 1
                if result is not None:
                    transforms[key] = result
                    tier_b_ok += 1
                else:
                    transforms[key] = (median_v_dx, median_v_dy, 1, 0.0)

        if d_failed:
            d_args = []
            for (r1, c1), (r2, c2), d in d_failed:
                dc = c2 - c1
                exp_dx = median_h_dx * dc + median_v_dx
                exp_dy = median_h_dy * dc + median_v_dy
                d_args.append(((r1, c1), (r2, c2), d, grid[r1][c1], grid[r2][c2], exp_dx, exp_dy))
            with Pool(NUM_WORKERS) as pool:
                d_results = pool.map(do_template_match, d_args)
            for (r1, c1), (r2, c2), direction, result in d_results:
                key = ((r1, c1), (r2, c2))
                dc = c2 - c1
                exp_dx = median_h_dx * dc + median_v_dx
                exp_dy = median_h_dy * dc + median_v_dy
                if result is not None:
                    if abs(result[0] - exp_dx) > half_period or abs(result[1] - exp_dy) > half_period:
                        result = None
                        tier_b_rejected += 1
                if result is not None:
                    transforms[key] = result
                    tier_b_ok += 1
                else:
                    transforms[key] = (exp_dx, exp_dy, 1, 0.0)

    log.info(f"Tier B: {tier_b_ok}/{len(failed_pairs)} matched in {time.time() - t0:.1f}s")

    is_periodic = not succeeded

    if is_periodic:
        log.info("Periodic grid: constrained template matching...")
        t0 = time.time()

        tmpl_args = [((r1, c1), (r2, c2), d, grid[r1][c1], grid[r2][c2],
                      transforms[((r1, c1), (r2, c2))][0], transforms[((r1, c1), (r2, c2))][1])
                     for (r1, c1), (r2, c2), d in pairs]
        with Pool(NUM_WORKERS) as pool:
            tmpl_results = pool.map(do_template_match, tmpl_args)

        n_refined = 0
        for (r1, c1), (r2, c2), direction, result in tmpl_results:
            key = ((r1, c1), (r2, c2))
            if result is None:
                continue
            new_dx, new_dy, new_conf, _ = result
            transforms[key] = (new_dx, new_dy, max(new_conf, 10), 0.0)
            n_refined += 1

        h_dxs_p1 = [transforms[((r1,c1),(r2,c2))][0] for (r1,c1),(r2,c2),d in pairs if d == 'h']
        h_dys_p1 = [transforms[((r1,c1),(r2,c2))][1] for (r1,c1),(r2,c2),d in pairs if d == 'h']
        v_dxs_p1 = [transforms[((r1,c1),(r2,c2))][0] for (r1,c1),(r2,c2),d in pairs if d == 'v']
        v_dys_p1 = [transforms[((r1,c1),(r2,c2))][1] for (r1,c1),(r2,c2),d in pairs if d == 'v']
        med2_h = (np.median(h_dxs_p1), np.median(h_dys_p1))
        med2_v = (np.median(v_dxs_p1), np.median(v_dys_p1))

        tmpl_args2 = []
        for (r1, c1), (r2, c2), direction in pairs:
            path1, path2 = grid[r1][c1], grid[r2][c2]
            key = ((r1, c1), (r2, c2))
            cur = transforms.get(key)
            if cur and abs(cur[0] - (med2_h[0] if direction == 'h' else med2_v[0])) < 80:
                exp_dx, exp_dy = cur[0], cur[1]
            else:
                exp_dx = med2_h[0] if direction == 'h' else med2_v[0]
                exp_dy = med2_h[1] if direction == 'h' else med2_v[1]
            tmpl_args2.append(((r1, c1), (r2, c2), direction, path1, path2, exp_dx, exp_dy))

        with Pool(NUM_WORKERS) as pool:
            tmpl_results2 = pool.map(do_template_match, tmpl_args2)

        n_pass2 = 0
        for (r1, c1), (r2, c2), direction, result in tmpl_results2:
            key = ((r1, c1), (r2, c2))
            if result is None:
                continue
            new_dx, new_dy, new_conf, _ = result
            old = transforms.get(key, (0, 0, 0, 0))
            if new_conf > old[2]:
                transforms[key] = (new_dx, new_dy, max(new_conf, 10), 0.0)
                n_pass2 += 1

        log.info(f"Periodic: pass 1 refined {n_refined}, pass 2 improved {n_pass2} in {time.time()-t0:.1f}s")

        grid_period = GRID_PERIOD
        SNAP_THRESH = 40
        h_dxs_all, h_dys_all, v_dxs_all, v_dys_all = [], [], [], []
        for (r1, c1), (r2, c2), direction in pairs:
            key = ((r1, c1), (r2, c2))
            dx, dy = transforms[key][0], transforms[key][1]
            if direction == 'h':
                h_dxs_all.append(dx); h_dys_all.append(dy)
            else:
                v_dxs_all.append(dx); v_dys_all.append(dy)

        h_ref_dx = np.median(h_dxs_all) if h_dxs_all else median_h_dx
        h_ref_dy = np.median(h_dys_all) if h_dys_all else median_h_dy
        v_ref_dx = np.median(v_dxs_all) if v_dxs_all else median_v_dx
        v_ref_dy = np.median(v_dys_all) if v_dys_all else median_v_dy

        n_snapped = 0
        for (r1, c1), (r2, c2), direction in pairs:
            key = ((r1, c1), (r2, c2))
            dx, dy, conf, angle = transforms[key]
            ref_dx = h_ref_dx if direction == 'h' else v_ref_dx
            ref_dy = h_ref_dy if direction == 'h' else v_ref_dy
            dev = np.sqrt((dx - ref_dx)**2 + (dy - ref_dy)**2)

            if dev > SNAP_THRESH:
                best_dx, best_dy, best_dev = dx, dy, dev
                for nx in range(-2, 3):
                    for ny in range(-2, 3):
                        if nx == 0 and ny == 0:
                            continue
                        snap_dx = dx - nx * GRID_PERIOD
                        snap_dy = dy - ny * GRID_PERIOD
                        snap_dev = np.sqrt((snap_dx - ref_dx)**2 + (snap_dy - ref_dy)**2)
                        if snap_dev < best_dev:
                            best_dev = snap_dev
                            best_dx, best_dy = snap_dx, snap_dy
                if best_dev < SNAP_THRESH:
                    transforms[key] = (best_dx, best_dy, conf, angle)
                    n_snapped += 1

        if n_snapped:
            log.info(f"  Grid aliasing: snapped {n_snapped} pairs")

        periodic_step = (median_h_dx, median_h_dy, median_v_dx, median_v_dy)
        return transforms, periodic_step

    log.info(f"Tier C: Phase correlation on {len(pairs)} pairs...")
    t0 = time.time()

    phase_args = []
    for (r1, c1), (r2, c2), direction in pairs:
        key = ((r1, c1), (r2, c2))
        if key in transforms:
            exp_dx, exp_dy = transforms[key][0], transforms[key][1]
        else:
            if direction == 'h':
                exp_dx, exp_dy = median_h_dx, median_h_dy
            else:
                exp_dx, exp_dy = median_v_dx, median_v_dy
        phase_args.append(((r1, c1), (r2, c2), direction, grid[r1][c1], grid[r2][c2], exp_dx, exp_dy))

    with Pool(NUM_WORKERS) as pool:
        phase_results = pool.map(do_phase_match, phase_args)

    phase_ok, phase_improved = 0, 0
    for (r1, c1), (r2, c2), direction, result in phase_results:
        key = ((r1, c1), (r2, c2))
        if result is None:
            continue
        phase_ok += 1
        phase_dx, phase_dy, phase_conf, _ = result
        old_dx, old_dy, old_conf, old_angle = transforms.get(key, (0, 0, 0, 0))
        diff = np.sqrt((phase_dx - old_dx)**2 + (phase_dy - old_dy)**2)

        if is_periodic and diff > 30:
            continue
        if old_conf <= 1:
            transforms[key] = (phase_dx, phase_dy, max(phase_conf, 5), old_angle)
            phase_improved += 1
        elif diff > 3.0 and phase_conf >= old_conf * 0.5:
            w_phase = phase_conf * 2.0
            w_old = old_conf
            w_total = w_phase + w_old
            new_dx = (phase_dx * w_phase + old_dx * w_old) / w_total
            new_dy = (phase_dy * w_phase + old_dy * w_old) / w_total
            transforms[key] = (new_dx, new_dy, max(old_conf, phase_conf), old_angle)
            phase_improved += 1

    log.info(f"Tier C: {phase_ok}/{len(pairs)} matched, {phase_improved} improved in {time.time()-t0:.1f}s")

    return transforms, None

def rematch_rotated_pairs(grid, features, transforms):
    PAIR_ANGLE_THRESH = 0.3
    MIN_ANGLE_CONF = 10

    rotated_pairs = {}
    for key, (dx, dy, conf, angle) in transforms.items():
        if conf >= MIN_ANGLE_CONF and abs(angle) > PAIR_ANGLE_THRESH:
            rotated_pairs[key] = angle

    if not rotated_pairs:
        log.info("No rotated pairs found")
        return transforms, {}

    log.info(f"Found {len(rotated_pairs)} pairs with |angle| > {PAIR_ANGLE_THRESH}deg")

    tile_angles = {}
    for ((r1, c1), (r2, c2)), (dx, dy, conf, angle) in transforms.items():
        if conf >= MIN_ANGLE_CONF:
            tile_angles.setdefault((r1, c1), []).append(angle)
            tile_angles.setdefault((r2, c2), []).append(-angle)

    all_medians = []
    for angles in tile_angles.values():
        if angles:
            all_medians.append(np.median(angles))
    global_median = np.median(all_medians) if all_medians else 0.0

    detected_tile_rotations = {}
    for (r, c), angles in tile_angles.items():
        if not angles:
            continue
        median_angle = np.median(angles) - global_median
        if abs(median_angle) > 1.0:
            detected_tile_rotations[(r, c)] = median_angle

    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bf = cv2.BFMatcher()
    derot_cache = {}

    def get_derot_features(path, angle):
        cache_key = (path, round(angle, 2))
        if cache_key in derot_cache:
            return derot_cache[cache_key]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        img_derot = cv2.warpAffine(img, M_rot, (w, h))
        valid_mask = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, M_rot, (w, h))
        valid_mask = (valid_mask > 128).astype(np.uint8) * 255
        kp, des = sift.detectAndCompute(clahe.apply(img_derot), valid_mask)
        derot_cache[cache_key] = (kp, des)
        return kp, des

    def get_features(path):
        cache_key = (path, 0.0)
        if cache_key in derot_cache:
            return derot_cache[cache_key]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(clahe.apply(img), None)
        derot_cache[cache_key] = (kp, des)
        return kp, des

    n_rematched = 0
    for key, pair_angle in rotated_pairs.items():
        (r1, c1), (r2, c2) = key
        path1, path2 = grid[r1][c1], grid[r2][c2]
        if not path1 or not path2:
            continue

        kp1, des1 = get_derot_features(path1, pair_angle)
        kp2, des2 = get_features(path2)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
        if len(good) < 6:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                  ransacReprojThreshold=RANSAC_THRESH)
        if M is None:
            continue
        n_inliers = int(np.sum(inliers))
        if n_inliers < MIN_INLIERS:
            continue

        new_dx, new_dy = float(M[0, 2]), float(M[1, 2])
        old_dx, old_dy, old_conf, old_angle = transforms[key]
        log.info(f"  ({r1},{c1})->({r2},{c2}): derot SIFT dx={new_dx:.1f} (was {old_dx:.1f}), "
                 f"dy={new_dy:.1f} (was {old_dy:.1f}), inliers={n_inliers}")
        transforms[key] = (new_dx, new_dy, n_inliers, old_angle)
        n_rematched += 1

    log.info(f"Re-matched {n_rematched}/{len(rotated_pairs)} rotated pairs")
    del derot_cache
    return transforms, detected_tile_rotations

def global_alignment(grid, transforms, periodic_step=None):
    n_tiles = GRID_ROWS * GRID_COLS

    def rc_to_idx(r, c):
        return r * GRID_COLS + c

    edges = []
    for ((r1, c1), (r2, c2)), (dx, dy, conf, angle) in transforms.items():
        i, j = rc_to_idx(r1, c1), rc_to_idx(r2, c2)
        edges.append((i, j, -dx, -dy, max(1, conf), angle))

    positions = np.zeros((n_tiles, 2))
    visited = set()

    adj = [[] for _ in range(n_tiles)]
    for i, j, dx, dy, w, angle in edges:
        adj[i].append((j, dx, dy, w, angle))
        adj[j].append((i, -dx, -dy, w, -angle))

    tile_conf = np.zeros(n_tiles)
    for i, j, dx, dy, w, angle in edges:
        tile_conf[i] += w
        tile_conf[j] += w
    root = int(np.argmax(tile_conf))
    log.info(f"BFS root: tile ({root // GRID_COLS},{root % GRID_COLS})")

    rotations = np.zeros(n_tiles)
    queue = deque([root])
    visited.add(root)

    while queue:
        node = queue.popleft()
        for neighbor, dx, dy, w, angle in sorted(adj[node], key=lambda x: -x[3]):
            if neighbor not in visited:
                positions[neighbor] = positions[node] + np.array([dx, dy])
                rotations[neighbor] = rotations[node] - angle
                visited.add(neighbor)
                queue.append(neighbor)

    for i in range(n_tiles):
        if i not in visited:
            r, c = i // GRID_COLS, i % GRID_COLS
            log.warning(f"Tile ({r},{c}) not reached by BFS")
            positions[i] = [c * 1500, r * 1500]

    if len(edges) > n_tiles:
        log.info("Refining with least-squares...")
        active_edges = list(edges)

        tile_rots = rotations.copy()
        for r in range(GRID_ROWS):
            row_rots = [tile_rots[r * GRID_COLS + c] for c in range(GRID_COLS)]
            row_med = float(np.median(row_rots))
            for c in range(GRID_COLS):
                tile_rots[r * GRID_COLS + c] = row_med
        tile_rots -= tile_rots[root]
        for i in range(n_tiles):
            if abs(tile_rots[i]) < 0.2:
                tile_rots[i] = 0.0

        def rebuild_bfs(edge_list):
            nonlocal rotations
            adj_clean = [[] for _ in range(n_tiles)]
            for i, j, dx, dy, w, angle in edge_list:
                adj_clean[i].append((j, dx, dy, w, angle))
                adj_clean[j].append((i, -dx, -dy, w, -angle))
            pos = np.zeros((n_tiles, 2))
            rot = np.zeros(n_tiles)
            vis = set()
            q = deque([root])
            vis.add(root)
            while q:
                node = q.popleft()
                for neighbor, dx, dy, w, angle in sorted(adj_clean[node], key=lambda x: -x[3]):
                    if neighbor not in vis:
                        pos[neighbor] = pos[node] + np.array([dx, dy])
                        rot[neighbor] = rot[node] - angle
                        vis.add(neighbor)
                        q.append(neighbor)
            for i in range(n_tiles):
                if i not in vis:
                    ri, ci = i // GRID_COLS, i % GRID_COLS
                    nbr_pos = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = ri + dr, ci + dc
                        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                            ni = nr * GRID_COLS + nc
                            if ni in vis:
                                nbr_pos.append(pos[ni] + np.array([-dc * 1200, -dr * 1200]))
                    pos[i] = np.mean(nbr_pos, axis=0) if nbr_pos else [ci * 1500, ri * 1500]
                    vis.add(i)
            rotations = rot
            return pos

        for iteration in range(5):
            def residuals(params, _edges=active_edges):
                pos = params.reshape(n_tiles, 2)
                res = []
                for i, j, dx, dy, w, _angle in _edges:
                    sqrt_w = np.sqrt(w)
                    theta_i = np.radians(tile_rots[i])
                    if abs(theta_i) > 1e-6:
                        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)
                        dx_w = dx * cos_t - dy * sin_t
                        dy_w = dx * sin_t + dy * cos_t
                    else:
                        dx_w, dy_w = dx, dy
                    res.append(sqrt_w * (pos[j, 0] - pos[i, 0] - dx_w))
                    res.append(sqrt_w * (pos[j, 1] - pos[i, 1] - dy_w))
                res.append(10 * pos[root, 0])
                res.append(10 * pos[root, 1])
                return np.array(res)

            result = least_squares(residuals, positions.flatten(), method='lm')
            positions = result.x.reshape(n_tiles, 2)

            edge_residuals = np.array([
                np.sqrt((positions[j, 0] - positions[i, 0] - dx)**2 +
                        (positions[j, 1] - positions[i, 1] - dy)**2)
                for i, j, dx, dy, w, _ in active_edges])

            outlier_mask = edge_residuals > 40.0
            n_outliers = np.sum(outlier_mask)

            if n_outliers == 0:
                log.info(f"  Iter {iteration+1}: residual={result.cost:.2f}, no outliers")
                break

            for idx, (i, j, dx, dy, w, _) in enumerate(active_edges):
                if outlier_mask[idx]:
                    ri, ci = i // GRID_COLS, i % GRID_COLS
                    rj, cj = j // GRID_COLS, j % GRID_COLS
                    log.info(f"  Outlier: ({ri},{ci})->({rj},{cj}) "
                             f"residual={edge_residuals[idx]:.1f}px")

            active_edges = [e for e, is_out in zip(active_edges, outlier_mask) if not is_out]
            log.info(f"  Iter {iteration+1}: residual={result.cost:.2f}, "
                     f"removed {n_outliers}, {len(active_edges)} remain")
            positions = rebuild_bfs(active_edges)

        log.info(f"Alignment residual: {result.cost:.2f} "
                 f"({len(edges) - len(active_edges)} edges removed)")

    log.info("Post-LS: snapping grid-aliased tile positions...")

    center_tile = grid[GRID_ROWS // 2][GRID_COLS // 2]
    if center_tile:
        ct_img = cv2.imread(center_tile, cv2.IMREAD_GRAYSCALE)
        ih, iw = ct_img.shape
        strip = ct_img[ih//2-200:ih//2+200, iw//4:3*iw//4].astype(np.float32)
        prof = np.mean(strip, axis=0)
        prof -= np.mean(prof)
        corr = np.correlate(prof, prof, mode='full')
        corr = corr[len(prof)-1:]
        corr = corr / max(corr[0], 1)
        period_peaks = []
        for i in range(50, min(250, len(corr)-1)):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > 0.2:
                period_peaks.append(i)
                break
        GP = period_peaks[0] if period_peaks else 118
        log.info(f"  Grid period: {GP}px")
    else:
        GP = 118

    h_dxs_all, h_dys_all, v_dxs_all, v_dys_all = [], [], [], []
    for ((r1, c1), (r2, c2)), (dx, dy, conf, _) in transforms.items():
        if r1 == r2 and abs(c2 - c1) == 1:
            h_dxs_all.append(dx); h_dys_all.append(dy)
        elif c1 == c2 and abs(r2 - r1) == 1:
            v_dxs_all.append(dx); v_dys_all.append(dy)

    med_h_dx = np.median(h_dxs_all) if h_dxs_all else -1164.0
    med_h_dy = np.median(h_dys_all) if h_dys_all else 49.0
    med_v_dx = np.median(v_dxs_all) if v_dxs_all else 49.0
    med_v_dy = np.median(v_dys_all) if v_dys_all else -1164.0

    n_rejected = len(edges) - len(active_edges)
    rejection_rate = n_rejected / max(len(edges), 1)

    if rejection_rate > 0.15:
        log.info(f"  High outlier rate ({rejection_rate:.0%}), using uniform grid")
        if periodic_step is not None:
            uf_h_dx, uf_h_dy = periodic_step[0], periodic_step[1]
            uf_v_dx, uf_v_dy = periodic_step[2], periodic_step[3]
        else:
            major = (med_h_dx + med_v_dy) / 2
            minor = (med_h_dy + med_v_dx) / 2
            uf_h_dx, uf_h_dy = major, minor
            uf_v_dx, uf_v_dy = minor, major

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                idx = r * GRID_COLS + c
                positions[idx, 0] = c * (-uf_h_dx) + r * (-uf_v_dx)
                positions[idx, 1] = c * (-uf_h_dy) + r * (-uf_v_dy)
    else:
        log.info(f"  Outlier rate OK ({rejection_rate:.0%}), keeping LS positions")

    positions[:, 0] -= positions[:, 0].min()
    positions[:, 1] -= positions[:, 1].min()

    for r in range(GRID_ROWS):
        row_rots = [rotations[r * GRID_COLS + c] for c in range(GRID_COLS)]
        row_median = float(np.median(row_rots))
        for c in range(GRID_COLS):
            rotations[r * GRID_COLS + c] = row_median

    root_row = root // GRID_COLS
    rotations -= rotations[root_row * GRID_COLS]
    for i in range(n_tiles):
        if abs(rotations[i]) < 0.2:
            rotations[i] = 0.0

    logged_rows = set()
    for i in range(n_tiles):
        if abs(rotations[i]) > 0.01:
            r = i // GRID_COLS
            if r not in logged_rows:
                log.info(f"Row {r} rotation: {rotations[i]:+.2f}deg")
                logged_rows.add(r)

    pos_grid = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    rot_grid = [[0.0] * GRID_COLS for _ in range(GRID_ROWS)]
    for i in range(n_tiles):
        r, c = i // GRID_COLS, i % GRID_COLS
        pos_grid[r][c] = (positions[i, 0], positions[i, 1])
        rot_grid[r][c] = float(rotations[i])

    return pos_grid, rot_grid

def estimate_vignetting(grid, positions):
    log.info("Estimating vignetting pattern...")
    center_paths = [grid[r][c] for r in range(3, 7) for c in range(3, 7) if grid[r][c]]
    if not center_paths:
        return None
    accumulator = None
    count = 0
    for path in center_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if accumulator is None:
            accumulator = np.zeros_like(gray)
        accumulator += gray
        count += 1
    if count == 0:
        return None
    vignette = cv2.GaussianBlur(accumulator / count, (0, 0), sigmaX=200, sigmaY=200)
    vmax = vignette.max()
    if vmax > 0:
        vignette = vignette / vmax
    vignette = np.clip(vignette, 0.4, 1.0)
    return vignette[:, :, np.newaxis]

def composite_mosaic(grid, positions, output_path, rotations=None):
    h0, w0 = cv2.imread(grid[0][0], cv2.IMREAD_GRAYSCALE).shape
    img_w, img_h = w0, h0

    tile_rot = {}
    tile_dims = {}
    if rotations:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rot_deg = rotations[r][c]
                if abs(rot_deg) > 0.5:
                    tile_rot[(r, c)] = rot_deg
                    rad = np.radians(abs(rot_deg))
                    cos_a, sin_a = np.cos(rad), np.sin(rad)
                    eff_w = int(np.ceil(img_w * cos_a + img_h * sin_a))
                    eff_h = int(np.ceil(img_w * sin_a + img_h * cos_a))
                    dx_off = (img_w - eff_w) / 2.0
                    dy_off = (img_h - eff_h) / 2.0
                    tile_dims[(r, c)] = (eff_w, eff_h, dx_off, dy_off)

    max_x, max_y = 0, 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            pos = positions[r][c]
            if pos is None:
                continue
            px, py = pos
            if (r, c) in tile_dims:
                ew, eh, dxo, dyo = tile_dims[(r, c)]
                max_x = max(max_x, px + dxo + ew)
                max_y = max(max_y, py + dyo + eh)
            else:
                max_x = max(max_x, px + img_w)
                max_y = max(max_y, py + img_h)

    canvas_w = int(np.ceil(max_x))
    canvas_h = int(np.ceil(max_y))
    log.info(f"Canvas size: {canvas_w} x {canvas_h}")
    vignette = estimate_vignetting(grid, positions)
    tiles = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            path = grid[r][c]
            if path and positions[r][c] is not None:
                px, py = positions[r][c]
                if (r, c) in tile_dims:
                    ew, eh, dxo, dyo = tile_dims[(r, c)]
                    tiles.append((px + dxo, py + dyo, r, c, path, ew, eh))
                else:
                    tiles.append((px, py, r, c, path, img_w, img_h))

    log.info("Computing per-tile gain compensation...")
    tile_means = {}
    for px, py, r, c, path, tw, th in tiles:
        img = cv2.imread(path)
        if vignette is not None:
            img = np.clip(img.astype(np.float32) / vignette, 0, 255)
        h, w = img.shape[:2]
        tile_means[(r, c)] = float(img[h//4:3*h//4, w//4:3*w//4].mean())
        del img

    tile_gains = {}
    if tile_means:
        target = np.median(list(tile_means.values()))
        tile_gains = {k: target / m if m > 0 else 1.0 for k, m in tile_means.items()}
        log.info(f"Gain range: {min(tile_gains.values()):.3f} - {max(tile_gains.values()):.3f}")

    gc.collect()

    FEATHER_PX = 30
    STRIP_HEIGHT = 500
    log.info(f"Compositing {len(tiles)} tiles with feathered Voronoi...")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for strip_y0 in range(0, canvas_h, STRIP_HEIGHT):
        strip_y1 = min(strip_y0 + STRIP_HEIGHT, canvas_h)
        strip_h = strip_y1 - strip_y0

        overlapping = [(px, py, r, c, path, tw, th) for px, py, r, c, path, tw, th in tiles
                       if int(round(py)) < strip_y1 and int(round(py)) + th > strip_y0]
        if not overlapping:
            continue

        strip_sum = np.zeros((strip_h, canvas_w, 3), dtype=np.float32)
        strip_wt = np.zeros((strip_h, canvas_w), dtype=np.float32)

        all_dist = []
        all_regions = []
        for px, py, r, c, path, tw, th in overlapping:
            ix, iy = int(round(px)), int(round(py))
            tile_y0 = max(0, strip_y0 - iy)
            tile_y1 = min(th, strip_y1 - iy)
            tile_x0 = max(0, -ix)
            tile_x1 = min(tw, canvas_w - ix)
            if tile_y1 <= tile_y0 or tile_x1 <= tile_x0:
                all_dist.append(None)
                all_regions.append(None)
                continue

            cy0 = max(0, iy - strip_y0)
            cy1 = cy0 + (tile_y1 - tile_y0)
            cx0 = max(0, ix)
            cx1 = cx0 + (tile_x1 - tile_x0)

            center_x, center_y = px + tw / 2.0, py + th / 2.0
            xs = np.arange(cx0, cx1, dtype=np.float32)
            ys = np.arange(strip_y0 + cy0, strip_y0 + cy1, dtype=np.float32)
            dist = np.sqrt((xs[np.newaxis, :] - center_x)**2 + (ys[:, np.newaxis] - center_y)**2)
            all_dist.append(dist)
            all_regions.append((cy0, cy1, cx0, cx1, tile_y0, tile_y1, tile_x0, tile_x1))

        for idx, (px, py, r, c, path, tw, th) in enumerate(overlapping):
            if all_dist[idx] is None:
                continue

            dist = all_dist[idx]
            cy0, cy1, cx0, cx1, tile_y0, tile_y1, tile_x0, tile_x1 = all_regions[idx]

            min_other_dist = np.full_like(dist, np.inf)
            for jdx, (px2, py2, r2, c2, _, tw2, th2) in enumerate(overlapping):
                if jdx == idx or all_dist[jdx] is None or all_regions[jdx] is None:
                    continue
                ocy0, ocy1, ocx0, ocx1 = all_regions[jdx][0], all_regions[jdx][1], all_regions[jdx][2], all_regions[jdx][3]
                ry0, ry1 = max(cy0, ocy0), min(cy1, ocy1)
                rx0, rx1 = max(cx0, ocx0), min(cx1, ocx1)
                if ry0 >= ry1 or rx0 >= rx1:
                    continue
                other_cx, other_cy = px2 + tw2 / 2.0, py2 + th2 / 2.0
                xs = np.arange(rx0, rx1, dtype=np.float32)
                ys = np.arange(strip_y0 + ry0, strip_y0 + ry1, dtype=np.float32)
                od = np.sqrt((xs[np.newaxis, :] - other_cx)**2 + (ys[:, np.newaxis] - other_cy)**2)
                local = min_other_dist[ry0 - cy0:ry1 - cy0, rx0 - cx0:rx1 - cx0]
                np.minimum(local, od, out=local)

            margin = min_other_dist - dist
            weight = np.clip(margin / (2.0 * FEATHER_PX) + 0.5, 0, 1)
            weight = np.where(np.isinf(min_other_dist), 1.0, weight)

            img = cv2.imread(path)
            if vignette is not None:
                img = img.astype(np.float32) / vignette
                gain = tile_gains.get((r, c), 1.0)
                if gain != 1.0:
                    img *= gain
                img = np.clip(img, 0, 255)
            else:
                img = img.astype(np.float32)

            rot_deg = tile_rot.get((r, c), 0.0)
            if abs(rot_deg) > 0.5:
                h_orig, w_orig = img.shape[:2]
                M_rot = cv2.getRotationMatrix2D((w_orig / 2.0, h_orig / 2.0), -rot_deg, 1.0)
                M_rot[0, 2] += (tw - w_orig) / 2.0
                M_rot[1, 2] += (th - h_orig) / 2.0
                img = cv2.warpAffine(img, M_rot, (tw, th), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            tile_region = img[tile_y0:tile_y1, tile_x0:tile_x1]
            w3 = weight[:, :, np.newaxis]
            strip_sum[cy0:cy1, cx0:cx1] += tile_region * w3
            strip_wt[cy0:cy1, cx0:cx1] += weight
            del img, tile_region, dist, weight, min_other_dist

        mask = strip_wt > 0
        for ch in range(3):
            strip_sum[:, :, ch] = np.where(mask, strip_sum[:, :, ch] / np.maximum(strip_wt, 1e-6), 0)
        canvas[strip_y0:strip_y1] = np.clip(strip_sum, 0, 255).astype(np.uint8)
        del strip_sum, strip_wt, all_dist, all_regions
    gc.collect()
    canvas, _ = crop_to_content(canvas)
    log.info(f"Final mosaic: {canvas.shape[1]} x {canvas.shape[0]}")
    png_path = output_path.rsplit('.', 1)[0] + '.png'
    cv2.imwrite(png_path, canvas)
    log.info(f"Saved PNG: {png_path}")
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    log.info(f"Saved JPEG: {output_path}")
    return canvas

def crop_to_content(image, threshold=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image, (0, 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image[rmin:rmax + 1, cmin:cmax + 1], (cmin, rmin)

def load_shift_model():
    class ShiftNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.regressor = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.regressor(x).squeeze(-1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ShiftNet()
    model.load_state_dict(torch.load(SHIFT_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def measure_shift(shift_model, device, gray_enh, x, y, grid_period=117):
    PATCH = 64
    h, w = gray_enh.shape
    hw = PATCH // 2
    if x - hw < 0 or x + hw > w or y - hw < 0 or y + hw > h:
        return None
    patch = gray_enh[y - hw:y + hw, x - hw:x + hw].astype(np.float32) / 255.0
    tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        shift_px = float(shift_model(tensor).cpu().item())
    shift_px = max(0, shift_px)
    um_per_px = 20.0 / grid_period
    if shift_px < 0.5:
        return 0.0, 0.0
    return round(shift_px, 2), round(shift_px * um_per_px, 3)

def enhance_tile(gray):
    clahe = cv2.createCLAHE(clipLimit=YOLO_CLAHE_CLIP, tileGridSize=(8, 8))
    return clahe.apply(gray)

def is_on_grid(enh, x, y, patch_hw=80, period_range=(80, 160)):
    h, w = enh.shape
    y0, y1 = max(0, y - patch_hw), min(h, y + patch_hw)
    x0, x1 = max(0, x - patch_hw), min(w, x + patch_hw)
    patch = enh[y0:y1, x0:x1].astype(np.float32)
    if patch.shape[0] < 100 or patch.shape[1] < 100:
        return False
    for axis in [0, 1]:
        proj = np.mean(patch, axis=1 - axis)
        proj -= np.mean(proj)
        if np.std(proj) < 2:
            continue
        acf = np.correlate(proj, proj, mode='full')
        acf = acf[len(acf) // 2:]
        acf /= acf[0] + 1e-10
        seg = acf[period_range[0]:min(period_range[1], len(acf))]
        if len(seg) > 0 and np.max(seg) > 0.25:
            return True
    return False

def detect_tile_yolo(model, tile_path, conf=YOLO_CONF):
    gray = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    enh = enhance_tile(gray)
    img = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    all_boxes = []
    for cy in range(0, h - YOLO_CROP_SIZE // 2, YOLO_CROP_SIZE - YOLO_CROP_OVERLAP):
        for cx in range(0, w - YOLO_CROP_SIZE // 2, YOLO_CROP_SIZE - YOLO_CROP_OVERLAP):
            x0 = min(cx, w - YOLO_CROP_SIZE)
            y0 = min(cy, h - YOLO_CROP_SIZE)
            crop = img[y0:y0 + YOLO_CROP_SIZE, x0:x0 + YOLO_CROP_SIZE]
            results = model.predict(crop, imgsz=YOLO_CROP_SIZE, conf=conf, verbose=False)
            if results and len(results[0].boxes):
                for box in results[0].boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    c = float(box.conf[0])
                    all_boxes.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, c))

    if not all_boxes:
        return []
    boxes = np.array(all_boxes)
    indices = cv2.dnn.NMSBoxes(
        [(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])) for b in boxes],
        [float(b[4]) for b in boxes], conf, YOLO_NMS_IOU)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2, c = boxes[i]
            dcx, dcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if is_on_grid(enh, dcx, dcy):
                detections.append(dict(tile_x=dcx, tile_y=dcy, conf=round(float(c), 3)))
    return detections

def detect_defects_yolo(grid, positions, mosaic, output_path, json_path):
    if not os.path.exists(YOLO_MODEL_PATH):
        log.error(f"YOLO model not found: {YOLO_MODEL_PATH}")
        return []
    model = _YOLO(YOLO_MODEL_PATH)
    log.info(f"Loaded YOLO model: {YOLO_MODEL_PATH}")
    shift_model, shift_device = None, None
    if os.path.exists(SHIFT_MODEL_PATH):
        try:
            shift_model, shift_device = load_shift_model()
            log.info(f"Loaded shift model: {SHIFT_MODEL_PATH}")
        except Exception as e:
            log.warning(f"Could not load shift model: {e}")
    h0, w0 = cv2.imread(grid[0][0], cv2.IMREAD_GRAYSCALE).shape
    img_w, img_h = w0, h0
    tile_centers = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if positions[r][c]:
                px, py = positions[r][c]
                tile_centers[(r, c)] = (px + img_w / 2, py + img_h / 2)

    all_defects = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            path = grid[r][c]
            if path is None or positions[r][c] is None:
                continue
            tile_dets = detect_tile_yolo(model, path)
            pos_x, pos_y = positions[r][c]

            if shift_model is not None and tile_dets:
                tile_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                tile_enh = enhance_tile(tile_gray)
                for d in tile_dets:
                    m = measure_shift(shift_model, shift_device,
                                      tile_enh, d["tile_x"], d["tile_y"])
                    if m:
                        d["shift_px"], d["shift_um"] = m
                del tile_gray, tile_enh
            for d in tile_dets:
                all_defects.append(dict(
                    x=int(pos_x + d["tile_x"]),
                    y=int(pos_y + d["tile_y"]),
                    conf=d["conf"],
                    shift_px=d.get("shift_px"),
                    shift_um=d.get("shift_um"),
                    tile=os.path.basename(path),
                    tile_row=r, tile_col=c,
                ))
            if tile_dets:
                log.info(f"  ({r},{c}) {os.path.basename(path)}: {len(tile_dets)} defects")

    before = len(all_defects)
    kept = []
    for d in all_defects:
        mx, my = d["x"], d["y"]
        tr, tc = d["tile_row"], d["tile_col"]
        my_cx, my_cy = tile_centers.get((tr, tc), (mx, my))
        my_dist = abs(mx - my_cx) + abs(my - my_cy)
        is_closest = True
        for (r2, c2), (cx2, cy2) in tile_centers.items():
            if (r2, c2) != (tr, tc) and abs(mx - cx2) + abs(my - cy2) < my_dist:
                is_closest = False
                break
        if is_closest:
            kept.append(d)
    all_defects = kept
    log.info(f"Voronoi dedup: {before} -> {len(all_defects)}")
    annotated = mosaic.copy()
    BOX = 22
    for d in all_defects:
        cx, cy = d["x"], d["y"]
        color = (0, 165, 255)
        cv2.rectangle(annotated, (cx - BOX, cy - BOX), (cx + BOX, cy + BOX), color, 2)
        if d.get("shift_um") is not None and d["shift_um"] > 0:
            label = f'{d["shift_um"]:.2f}um'
        elif d.get("shift_px") is not None and d["shift_px"] > 0:
            label = f'{d["shift_px"]:.1f}px'
        else:
            label = f'{d["conf"]:.0%}'
        cv2.putText(annotated, label, (cx - BOX, cy - BOX - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    log.info(f"Saved annotated mosaic: {output_path} ({len(all_defects)} defects)")
    defects_out = [dict(type='defect', x=d['x'], y=d['y'],
                        confidence=d['conf'],
                        shift_px=d.get('shift_px'),
                        shift_um=d.get('shift_um'),
                        tile=d['tile'])
                   for d in all_defects]
    with open(json_path, 'w') as f:
        json.dump({'total_defects': len(defects_out), 'defects': defects_out}, f, indent=2)
    log.info(f"Saved defect report: {json_path}")
    return all_defects

def process_directory(directory):
    dir_name = os.path.basename(os.path.normpath(directory))
    log.info(f"Processing: {dir_name}")
    t_start = time.time()
    grid = get_ordered_files(directory)
    features = extract_all_features(grid)
    transforms, periodic_step = compute_pairwise_transforms(grid, features)
    log.info(f"Computed {len(transforms)} pairwise transforms")
    transforms, detected_tile_rotations = rematch_rotated_pairs(grid, features, transforms)
    positions, rotations = global_alignment(grid, transforms, periodic_step=periodic_step)
    for (r, c), angle in detected_tile_rotations.items():
        rotations[r][c] = angle
        log.info(f"Tile ({r},{c}) rotation override: {angle:+.2f}deg")
    xs = [p[0] for row in positions for p in row if p]
    ys = [p[1] for row in positions for p in row if p]
    log.info(f"Position range: X=[{min(xs):.0f}, {max(xs):.0f}], Y=[{min(ys):.0f}, {max(ys):.0f}]")
    del features
    gc.collect()
    parent_dir = os.path.dirname(os.path.abspath(directory))
    stitch_path = os.path.join(parent_dir, f"{dir_name}_stitched.jpg")
    mosaic = composite_mosaic(grid, positions, stitch_path, rotations)
    defect_path = os.path.join(parent_dir, f"{dir_name}_defects.jpg")
    json_path = os.path.join(parent_dir, f"{dir_name}_defects.json")
    detect_defects_yolo(grid, positions, mosaic, defect_path, json_path)
    del mosaic
    gc.collect()
    elapsed = time.time() - t_start
    log.info(f"Directory {dir_name} done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    return stitch_path, defect_path, json_path

def main():
    directories = sys.argv[1:]
    for d in directories:
        if not os.path.isdir(d):
            log.error(f"Not a directory: {d}")
            sys.exit(1)
    log.info(f"Processing {len(directories)} directories with {NUM_WORKERS} CPU workers")
    results = []
    for d in directories:
        results.append(process_directory(d))
    log.info("All done! Output files:")
    for stitch_path, defect_path, json_path in results:
        log.info(f"Stitched: {stitch_path}")
        log.info(f"Defects: {defect_path}")
        log.info(f"Report: {json_path}")

if __name__ == '__main__':
    main()
