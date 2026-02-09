import numpy as np
import cv2
import pickle
import os
import torch
from tqdm import tqdm

from .face_detection import FaceAlignment, LandmarksType


# Initialize face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global face detector - will be initialized by init_face_detector()
fa = None


def init_face_detector(
    use_yolo=True,
    yolo_conf_threshold=0.5,
    yolo_temporal_weight=0.25,
    yolo_size_weight=0.30,
    yolo_center_weight=0.20,
    yolo_max_face_jump=0.3,
    yolo_primary_face_lock_threshold=10,
    yolo_primary_face_confidence_drop=0.8,
):
    """Initialize face detector with configurable parameters."""
    global fa

    if use_yolo:
        try:
            from .face_detection.api import YOLOv8_face

            # NOTE: Align with download_weights scripts
            model_path = "models/face_detection/weights/yoloface_8n.onnx"
            if os.path.exists(model_path):
                fa = YOLOv8_face(
                    path=model_path,
                    conf_thres=yolo_conf_threshold,
                    temporal_weight=yolo_temporal_weight,
                    size_weight=yolo_size_weight,
                    center_weight=yolo_center_weight,
                    max_face_jump=yolo_max_face_jump,
                    primary_face_lock_threshold=yolo_primary_face_lock_threshold,
                    primary_face_confidence_drop=yolo_primary_face_confidence_drop,
                )
                print(
                    f"Using YOLOv8 face detection (conf={yolo_conf_threshold}, "
                    f"temporal={yolo_temporal_weight}, size={yolo_size_weight}, "
                    f"lock_threshold={yolo_primary_face_lock_threshold})"
                )
                return fa
            else:
                print(f"WARNING: YOLOv8 model not found at {model_path}")
                raise FileNotFoundError("YOLOv8 model not available")
        except Exception as e:
            print(f"WARNING: YOLOv8 failed to load: {e}")
            print("Falling back to SFD face detection")

    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    print("Using SFD face detection")
    return fa


# Initialize with default parameters
fa = init_face_detector()

# marker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def _is_coord_placeholder(bbox, placeholder=None):
    """Safely check if a bbox is the coord_placeholder, handling numpy arrays."""
    if placeholder is None:
        placeholder = coord_placeholder
    if bbox is None:
        return True
    try:
        if hasattr(bbox, '__len__') and hasattr(bbox, '__getitem__'):
            import numpy as np
            if isinstance(bbox, np.ndarray):
                return np.allclose(bbox, placeholder)
        return tuple(bbox) == tuple(placeholder)
    except (TypeError, ValueError):
        return False


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def read_video_frames(video_path):
    """Read all frames from a video directly into memory via cv2.VideoCapture.
    Skips the FFmpeg extract-to-disk step for a major speedup on frame loading.
    Returns (frames, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    print(f"Reading {frame_count} frames directly from video...")
    for _ in tqdm(range(frame_count), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Read {len(frames)} frames at {fps:.1f} fps")
    return frames, fps


def _detect_single_frame(frame, upperbondrange, using_yolo):
    """Run face detection on a single frame. Returns (coord, landmark_points).

    When using YOLOv8, this routes through ``get_detections_for_batch`` so that
    the primary-face locking system (including ASD speaker selection) is honoured.
    """
    if using_yolo:
        # Detect ALL faces in the frame
        det_bboxes, det_conf, _det_classid, landmarks = fa.detect(frame)

        if len(det_bboxes) == 0 or len(det_conf) == 0:
            return coord_placeholder, None

        # Filter by confidence
        valid_indices = [i for i, c in enumerate(det_conf) if c > fa.conf_threshold]
        if not valid_indices:
            return coord_placeholder, None

        # Use the primary-face-aware selection (honours set_primary_face lock)
        selected_bbox = fa._select_best_face(
            det_bboxes[valid_indices],
            det_conf[valid_indices],
            landmarks[valid_indices] if len(landmarks) > 0 else None,
            frame.shape,
            fa.primary_face_bbox,  # temporal reference = locked primary face
        )

        if selected_bbox is None:
            return coord_placeholder, None

        x1, y1, x2, y2 = selected_bbox.astype(int)

        if upperbondrange != 0:
            y1 = max(0, y1 + upperbondrange)

        img_height, img_width = frame.shape[:2]
        x1 = max(0, x1)
        x2 = min(img_width, x2)
        y1 = max(0, y1)
        y2 = min(img_height, y2)

        # Extract landmarks for the selected face (find matching index)
        landmark_points = None
        if len(landmarks) > 0 and selected_bbox is not None:
            # Find which valid detection matches the selected bbox
            for idx in valid_indices:
                if np.allclose(det_bboxes[idx], selected_bbox, atol=1.0):
                    face_landmarks = landmarks[idx]
                    landmark_points = [(float(pt[0]), float(pt[1])) for pt in face_landmarks]
                    break
            # Fallback: use landmarks from the closest detection
            if landmark_points is None:
                best_dist = float('inf')
                best_lm_idx = valid_indices[0]
                sel_center = np.array([(selected_bbox[0]+selected_bbox[2])/2, (selected_bbox[1]+selected_bbox[3])/2])
                for idx in valid_indices:
                    det_center = np.array([(det_bboxes[idx][0]+det_bboxes[idx][2])/2, (det_bboxes[idx][1]+det_bboxes[idx][3])/2])
                    dist = np.linalg.norm(sel_center - det_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_lm_idx = idx
                face_landmarks = landmarks[best_lm_idx]
                landmark_points = [(float(pt[0]), float(pt[1])) for pt in face_landmarks]

        return (x1, y1, x2, y2), landmark_points
    else:
        bbox_result = fa.get_detections_for_batch(np.asarray([frame]))
        f = bbox_result[0]
        if f is None:
            return coord_placeholder, None

        x1, y1, x2, y2 = f
        if upperbondrange != 0:
            y1 = max(0, y1 + upperbondrange)

        img_height, img_width = frame.shape[:2]
        x1 = max(0, x1)
        x2 = min(img_width, x2)
        y1 = max(0, y1)
        y2 = min(img_height, y2)

        return (x1, y1, x2, y2), None


def _interpolate_bbox(bbox1, bbox2, alpha):
    """Linearly interpolate between two bounding boxes."""
    return (
        int(round(bbox1[0] + (bbox2[0] - bbox1[0]) * alpha)),
        int(round(bbox1[1] + (bbox2[1] - bbox1[1]) * alpha)),
        int(round(bbox1[2] + (bbox2[2] - bbox1[2]) * alpha)),
        int(round(bbox1[3] + (bbox2[3] - bbox1[3]) * alpha)),
    )


def _interpolate_landmarks(lm1, lm2, alpha):
    """Linearly interpolate between two sets of landmark points."""
    if lm1 is None or lm2 is None:
        return lm1 if alpha < 0.5 else lm2
    return [
        (pt1[0] + (pt2[0] - pt1[0]) * alpha, pt1[1] + (pt2[1] - pt1[1]) * alpha)
        for pt1, pt2 in zip(lm1, lm2)
    ]


def get_landmark_and_bbox(frames, upperbondrange=0, detection_interval=1):
    """
    Face detection returning bounding boxes and facial landmarks.

    Args:
        frames: List of numpy arrays (pre-loaded video frames).
        upperbondrange: Bounding box shift value.
        detection_interval: Run detection every N frames (1=every frame).
            When >1, intermediate frames use linear interpolation for speed.

    Returns:
        (coords_list, landmarks_list) - bounding boxes and landmarks per frame.
    """
    n = len(frames)
    using_yolo = hasattr(fa, "detect")

    if upperbondrange != 0:
        print(f"Face detection with bbox_shift: {upperbondrange}")
    else:
        print("Face detection with default bbox")

    if using_yolo:
        print("Using YOLOv8 with facial landmarks for surgical positioning")
    else:
        print("Using SFD (bounding boxes only)")

    if detection_interval > 1:
        print(f"Sparse detection enabled: every {detection_interval} frames with interpolation")

    coords_list = [None] * n
    landmarks_list = [None] * n
    detected_keyframes = n  # track for summary

    if detection_interval <= 1:
        # Original behavior - detect every frame
        for i, frame in enumerate(tqdm(frames, desc="Detecting faces")):
            coord, landmark = _detect_single_frame(frame, upperbondrange, using_yolo)
            coords_list[i] = coord
            landmarks_list[i] = landmark
    else:
        # Sparse detection with interpolation
        keyframe_indices = list(range(0, n, detection_interval))
        if keyframe_indices[-1] != n - 1:
            keyframe_indices.append(n - 1)
        detected_keyframes = len(keyframe_indices)

        # Run detection only on keyframes
        key_results = {}
        for idx in tqdm(keyframe_indices, desc=f"Detecting faces (every {detection_interval} frames)"):
            coord, landmark = _detect_single_frame(frames[idx], upperbondrange, using_yolo)
            key_results[idx] = (coord, landmark)

        # Interpolate for all frames between keyframes
        for k in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[k]
            end_idx = keyframe_indices[k + 1]

            start_coord, start_lm = key_results[start_idx]
            end_coord, end_lm = key_results[end_idx]

            # Set start keyframe values
            coords_list[start_idx] = start_coord
            landmarks_list[start_idx] = start_lm

            start_has_face = not _is_coord_placeholder(start_coord)
            end_has_face = not _is_coord_placeholder(end_coord)

            for idx in range(start_idx + 1, end_idx):
                alpha = (idx - start_idx) / (end_idx - start_idx)
                if start_has_face and end_has_face:
                    # Both keyframes have a face - interpolate smoothly
                    coords_list[idx] = _interpolate_bbox(start_coord, end_coord, alpha)
                    landmarks_list[idx] = _interpolate_landmarks(start_lm, end_lm, alpha)
                elif start_has_face:
                    # Face leaving - hold last known position
                    coords_list[idx] = start_coord
                    landmarks_list[idx] = start_lm
                elif end_has_face:
                    # Face entering - hold next known position
                    coords_list[idx] = end_coord
                    landmarks_list[idx] = end_lm
                else:
                    coords_list[idx] = coord_placeholder
                    landmarks_list[idx] = None

        # Set last keyframe
        last_idx = keyframe_indices[-1]
        coords_list[last_idx] = key_results[last_idx][0]
        landmarks_list[last_idx] = key_results[last_idx][1]

    print("=" * 80)
    print(f"Face detection complete: {n} frames processed")
    if detection_interval > 1:
        print(f"  Keyframes detected: {detected_keyframes}, interpolated: {n - detected_keyframes}")
    face_count = sum(1 for coord in coords_list if not _is_coord_placeholder(coord))
    landmark_count = sum(1 for lm in landmarks_list if lm is not None)
    print(f"Faces detected: {face_count}/{n} frames")
    if using_yolo:
        print(f"Landmarks extracted: {landmark_count}/{n} frames")
    print("=" * 80)

    return coords_list, landmarks_list

