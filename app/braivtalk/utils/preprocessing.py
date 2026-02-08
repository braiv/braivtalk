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


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Enhanced face detection that returns both bounding boxes and facial landmarks.
    Returns face bounding boxes and landmarks for frames with faces, placeholders for frames without.
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i : i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks_list = []

    using_yolo = hasattr(fa, "detect")

    if upperbondrange != 0:
        print(f"🔍 Face detection with bbox_shift: {upperbondrange}")
    else:
        print("🔍 Face detection with default bbox")

    if using_yolo:
        print("🎯 Using YOLOv8 with facial landmarks for surgical positioning")
    else:
        print("🔧 Using SFD (bounding boxes only)")

    for fb in tqdm(batches, desc="Detecting faces"):
        if using_yolo:
            for frame in fb:
                det_bboxes, det_conf, _det_classid, landmarks = fa.detect(frame)

                if len(det_bboxes) > 0 and len(det_conf) > 0 and det_conf[0] > fa.conf_threshold:
                    bbox = det_bboxes[0]
                    x1, y1, x2, y2 = bbox.astype(int)

                    if upperbondrange != 0:
                        y1 = max(0, y1 + upperbondrange)

                    img_height, img_width = frame.shape[:2]
                    x1 = max(0, x1)
                    x2 = min(img_width, x2)
                    y1 = max(0, y1)
                    y2 = min(img_height, y2)

                    coords_list += [(x1, y1, x2, y2)]

                    if len(landmarks) > 0:
                        face_landmarks = landmarks[0]
                        landmark_points = [(float(pt[0]), float(pt[1])) for pt in face_landmarks]
                        landmarks_list += [landmark_points]
                    else:
                        landmarks_list += [None]
                else:
                    coords_list += [coord_placeholder]
                    landmarks_list += [None]
        else:
            bbox = fa.get_detections_for_batch(np.asarray(fb))

            for j, f in enumerate(bbox):
                if f is None:
                    coords_list += [coord_placeholder]
                    landmarks_list += [None]
                    continue

                x1, y1, x2, y2 = f
                if upperbondrange != 0:
                    y1 = max(0, y1 + upperbondrange)

                img_height, img_width = fb[j].shape[:2]
                x1 = max(0, x1)
                x2 = min(img_width, x2)
                y1 = max(0, y1)
                y2 = min(img_height, y2)

                coords_list += [(x1, y1, x2, y2)]
                landmarks_list += [None]

    print("=" * 80)
    print(f"✅ Face detection complete: {len(frames)} frames processed")
    face_count = sum(1 for coord in coords_list if coord != coord_placeholder)
    landmark_count = sum(1 for lm in landmarks_list if lm is not None)
    print(f"📊 Faces detected: {face_count}/{len(frames)} frames")
    if using_yolo:
        print(f"🎯 Landmarks extracted: {landmark_count}/{len(frames)} frames")
    print("=" * 80)

    return coords_list, frames, landmarks_list

