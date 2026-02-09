"""
ActiveSpeakerDetector -- integration wrapper for LR-ASD in the braivtalk pipeline.

This module bridges LR-ASD's audio-visual model with the existing braivtalk
inference pipeline.  It reuses YOLOv8 face bounding boxes (no S3FD needed),
extracts MFCC features from the audio track, and returns per-frame speaking
scores + a boolean mask.
"""

import math
import os
import subprocess
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import torch

try:
    import python_speech_features
except ImportError:
    python_speech_features = None

from .asd_model import ASD


def _compute_iou(boxA, boxB):
    """Compute intersection-over-union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _is_placeholder(bbox, placeholder=(0, 0, 0, 0)):
    """Safely check if a bbox is a placeholder, handling numpy arrays."""
    if bbox is None:
        return True
    try:
        if isinstance(bbox, np.ndarray):
            return np.allclose(bbox, placeholder)
        return tuple(bbox) == placeholder
    except (TypeError, ValueError):
        return False


def _nms_faces(bboxes, confs, conf_threshold, iou_threshold=0.5):
    """
    Per-frame Non-Maximum Suppression: keep only one detection per face.

    Returns list of indices into bboxes/confs to keep.
    """
    # Filter by confidence first
    indices = [i for i in range(len(confs)) if confs[i] > conf_threshold]
    if not indices:
        return []

    # Sort by confidence (highest first)
    indices.sort(key=lambda i: confs[i], reverse=True)

    kept = []
    suppressed = set()
    for i in indices:
        if i in suppressed:
            continue
        kept.append(i)
        for j in indices:
            if j in suppressed or j == i:
                continue
            if _compute_iou(bboxes[i], bboxes[j]) > iou_threshold:
                suppressed.add(j)

    return kept


# Duration windows used for multi-window scoring (matches LR-ASD best practice).
# Duplicates bias toward shorter windows for responsiveness.
_DURATION_SET = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]


class ActiveSpeakerDetector:
    """
    Wraps LR-ASD for use inside the braivtalk lip-sync pipeline.

    Usage::

        asd = ActiveSpeakerDetector("models/lr_asd/pretrain_AVA.model")
        scores, mask = asd.detect_speaking_frames(frames, bboxes, audio_path)
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        if python_speech_features is None:
            raise ImportError(
                "python_speech_features is required for ASD. "
                "Install it with: pip install python_speech_features"
            )

        self.device = device if torch.cuda.is_available() else "cpu"

        # Build model and load weights
        self.asd = ASD()
        self.asd.loadParameters(model_path, device=self.device)
        self.asd = self.asd.to(self.device)
        self.asd.eval()

        param_count = sum(p.numel() for p in self.asd.parameters()) / 1e6
        print(f"[ASD] LR-ASD loaded from {model_path} ({param_count:.2f}M params, device={self.device})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_speaking_frames(
        self,
        frames: List[np.ndarray],
        face_bboxes: List[tuple],
        audio_path: str,
        fps: float = 25.0,
        threshold: float = 0.0,
        hold_seconds: float = 0.3,
        gap_fill_seconds: float = 0.4,
    ) -> Tuple[List[float], List[bool]]:
        """
        Determine which frames contain an actively speaking face.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR video frames (from ``read_video_frames``).
        face_bboxes : list[tuple]
            Per-frame ``(x1, y1, x2, y2)`` from ``get_landmark_and_bbox``.
            Frames with ``coord_placeholder`` (no face) are scored as 0.
        audio_path : str
            Path to the audio file (any format FFmpeg can read).
        fps : float
            Video frame rate.
        threshold : float
            Score above which a frame is considered "speaking".

        Returns
        -------
        scores : list[float]
            Raw speaking score per frame.
        speaking_mask : list[bool]
            ``True`` where the face is speaking.
        """
        n_frames = len(frames)
        faces_with_bbox = sum(1 for b in face_bboxes if b != (0.0, 0.0, 0.0, 0.0) and b is not None)
        print(f"[ASD] Starting active speaker detection on {n_frames} frames "
              f"({faces_with_bbox} with face bboxes, fps={fps})")

        # --- 1. Extract 16 kHz mono WAV and compute MFCC -----------------
        print(f"[ASD] Extracting MFCC from: {audio_path}")
        audio_features = self._extract_mfcc(audio_path)
        if audio_features is None:
            print("[ASD] WARNING: Could not extract audio -- marking all frames as speaking")
            return [1.0] * n_frames, [True] * n_frames
        print(f"[ASD] MFCC features: shape={audio_features.shape} "
              f"(~{audio_features.shape[0] / 100:.1f}s of audio at 100 MFCC-fps)")

        # --- 2. Prepare visual features (grayscale 112x112 face crops) ----
        print("[ASD] Preparing visual features (grayscale 112x112 face crops)...")
        video_features = self._prepare_visual_features(frames, face_bboxes)
        blank_crops = np.sum(np.all(video_features == 0, axis=(1, 2)))
        print(f"[ASD] Visual features: shape={video_features.shape}, "
              f"blank crops (no face): {blank_crops}/{video_features.shape[0]}")

        # --- 2b. Resample visual features to 25fps -----------------------
        # LR-ASD was trained on 25fps video. The audio encoder produces
        # 25 time-steps per second (from 100 MFCC frames), so the visual
        # stream must also deliver exactly 25 frames per second for the
        # fusion layer to concatenate them.
        MODEL_FPS = 25.0
        if abs(fps - MODEL_FPS) > 0.5:
            orig_len = video_features.shape[0]
            target_len = int(round(orig_len * MODEL_FPS / fps))
            indices = np.linspace(0, orig_len - 1, target_len).astype(int)
            video_features = video_features[indices]
            print(f"[ASD] Resampled visual features from {orig_len} frames "
                  f"({fps:.1f}fps) -> {target_len} frames ({MODEL_FPS}fps)")
        else:
            print(f"[ASD] Video is already ~{MODEL_FPS}fps, no resampling needed")

        # --- 3. Align lengths (audio @ 100 fps, video @ 25fps) -----------
        duration = min(
            (audio_features.shape[0] - audio_features.shape[0] % 4) / 100,
            video_features.shape[0] / MODEL_FPS,
        )
        if duration <= 0:
            print("[ASD] WARNING: Duration too short for ASD -- marking all frames as speaking")
            return [1.0] * n_frames, [True] * n_frames

        audio_len = int(round(duration * 100))
        video_len = int(round(duration * MODEL_FPS))
        audio_features = audio_features[:audio_len, :]
        video_features = video_features[:video_len, :, :]
        print(f"[ASD] Aligned duration: {duration:.2f}s "
              f"(audio={audio_len} MFCC frames, video={video_len} frames @ {MODEL_FPS}fps)")

        # --- 4. Multi-window inference ------------------------------------
        # All windowing uses MODEL_FPS (25) since visual features are now
        # resampled to 25fps, matching the audio encoder's output rate.
        print(f"[ASD] Running multi-window inference with {len(set(_DURATION_SET))} "
              f"unique durations: {sorted(set(_DURATION_SET))}s ...")
        all_scores = []
        for window_dur in _DURATION_SET:
            batch_size = int(math.ceil(duration / window_dur))
            scores = []
            for i in range(batch_size):
                a_start = i * window_dur * 100
                a_end = (i + 1) * window_dur * 100
                v_start = int(i * window_dur * MODEL_FPS)
                v_end = int((i + 1) * window_dur * MODEL_FPS)

                inputA = torch.FloatTensor(audio_features[a_start:a_end, :]).unsqueeze(0).to(self.device)
                inputV = torch.FloatTensor(video_features[v_start:v_end, :, :]).unsqueeze(0).to(self.device)

                # Skip empty slices
                if inputA.shape[1] == 0 or inputV.shape[1] == 0:
                    continue

                embedA = self.asd.model.forward_audio_frontend(inputA)
                embedV = self.asd.model.forward_visual_frontend(inputV)
                out = self.asd.model.forward_audio_visual_backend(embedA, embedV)
                score = self.asd.lossAV.forward(out, labels=None)
                scores.extend(score)
            all_scores.append(scores)

        # Average across windows, rounding to 1 decimal
        min_len = min(len(s) for s in all_scores) if all_scores else 0
        if min_len == 0:
            print("[ASD] WARNING: No scores produced -- marking all frames as speaking")
            return [1.0] * n_frames, [True] * n_frames

        all_scores = [s[:min_len] for s in all_scores]
        avg_scores = np.round(np.mean(np.array(all_scores), axis=0), 1).astype(float)
        print(f"[ASD] Raw scores: {min_len} values, "
              f"range=[{avg_scores.min():.1f}, {avg_scores.max():.1f}], "
              f"mean={avg_scores.mean():.2f}")

        # --- 5. Map ASD scores back to original frame count ---------------
        # ASD produces one score per video_len frame; we need n_frames scores.
        per_frame_scores = self._map_scores_to_frames(avg_scores, n_frames, fps)
        raw_mask = [s > threshold for s in per_frame_scores]

        # --- 6. Temporal smoothing -----------------------------------------
        # Raw per-frame thresholding causes flicker: the AI mouth appears and
        # disappears between words.  We apply two passes:
        #   a) Hold: once speaking starts, keep it for at least `hold_seconds`
        #      even if scores dip below threshold.
        #   b) Gap-fill: short silent gaps (< `gap_fill_seconds`) between two
        #      speaking segments are filled in as speaking.
        # Values are configurable via --asd_hold_seconds / --asd_gap_fill_seconds.
        hold_frames = int(round(hold_seconds * fps))
        gap_fill_frames = int(round(gap_fill_seconds * fps))

        # Pass (a): Hold -- extend every speaking onset by hold_frames
        speaking_mask = list(raw_mask)
        frames_since_last_speaking = hold_frames + 1  # start as "not holding"
        for i in range(n_frames):
            if raw_mask[i]:
                frames_since_last_speaking = 0
                speaking_mask[i] = True
            else:
                frames_since_last_speaking += 1
                if frames_since_last_speaking <= hold_frames:
                    speaking_mask[i] = True

        # Pass (b): Gap-fill -- if a short silent gap is between two speaking
        # segments, fill it in.
        speaking_mask = self._fill_short_gaps(speaking_mask, gap_fill_frames)

        raw_speaking = sum(raw_mask)
        smoothed_speaking = sum(speaking_mask)
        print("=" * 60)
        print(f"[ASD] RESULTS: {n_frames} total frames")
        print(f"[ASD]   Speaking (raw):      {raw_speaking} frames "
              f"({100 * raw_speaking / n_frames:.1f}%)")
        print(f"[ASD]   Speaking (smoothed): {smoothed_speaking} frames "
              f"({100 * smoothed_speaking / n_frames:.1f}%)")
        silent_count = n_frames - smoothed_speaking
        print(f"[ASD]   Silent:              {silent_count} frames "
              f"({100 * silent_count / n_frames:.1f}%)")
        print(f"[ASD]   Threshold: {threshold}, "
              f"hold: {hold_seconds}s ({hold_frames}f), "
              f"gap-fill: {gap_fill_seconds}s ({gap_fill_frames}f)")

        # Show score distribution for debugging
        scores_arr = np.array(per_frame_scores)
        for bucket_min, bucket_max, label in [
            (-999, -2.0, "strong silent (<-2)"),
            (-2.0, -0.5, "likely silent (-2 to -0.5)"),
            (-0.5, 0.5, "borderline (-0.5 to 0.5)"),
            (0.5, 2.0, "likely speaking (0.5 to 2)"),
            (2.0, 999, "strong speaking (>2)"),
        ]:
            count = int(np.sum((scores_arr > bucket_min) & (scores_arr <= bucket_max)))
            if count > 0:
                print(f"[ASD]   {label}: {count} frames")

        # Log first/last few frame scores for spot-checking
        preview_n = min(10, n_frames)
        head_scores = [f"{per_frame_scores[i]:+.1f}" for i in range(preview_n)]
        tail_scores = [f"{per_frame_scores[-(preview_n - i)]:+.1f}" for i in range(preview_n)]
        print(f"[ASD]   First {preview_n} scores: [{', '.join(head_scores)}]")
        print(f"[ASD]   Last  {preview_n} scores: [{', '.join(tail_scores)}]")
        print("=" * 60)

        return per_frame_scores, speaking_mask

    # ------------------------------------------------------------------
    # Multi-face speaker selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_speaker_face(
        self,
        frames: List[np.ndarray],
        audio_path: str,
        face_detector,
        fps: float = 25.0,
        sample_seconds: float = 6.0,
        conf_threshold: float = 0.5,
        debug_dir: str = None,
    ) -> "np.ndarray | None":
        """
        Detect ALL faces, run ASD on each, return the bbox of the speaker.

        Instead of relying on YOLOv8's primary-face heuristic, this method:
        1. Samples several windows of video (beginning, middle, end)
        2. Detects ALL faces with per-frame NMS to deduplicate
        3. Groups detections into face tracks via IOU
        4. Runs ASD on each track
        5. Returns the average bbox of the track with the highest speaking score

        Parameters
        ----------
        frames : list[np.ndarray]
            Full video frames.
        audio_path : str
            Path to the audio file.
        face_detector
            The YOLOv8_face instance (must have a ``.detect()`` method).
        fps : float
            Video frame rate.
        sample_seconds : float
            How many seconds per sample window.
        conf_threshold : float
            Minimum detection confidence for a face.
        debug_dir : str or None
            If set, save annotated debug PNGs showing detected faces,
            scores, and the selected speaker to this directory.

        Returns
        -------
        np.ndarray or None
            The speaker's average bbox ``[x1, y1, x2, y2]``, or ``None``
            if no speaker could be determined.
        """
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
        n_frames = len(frames)
        MODEL_FPS = 25.0
        window_size = int(sample_seconds * fps)

        # --- 1. Extract MFCC for the full audio (reused across windows) ---
        audio_features = self._extract_mfcc(audio_path)
        if audio_features is None:
            print("[ASD-SELECT] WARNING: Could not extract audio")
            return None

        # --- 2. Try multiple sample windows to find active speech ---------
        # Try 25%, 50%, 75% positions in the video
        candidate_positions = [0.25, 0.50, 0.75]
        best_overall_track = None
        best_overall_score = float("-inf")

        for pos_frac in candidate_positions:
            mid = int(n_frames * pos_frac)
            start_idx = max(0, mid - window_size // 2)
            end_idx = min(n_frames, start_idx + window_size)
            if end_idx - start_idx < int(fps):
                continue  # too short

            sample_frames = frames[start_idx:end_idx]
            print(f"[ASD-SELECT] Window @{pos_frac:.0%}: frames {start_idx}-{end_idx} "
                  f"({len(sample_frames)} frames, ~{len(sample_frames)/fps:.1f}s)")

            # --- 2a. Detect ALL faces with per-frame NMS ------------------
            detect_interval = 5
            all_detections = []
            for i in range(0, len(sample_frames), detect_interval):
                det_bboxes, det_conf, _, _ = face_detector.detect(sample_frames[i])
                if len(det_bboxes) == 0:
                    continue

                # Per-frame NMS: keep only the best detection per cluster
                kept = _nms_faces(det_bboxes, det_conf, conf_threshold, iou_threshold=0.5)
                for idx in kept:
                    all_detections.append((i, det_bboxes[idx].astype(int), float(det_conf[idx])))

            if not all_detections:
                print(f"[ASD-SELECT]   No faces detected in this window")
                continue

            # --- 2b. Group into tracks ------------------------------------
            tracks = self._build_face_tracks(all_detections, iou_threshold=0.4)

            # Merge tracks that represent the same face (similar avg bbox)
            tracks = self._merge_similar_tracks(tracks, iou_threshold=0.5)

            n_faces = len(tracks)
            total_dets = sum(len(t) for t in tracks)
            print(f"[ASD-SELECT]   {n_faces} unique face(s) found ({total_dets} detections)")

            if n_faces == 0:
                continue

            # --- 2c. Trim audio to this window ----------------------------
            audio_start = int(start_idx / fps * 100)
            audio_end = int(end_idx / fps * 100)
            window_audio = audio_features[audio_start:audio_end]

            # --- 2d. Score each track with ASD ----------------------------
            window_results = []  # (track_idx, score, bbox, n_detections) for debug
            window_best_idx = None
            for t_idx, track in enumerate(tracks):
                track_score = self._score_track(
                    track, sample_frames, window_audio, fps, MODEL_FPS,
                )
                if track_score is None:
                    continue

                track_bbox = np.mean([d[1] for d in track], axis=0).astype(int)
                area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                print(f"[ASD-SELECT]   Face {t_idx}: score={track_score:+.2f}, "
                      f"detections={len(track)}, bbox~{track_bbox.tolist()}, area={area}")

                window_results.append((t_idx, track_score, track_bbox, len(track)))

                if track_score > best_overall_score:
                    best_overall_score = track_score
                    best_overall_track = track
                    window_best_idx = (pos_frac, t_idx)

            # --- 2e. Save debug image for this window ----------------------
            if debug_dir is not None and window_results:
                mid_frame_idx = len(sample_frames) // 2
                self._save_debug_window(
                    sample_frames[mid_frame_idx].copy(),
                    window_results,
                    pos_frac,
                    start_idx + mid_frame_idx,
                    debug_dir,
                )

        # --- 3. Return the winner -----------------------------------------
        if best_overall_track is None:
            print("[ASD-SELECT] WARNING: No speaker found across all windows")
            return None

        winner_bbox = np.mean([d[1] for d in best_overall_track], axis=0).astype(int)
        print(f"[ASD-SELECT] SPEAKER SELECTED: score={best_overall_score:+.2f}, "
              f"bbox~{winner_bbox.tolist()}")

        # Save a final "selected speaker" debug image
        if debug_dir is not None:
            # Use a frame near the best track
            best_track_frame = best_overall_track[len(best_overall_track) // 2][0]
            # Locate which window this came from
            if window_best_idx is not None:
                best_pos = window_best_idx[0]
                mid = int(n_frames * best_pos)
                ws_start = max(0, mid - window_size // 2)
                abs_frame_idx = ws_start + best_track_frame
            else:
                abs_frame_idx = min(best_track_frame, n_frames - 1)
            abs_frame_idx = min(abs_frame_idx, n_frames - 1)
            self._save_debug_speaker(
                frames[abs_frame_idx].copy(),
                winner_bbox,
                best_overall_score,
                abs_frame_idx,
                debug_dir,
            )

        return winner_bbox

    # ------------------------------------------------------------------
    # Dynamic per-frame active speaker detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_per_frame_active_speaker(
        self,
        frames: List[np.ndarray],
        audio_path: str,
        face_detector,
        fps: float = 25.0,
        detection_interval: int = 5,
        conf_threshold: float = 0.5,
        threshold: float = 0.0,
        hold_seconds: float = 0.3,
        gap_fill_seconds: float = 0.4,
        window_seconds: float = 6.0,
        hop_seconds: float = 3.0,
        debug_dir: str = None,
    ) -> Tuple[List["np.ndarray | None"], List[bool]]:
        """
        Identify the active speaker per frame across the whole video.

        Unlike ``select_speaker_face`` which picks ONE speaker for the whole
        video, this method dynamically assigns the active speaker per frame,
        so lip-sync follows whoever is currently speaking.

        Approach:
        1. Detect ALL faces densely across the video
        2. Build face tracks (group same person across frames via IOU)
        3. Slide ASD scoring windows across the video, scoring each track
        4. For each frame, the track with the highest speaking score wins
        5. Apply temporal smoothing to prevent rapid switching

        Returns
        -------
        speaker_bboxes : list[np.ndarray | None]
            Per-frame bbox ``[x1, y1, x2, y2]`` of the active speaker,
            or ``None`` for frames where nobody is speaking.
        speaking_mask : list[bool]
            ``True`` where someone is speaking.
        """
        n_frames = len(frames)
        MODEL_FPS = 25.0
        coord_placeholder = (0, 0, 0, 0)

        print(f"[ASD-DYN] Starting dynamic per-frame speaker detection "
              f"({n_frames} frames, {fps:.1f} fps)")

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

        # --- 1. Extract MFCC for the full audio (one-time) ---------------
        audio_features = self._extract_mfcc(audio_path)
        if audio_features is None:
            print("[ASD-DYN] WARNING: Could not extract audio -- "
                  "marking all frames as speaking (no speaker switch)")
            return [None] * n_frames, [True] * n_frames

        # --- 2. Detect ALL faces densely across the video ----------------
        print(f"[ASD-DYN] Detecting faces every {detection_interval} frames...")
        all_detections = []  # (abs_frame_idx, bbox, conf)
        face_landmarks_map = {}  # abs_frame_idx -> {bbox_key: landmarks}

        for fi in range(0, n_frames, detection_interval):
            det_bboxes, det_conf, _, det_landmarks = face_detector.detect(frames[fi])
            if len(det_bboxes) == 0:
                continue
            kept = _nms_faces(det_bboxes, det_conf, conf_threshold, iou_threshold=0.5)
            for idx in kept:
                bbox = det_bboxes[idx].astype(int)
                all_detections.append((fi, bbox, float(det_conf[idx])))
                # Store landmarks keyed by frame + bbox center
                if len(det_landmarks) > idx:
                    key = (fi, int(bbox[0] + bbox[2]) // 2, int(bbox[1] + bbox[3]) // 2)
                    face_landmarks_map[key] = det_landmarks[idx]

        if not all_detections:
            print("[ASD-DYN] No faces detected in the video")
            return [None] * n_frames, [False] * n_frames

        # --- 3. Build face tracks ----------------------------------------
        tracks = self._build_face_tracks(all_detections, iou_threshold=0.4)
        tracks = self._merge_similar_tracks(tracks, iou_threshold=0.5)
        # Filter short tracks (noise)
        min_track_len = max(3, int(fps / detection_interval))
        tracks = [t for t in tracks if len(t) >= min_track_len]

        if not tracks:
            print("[ASD-DYN] No sustained face tracks found")
            return [None] * n_frames, [False] * n_frames

        print(f"[ASD-DYN] Built {len(tracks)} face tracks "
              f"(min {min_track_len} detections)")

        # For each track, compute its average bbox (for logging / debug)
        track_avg_bboxes = []
        for track in tracks:
            avg_bbox = np.mean([d[1] for d in track], axis=0).astype(int)
            track_avg_bboxes.append(avg_bbox)
            area = (avg_bbox[2] - avg_bbox[0]) * (avg_bbox[3] - avg_bbox[1])
            print(f"[ASD-DYN]   Track: {len(track)} dets, "
                  f"bbox~{avg_bbox.tolist()}, area={area}")

        # --- 4. Build per-frame bbox for each track (with interpolation) --
        track_per_frame_bboxes = []
        for track in tracks:
            bboxes = [None] * n_frames
            for (fidx, bbox, _conf) in track:
                bboxes[fidx] = bbox
            bboxes = self._fill_track_gaps(bboxes)
            # Convert None->placeholder for bboxes that couldn't be filled
            bboxes = [b if b is not None else coord_placeholder for b in bboxes]
            # Linear interpolation between detected frames
            bboxes = self._interpolate_bboxes(bboxes, coord_placeholder)
            track_per_frame_bboxes.append(bboxes)

        # --- 5. Score each track with ASD in sliding windows -------------
        window_frames = int(window_seconds * fps)
        hop_frames = int(hop_seconds * fps)

        # Per-track accumulator: sum of scores and count per frame
        track_score_sum = [np.zeros(n_frames) for _ in range(len(tracks))]
        track_score_count = [np.zeros(n_frames) for _ in range(len(tracks))]

        n_windows = 0
        for win_start in range(0, n_frames, hop_frames):
            win_end = min(win_start + window_frames, n_frames)
            if win_end - win_start < int(fps):
                continue  # Window too short
            n_windows += 1

            # Audio slice for this window
            audio_start = int(win_start / fps * 100)
            audio_end = int(win_end / fps * 100)
            window_audio = audio_features[audio_start:audio_end]
            if window_audio.shape[0] == 0:
                continue

            sample_frames = frames[win_start:win_end]

            # Score each track in this window
            for ti, track in enumerate(tracks):
                # Build local bboxes for this window
                local_bboxes = track_per_frame_bboxes[ti][win_start:win_end]
                # Check if track has enough presence in this window
                valid = sum(1 for b in local_bboxes if not _is_placeholder(b))
                if valid < len(local_bboxes) * 0.3:
                    continue  # Track barely visible in this window

                # Build a mini-track for _score_track
                mini_track = []
                for li, bbox in enumerate(local_bboxes):
                    if not _is_placeholder(bbox):
                        mini_track.append((li, np.array(bbox), 1.0))

                if len(mini_track) < 3:
                    continue

                score = self._score_track(
                    mini_track, sample_frames, window_audio, fps, MODEL_FPS,
                )
                if score is None:
                    continue

                # Assign this score to all frames in the window
                for fi in range(win_start, win_end):
                    track_score_sum[ti][fi] += score
                    track_score_count[ti][fi] += 1

        print(f"[ASD-DYN] Scored {len(tracks)} tracks across {n_windows} windows")

        # --- 6. Compute average score per track per frame ----------------
        track_avg_scores = []
        for ti in range(len(tracks)):
            avg = np.zeros(n_frames)
            for fi in range(n_frames):
                if track_score_count[ti][fi] > 0:
                    avg[fi] = track_score_sum[ti][fi] / track_score_count[ti][fi]
                else:
                    avg[fi] = -10.0  # No data -> treat as strongly silent
            track_avg_scores.append(avg)
            print(f"[ASD-DYN]   Track {ti}: avg_score={avg[avg > -10].mean():.2f}, "
                  f"max={avg.max():.2f}, "
                  f"frames_above_threshold={int((avg > threshold).sum())}")

        # --- 7. Per frame, pick the track with the highest score ---------
        speaker_bboxes: List["np.ndarray | None"] = [None] * n_frames
        raw_mask = [False] * n_frames

        for fi in range(n_frames):
            best_track = -1
            best_score = threshold
            for ti in range(len(tracks)):
                s = track_avg_scores[ti][fi]
                if s > best_score and not _is_placeholder(track_per_frame_bboxes[ti][fi]):
                    best_score = s
                    best_track = ti

            if best_track >= 0:
                bbox = track_per_frame_bboxes[best_track][fi]
                speaker_bboxes[fi] = np.array(bbox)
                raw_mask[fi] = True

        # --- 8. Temporal smoothing ---------------------------------------
        hold_frames = int(round(hold_seconds * fps))
        gap_fill_frames = int(round(gap_fill_seconds * fps))

        speaking_mask = list(raw_mask)
        frames_since_speaking = hold_frames + 1
        last_speaker_bbox = None
        for i in range(n_frames):
            if raw_mask[i]:
                frames_since_speaking = 0
                last_speaker_bbox = speaker_bboxes[i]
            else:
                frames_since_speaking += 1
                if frames_since_speaking <= hold_frames and last_speaker_bbox is not None:
                    speaking_mask[i] = True
                    if speaker_bboxes[i] is None:
                        speaker_bboxes[i] = last_speaker_bbox

        speaking_mask = self._fill_short_gaps(speaking_mask, gap_fill_frames)

        # Fill bboxes for ALL frames (not just speaking ones) so every frame
        # has a face target.  The speaking_mask separately controls whether
        # lip-sync inference runs; the bbox just tells YOLO which face to
        # track.  Forward-fill first, then backward-fill any leading gap.
        last_known = None
        for i in range(n_frames):
            if speaker_bboxes[i] is not None:
                last_known = speaker_bboxes[i]
            elif last_known is not None:
                speaker_bboxes[i] = last_known
        # Backward-fill any frames before the first detection
        first_known = None
        for i in range(n_frames - 1, -1, -1):
            if speaker_bboxes[i] is not None and first_known is None:
                first_known = speaker_bboxes[i]
            elif speaker_bboxes[i] is None and first_known is not None:
                speaker_bboxes[i] = first_known

        # --- 9. Log results ----------------------------------------------
        speaking_count = sum(speaking_mask)
        silent_count = n_frames - speaking_count
        print("=" * 60)
        print(f"[ASD-DYN] RESULTS: {n_frames} total frames")
        print(f"[ASD-DYN]   Speaking: {speaking_count} frames "
              f"({100 * speaking_count / n_frames:.1f}%)")
        print(f"[ASD-DYN]   Silent:   {silent_count} frames "
              f"({100 * silent_count / n_frames:.1f}%)")

        # Count per-track frame assignments
        track_frame_counts = [0] * len(tracks)
        for fi in range(n_frames):
            if speaker_bboxes[fi] is not None:
                for ti in range(len(tracks)):
                    bbox = track_per_frame_bboxes[ti][fi]
                    if not _is_placeholder(bbox) and np.allclose(speaker_bboxes[fi], bbox):
                        track_frame_counts[ti] += 1
                        break
        for ti in range(len(tracks)):
            print(f"[ASD-DYN]   Track {ti}: assigned {track_frame_counts[ti]} frames")
        print("=" * 60)

        # --- 10. Save debug image ----------------------------------------
        if debug_dir is not None and len(tracks) > 0:
            # Save a debug image at 50% point
            mid = n_frames // 2
            vis = frames[mid].copy()
            for ti in range(len(tracks)):
                bbox = track_per_frame_bboxes[ti][mid]
                if _is_placeholder(bbox):
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                is_speaker = (speaker_bboxes[mid] is not None and
                              np.allclose(speaker_bboxes[mid], bbox))
                color = (0, 220, 0) if is_speaker else (0, 0, 220)
                thickness = 4 if is_speaker else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                score = track_avg_scores[ti][mid]
                label = f"Track {ti}: {score:+.2f}"
                if is_speaker:
                    label = "SPEAKER " + label
                cv2.putText(vis, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(os.path.join(debug_dir, f"asd_dynamic_frame{mid:06d}.png"), vis)

        return speaker_bboxes, speaking_mask

    @staticmethod
    def _interpolate_bboxes(bboxes, placeholder):
        """Linear interpolation of bboxes between detected frames."""
        n = len(bboxes)
        result = list(bboxes)

        # Find segments of placeholders and interpolate
        i = 0
        while i < n:
            if _is_placeholder(result[i], placeholder):
                # Find the start and end of this gap
                gap_start = i
                while i < n and _is_placeholder(result[i], placeholder):
                    i += 1
                gap_end = i

                # Get bounding bboxes
                before = result[gap_start - 1] if gap_start > 0 and not _is_placeholder(result[gap_start - 1], placeholder) else None
                after = result[gap_end] if gap_end < n and not _is_placeholder(result[gap_end], placeholder) else None

                if before is not None and after is not None:
                    # Interpolate
                    before = np.array(before, dtype=float)
                    after = np.array(after, dtype=float)
                    gap_len = gap_end - gap_start
                    for j in range(gap_len):
                        t = (j + 1) / (gap_len + 1)
                        interp = before * (1 - t) + after * t
                        result[gap_start + j] = tuple(interp.astype(int))
                elif before is not None:
                    for j in range(gap_start, gap_end):
                        result[j] = before
                elif after is not None:
                    for j in range(gap_start, gap_end):
                        result[j] = after
            else:
                i += 1
        return result

    # ------------------------------------------------------------------
    # Debug visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_debug_window(frame, window_results, pos_frac, abs_frame_idx, debug_dir):
        """
        Draw all detected face tracks on a frame with scores.

        Green = highest score (speaker candidate), red = other faces.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Sort results by score descending so the best is drawn last (on top)
        sorted_results = sorted(window_results, key=lambda r: r[1])
        best_score = max(r[1] for r in window_results)

        for (t_idx, score, bbox, n_dets) in sorted_results:
            x1, y1, x2, y2 = bbox
            is_best = (score == best_score)

            # Color: green for best, red for others
            color = (0, 220, 0) if is_best else (0, 0, 220)
            thickness = 4 if is_best else 2

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Label: "Face N: +0.11 (36 dets)" or "SPEAKER: +0.11 (36 dets)"
            if is_best:
                label = f"BEST Face {t_idx}: {score:+.2f} ({n_dets} dets)"
            else:
                label = f"Face {t_idx}: {score:+.2f} ({n_dets} dets)"

            # Background rectangle for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(h, w) / 1500)
            text_thickness = max(1, int(font_scale * 2))
            (tw, th_text), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

            # Position label above the box, or below if near the top
            label_y = y1 - 10 if y1 - 10 - th_text > 0 else y2 + th_text + 10
            label_x = x1

            cv2.rectangle(vis, (label_x, label_y - th_text - 4),
                          (label_x + tw + 4, label_y + 4), color, -1)
            cv2.putText(vis, label, (label_x + 2, label_y),
                        font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Title bar at the top
        title = f"ASD Window @{pos_frac:.0%} | Frame {abs_frame_idx} | {len(window_results)} faces"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(h, w) / 1200)
        text_thickness = max(1, int(font_scale * 2))
        (tw, th_text), _ = cv2.getTextSize(title, font, font_scale, text_thickness)
        cv2.rectangle(vis, (0, 0), (tw + 16, th_text + 20), (40, 40, 40), -1)
        cv2.putText(vis, title, (8, th_text + 10),
                    font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        pct = int(pos_frac * 100)
        out_path = os.path.join(debug_dir, f"asd_window_{pct}pct_frame{abs_frame_idx:06d}.png")
        cv2.imwrite(out_path, vis)
        print(f"[ASD-DEBUG] Saved: {out_path}")

    @staticmethod
    def _save_debug_speaker(frame, winner_bbox, score, abs_frame_idx, debug_dir):
        """
        Draw the final selected speaker on a frame with a prominent green box.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = winner_bbox

        # Thick green box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Label
        label = f"SPEAKER SELECTED (score: {score:+.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, min(h, w) / 1000)
        text_thickness = max(2, int(font_scale * 2.5))
        (tw, th_text), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

        label_y = y1 - 15 if y1 - 15 - th_text > 0 else y2 + th_text + 15
        label_x = x1

        cv2.rectangle(vis, (label_x, label_y - th_text - 6),
                      (label_x + tw + 8, label_y + 6), (0, 180, 0), -1)
        cv2.putText(vis, label, (label_x + 4, label_y),
                    font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Title bar
        title = f"ASD FINAL RESULT | Frame {abs_frame_idx}"
        (tw, th_text), _ = cv2.getTextSize(title, font, font_scale, text_thickness)
        cv2.rectangle(vis, (0, 0), (tw + 16, th_text + 20), (0, 120, 0), -1)
        cv2.putText(vis, title, (8, th_text + 10),
                    font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        out_path = os.path.join(debug_dir, f"asd_speaker_selected_frame{abs_frame_idx:06d}.png")
        cv2.imwrite(out_path, vis)
        print(f"[ASD-DEBUG] Saved: {out_path}")

    def _score_track(self, track, sample_frames, window_audio, fps, model_fps):
        """Run ASD on a single face track and return average score."""
        # Build per-frame bboxes for this track
        track_bboxes = [None] * len(sample_frames)
        for (fidx, bbox, _conf) in track:
            track_bboxes[fidx] = tuple(bbox)

        track_bboxes = self._fill_track_gaps(track_bboxes)
        video_features = self._prepare_visual_features(sample_frames, track_bboxes)

        # Resample to 25fps if needed
        if abs(fps - model_fps) > 0.5:
            orig_len = video_features.shape[0]
            target_len = int(round(orig_len * model_fps / fps))
            if target_len > 0:
                indices = np.linspace(0, orig_len - 1, target_len).astype(int)
                video_features = video_features[indices]

        # Align audio/video
        duration = min(
            (window_audio.shape[0] - window_audio.shape[0] % 4) / 100,
            video_features.shape[0] / model_fps,
        )
        if duration <= 0:
            return None

        a_len = int(round(duration * 100))
        v_len = int(round(duration * model_fps))
        af = window_audio[:a_len]
        vf = video_features[:v_len]

        # Run ASD with a few windows for speed
        quick_durations = [1, 2, 3]
        all_scores = []
        for window_dur in quick_durations:
            batch_count = int(math.ceil(duration / window_dur))
            scores = []
            for i in range(batch_count):
                a_s = i * window_dur * 100
                a_e = (i + 1) * window_dur * 100
                v_s = int(i * window_dur * model_fps)
                v_e = int((i + 1) * window_dur * model_fps)

                inputA = torch.FloatTensor(af[a_s:a_e]).unsqueeze(0).to(self.device)
                inputV = torch.FloatTensor(vf[v_s:v_e]).unsqueeze(0).to(self.device)
                if inputA.shape[1] == 0 or inputV.shape[1] == 0:
                    continue

                embedA = self.asd.model.forward_audio_frontend(inputA)
                embedV = self.asd.model.forward_visual_frontend(inputV)
                out = self.asd.model.forward_audio_visual_backend(embedA, embedV)
                score = self.asd.lossAV.forward(out, labels=None)
                scores.extend(score)
            if scores:
                all_scores.append(scores)

        if not all_scores:
            return None

        min_len = min(len(s) for s in all_scores)
        avg = np.mean([s[:min_len] for s in all_scores], axis=0)
        return float(np.mean(avg))

    @staticmethod
    def _build_face_tracks(detections, iou_threshold=0.4):
        """Group per-frame face detections into tracks using IOU overlap."""
        tracks = []

        for (fidx, bbox, conf) in detections:
            best_track = None
            best_iou = iou_threshold  # minimum to match

            for track in tracks:
                last_fidx, last_bbox, _ = track[-1]
                if fidx <= last_fidx:
                    continue
                iou = _compute_iou(bbox, last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None:
                best_track.append((fidx, bbox, conf))
            else:
                tracks.append([(fidx, bbox, conf)])

        # Filter very short tracks (noise)
        min_track_len = 3
        tracks = [t for t in tracks if len(t) >= min_track_len]
        return tracks

    @staticmethod
    def _merge_similar_tracks(tracks, iou_threshold=0.5):
        """Merge tracks whose average bboxes overlap (same face, fragmented)."""
        if len(tracks) <= 1:
            return tracks

        # Compute average bbox per track
        avg_bboxes = []
        for track in tracks:
            avg = np.mean([d[1] for d in track], axis=0)
            avg_bboxes.append(avg)

        # Greedy merge
        merged = []
        used = [False] * len(tracks)
        for i in range(len(tracks)):
            if used[i]:
                continue
            group = list(tracks[i])
            used[i] = True
            for j in range(i + 1, len(tracks)):
                if used[j]:
                    continue
                if _compute_iou(avg_bboxes[i], avg_bboxes[j]) > iou_threshold:
                    group.extend(tracks[j])
                    used[j] = True
            # Sort by frame index
            group.sort(key=lambda d: d[0])
            merged.append(group)

        return merged

    @staticmethod
    def _fill_track_gaps(track_bboxes):
        """Fill None entries in a sparse track with nearest known bbox."""
        filled = list(track_bboxes)
        last = None
        for i in range(len(filled)):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last
        last = None
        for i in range(len(filled) - 1, -1, -1):
            if filled[i] is not None:
                last = filled[i]
            elif last is not None:
                filled[i] = last
        for i in range(len(filled)):
            if filled[i] is None:
                filled[i] = (0.0, 0.0, 0.0, 0.0)
        return filled

    @staticmethod
    def _pick_largest_track(tracks):
        """Fallback: return average bbox of the track with the largest faces."""
        best = None
        best_area = 0
        for track in tracks:
            areas = [(d[1][2] - d[1][0]) * (d[1][3] - d[1][1]) for d in track]
            avg_area = np.mean(areas)
            if avg_area > best_area:
                best_area = avg_area
                best = np.mean([d[1] for d in track], axis=0).astype(int)
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mfcc(audio_path: str) -> "np.ndarray | None":
        """Extract MFCC features from an audio file via FFmpeg + python_speech_features."""
        try:
            from scipy.io import wavfile

            # Convert to 16 kHz mono WAV in a temp file
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()

            cmd = (
                f'ffmpeg -y -i "{audio_path}" -ac 1 -ar 16000 '
                f'-vn -acodec pcm_s16le "{tmp_wav.name}" -loglevel panic'
            )
            subprocess.call(cmd, shell=True)

            sr, audio = wavfile.read(tmp_wav.name)
            os.unlink(tmp_wav.name)

            if len(audio) == 0:
                return None

            mfcc = python_speech_features.mfcc(
                audio, sr, numcep=13, winlen=0.025, winstep=0.010
            )
            return mfcc

        except Exception as e:
            print(f"[ASD] MFCC extraction failed: {e}")
            return None

    @staticmethod
    def _prepare_visual_features(
        frames: List[np.ndarray],
        face_bboxes: List[tuple],
    ) -> np.ndarray:
        """
        Crop faces from YOLOv8 bboxes, convert to grayscale 112x112.

        Follows LR-ASD convention: resize crop to 224x224 then center-crop to
        112x112.  Frames without a valid face get a blank (zero) crop.
        """
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        crops = []
        for frame, bbox in zip(frames, face_bboxes):
            if _is_placeholder(bbox):
                crops.append(np.zeros((112, 112), dtype=np.uint8))
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((112, 112), dtype=np.uint8))
                continue

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            # Center-crop to 112x112 (matches LR-ASD demo preprocessing)
            face = face[56:168, 56:168]
            crops.append(face)

        return np.array(crops)

    @staticmethod
    def _fill_short_gaps(mask: List[bool], max_gap: int) -> List[bool]:
        """Fill silent gaps shorter than *max_gap* frames between speaking segments."""
        result = list(mask)
        n = len(result)
        i = 0
        while i < n:
            if not result[i]:
                # Start of a silent gap
                gap_start = i
                while i < n and not result[i]:
                    i += 1
                gap_len = i - gap_start
                # If this gap is short AND bordered by speaking on both sides, fill it
                if gap_len <= max_gap and gap_start > 0 and i < n:
                    for j in range(gap_start, i):
                        result[j] = True
            else:
                i += 1
        return result

    @staticmethod
    def _map_scores_to_frames(
        asd_scores: np.ndarray,
        n_frames: int,
        fps: float,
    ) -> List[float]:
        """
        Map ASD scores (one per ASD video frame) to the original frame count.

        ASD processes at the video's native fps, so normally len(asd_scores)
        is close to n_frames.  We use nearest-neighbor interpolation when
        they differ.
        """
        n_scores = len(asd_scores)
        if n_scores == 0:
            return [0.0] * n_frames

        if n_scores >= n_frames:
            return [float(asd_scores[i]) for i in range(n_frames)]

        # Stretch scores to cover all frames via nearest-neighbor
        per_frame = []
        for i in range(n_frames):
            score_idx = min(int(i * n_scores / n_frames), n_scores - 1)
            per_frame.append(float(asd_scores[score_idx]))
        return per_frame
