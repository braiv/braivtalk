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
        speaking_mask = [s > threshold for s in per_frame_scores]

        speaking_count = sum(speaking_mask)
        silent_count = n_frames - speaking_count
        print("=" * 60)
        print(f"[ASD] RESULTS: {n_frames} total frames")
        print(f"[ASD]   Speaking:  {speaking_count} frames "
              f"({100 * speaking_count / n_frames:.1f}%)")
        print(f"[ASD]   Silent:    {silent_count} frames "
              f"({100 * silent_count / n_frames:.1f}%)")
        print(f"[ASD]   Threshold: {threshold}")

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
            if bbox == coord_placeholder or bbox is None:
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
