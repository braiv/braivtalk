"""
Ditto TalkingHead ONNX inference pipeline.

Adapted from antgroup/ditto-talkinghead reference implementation.
All inference is pure ONNX + NumPy (no PyTorch/TensorRT required at runtime).

Pipeline stages:
  1. Avatar registration: face detect → crop → appearance + motion extraction
  2. Audio features: HuBERT (16 kHz waveform → [T, 1024])
  3. Conditioning: audio features + emotion + eye + shape code
  4. Audio → motion: LMDM diffusion → driven motion per frame
  5. Per-frame render: motion stitch → warp → decode → putback
"""

import copy
import math
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from tqdm import tqdm


HUGGINGFACE_BASE = 'https://huggingface.co/digital-avatar/ditto-talkinghead/resolve/main/ditto_onnx'

REQUIRED_ONNX_MODELS = [
    'insightface_det',
    'landmark106',
    'landmark203',
    'appearance_extractor',
    'motion_extractor',
    'hubert',
    'lmdm_v0.4_hubert',
    'stitch_network',
    'warp_network_ori',
    'decoder',
]

ONNX_PROVIDERS = None


def _get_providers():
    global ONNX_PROVIDERS
    if ONNX_PROVIDERS is not None:
        return ONNX_PROVIDERS
    try:
        import onnxruntime
        available = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            ONNX_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            ONNX_PROVIDERS = ['CPUExecutionProvider']
    except Exception:
        ONNX_PROVIDERS = ['CPUExecutionProvider']
    return ONNX_PROVIDERS


def _create_session(model_path: str):
    import onnxruntime
    return onnxruntime.InferenceSession(model_path, providers=_get_providers())


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_download_lock = None


def _get_download_lock():
    global _download_lock
    if _download_lock is None:
        import threading
        _download_lock = threading.Lock()
    return _download_lock


def download_models(model_dir: str, progress: bool = True) -> bool:
    lock = _get_download_lock()
    with lock:
        os.makedirs(model_dir, exist_ok=True)
        all_ok = True
        for name in REQUIRED_ONNX_MODELS:
            local_path = os.path.join(model_dir, f'{name}.onnx')
            if os.path.isfile(local_path):
                continue
            url = f'{HUGGINGFACE_BASE}/{name}.onnx'
            print(f'[ditto] Downloading {name}.onnx …')
            try:
                if progress:
                    _download_with_progress(url, local_path)
                else:
                    urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                print(f'[ditto] Failed to download {name}.onnx: {e}')
                for stale in [local_path, local_path + '.part']:
                    try:
                        if os.path.isfile(stale):
                            os.remove(stale)
                    except OSError:
                        pass
                all_ok = False
        return all_ok


def _download_with_progress(url: str, dest: str):
    import time
    tmp = dest + '.part'
    # Clean up stale .part file from a previous interrupted download
    if os.path.isfile(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    req = urllib.request.Request(url, headers={'User-Agent': 'braivtalk/1.0'})
    resp = urllib.request.urlopen(req)
    total = int(resp.headers.get('Content-Length', 0))
    block = 1 << 20  # 1 MiB
    with open(tmp, 'wb') as f:
        bar = tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest))
        while True:
            chunk = resp.read(block)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()
    # Windows: brief sleep to let antivirus / OS release the file handle
    time.sleep(0.2)
    for attempt in range(5):
        try:
            os.replace(tmp, dest)
            return
        except OSError:
            time.sleep(1.0 * (attempt + 1))
    raise OSError(f'Could not rename {tmp} → {dest} after 5 attempts')


def check_models(model_dir: str) -> bool:
    return all(
        os.path.isfile(os.path.join(model_dir, f'{name}.onnx'))
        for name in REQUIRED_ONNX_MODELS
    )


# ---------------------------------------------------------------------------
# Geometry helpers (replaces skimage dependency)
# ---------------------------------------------------------------------------

def _build_similarity_transform(center, output_size, scale, rotation_deg):
    """Build a 2×3 affine matrix equivalent to skimage's SimilarityTransform chain."""
    rot = float(rotation_deg) * np.pi / 180.0
    s = float(scale)
    cx, cy = float(center[0]), float(center[1])
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    M = np.array([
        [s * cos_r, s * sin_r, output_size / 2 - s * (cos_r * cx + sin_r * cy)],
        [-s * sin_r, s * cos_r, output_size / 2 - s * (-sin_r * cx + cos_r * cy)],
    ], dtype=np.float32)
    return M


def _transform_img(img, M, dsize, flags=cv2.INTER_LINEAR, border_mode=None):
    if isinstance(dsize, (list, tuple)):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)
    if border_mode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags,
                              borderMode=border_mode, borderValue=(0, 0, 0))
    return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    return pts @ M[:2, :2].T + M[:2, 2]


# ---------------------------------------------------------------------------
# Landmark parsing (extract 2 canonical points from various formats)
# ---------------------------------------------------------------------------

def _parse_pt2_from_pt106(pt106, use_lip=True):
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def _parse_pt2_from_pt203(pt203, use_lip=True):
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def _parse_pt2(pts, use_lip=True):
    n = pts.shape[0]
    if n == 106:
        return _parse_pt2_from_pt106(pts, use_lip)
    if n == 203:
        return _parse_pt2_from_pt203(pts, use_lip)
    if n > 101:
        return _parse_pt2_from_pt106(pts[:106], use_lip)
    raise ValueError(f'Unsupported landmark count: {n}')


def _parse_rect_from_landmark(pts, scale=1.5, vx_ratio=0, vy_ratio=0, **kw):
    pt2 = _parse_pt2(pts, use_lip=kw.get('use_lip', True))
    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)
    angle = np.arccos(np.clip(ux[0], -1, 1))
    if ux[1] < 0:
        angle = -angle
    M = np.array([ux, uy])
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2
    size = rb_pt - lt_pt
    m = max(size[0], size[1])
    size[0] = size[1] = m
    size *= scale
    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    return center, size, angle


def crop_image(img, pts, **kw):
    dsize = kw.get('dsize', 224)
    scale = kw.get('scale', 1.5)
    vy_ratio = kw.get('vy_ratio', -0.1)
    flag_do_rot = kw.get('flag_do_rot', True)
    center, size, angle = _parse_rect_from_landmark(pts, scale=scale, vy_ratio=vy_ratio,
                                                     use_lip=kw.get('use_lip', True))
    s = dsize / size[0]
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)
    if flag_do_rot:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        cx, cy = center
        tcx, tcy = tgt_center
        M_INV = np.array([
            [s * cos_a, s * sin_a, tcx - s * (cos_a * cx + sin_a * cy)],
            [-s * sin_a, s * cos_a, tcy - s * (-sin_a * cx + cos_a * cy)],
        ], dtype=np.float32)
    else:
        M_INV = np.array([
            [s, 0, tgt_center[0] - s * center[0]],
            [0, s, tgt_center[1] - s * center[1]],
        ], dtype=np.float32)
    img_crop = _transform_img(img, M_INV, dsize)
    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=np.float32)])
    M_c2o = np.linalg.inv(M_o2c)
    return {'img_crop': img_crop, 'M_o2c': M_o2c, 'M_c2o': M_c2o}


# ---------------------------------------------------------------------------
# Face detection – InsightFace RetinaFace (ONNX)
# ---------------------------------------------------------------------------

def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets, thresh=0.4):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class InsightFaceDetONNX:
    INPUT_SIZE = (512, 512)
    INPUT_MEAN = 127.5
    INPUT_STD = 128.0
    FMC = 3
    FEAT_STRIDE = [8, 16, 32]
    NUM_ANCHORS = 2
    DET_THRESH = 0.5
    NMS_THRESH = 0.4

    def __init__(self, model_path: str):
        self.session = _create_session(model_path)
        self._center_cache: Dict = {}

    def detect(self, img_rgb: NDArray) -> Tuple[NDArray, Optional[NDArray]]:
        input_size = self.INPUT_SIZE
        im_ratio = img_rgb.shape[0] / img_rgb.shape[1]
        model_ratio = input_size[1] / input_size[0]
        if im_ratio > model_ratio:
            new_h = input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = new_h / img_rgb.shape[0]
        resized = cv2.resize(img_rgb, (new_w, new_h))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / self.INPUT_STD, input_size,
                                     (self.INPUT_MEAN,) * 3, swapRB=True)
        net_outs = self.session.run(None, {'image': blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self.FEAT_STRIDE):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.FMC] * stride
            kps_preds = net_outs[idx + self.FMC * 2] * stride
            h_f = blob.shape[2] // stride
            w_f = blob.shape[3] // stride
            key = (h_f, w_f, stride)
            if key in self._center_cache:
                ac = self._center_cache[key]
            else:
                ac = np.stack(np.mgrid[:h_f, :w_f][::-1], axis=-1).astype(np.float32)
                ac = (ac * stride).reshape(-1, 2)
                if self.NUM_ANCHORS > 1:
                    ac = np.stack([ac] * self.NUM_ANCHORS, axis=1).reshape(-1, 2)
                self._center_cache[key] = ac
            pos_inds = np.where(scores >= self.DET_THRESH)[0]
            bboxes = _distance2bbox(ac, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss = _distance2kps(ac, kps_preds).reshape(kps_preds.shape[0], -1, 2)
            kpss_list.append(kpss[pos_inds])

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        pre_det = pre_det[order]
        keep = _nms(pre_det, self.NMS_THRESH)
        det = pre_det[keep]
        kpss = kpss[order][keep]
        return det, kpss


# ---------------------------------------------------------------------------
# Landmark106 (ONNX)
# ---------------------------------------------------------------------------

class Landmark106ONNX:
    INPUT_SIZE = (192, 192)
    LMK_NUM = 106

    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def get(self, img_rgb: NDArray, bbox: NDArray) -> NDArray:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
        s = self.INPUT_SIZE[0] / (max(w, h) * 1.5)
        M = _build_similarity_transform(center, self.INPUT_SIZE[0], s, 0)
        aimg = cv2.warpAffine(img_rgb, M, self.INPUT_SIZE, flags=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(aimg, 1.0, self.INPUT_SIZE, (0, 0, 0), swapRB=True)
        pred = self.session.run(None, {'data': blob})[0]
        pred = pred.reshape(-1, 2)
        if self.LMK_NUM < pred.shape[0]:
            pred = pred[-self.LMK_NUM:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.INPUT_SIZE[0] // 2)
        IM = cv2.invertAffineTransform(M)
        pred = _transform_pts(pred, IM)
        return pred


# ---------------------------------------------------------------------------
# Landmark203 (ONNX)
# ---------------------------------------------------------------------------

class Landmark203ONNX:
    DSIZE = 224

    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def run(self, img_crop_rgb: NDArray, M_c2o: Optional[NDArray] = None) -> NDArray:
        inp = (img_crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out_pts = self.session.run(None, {'input': inp})[0]
        lmk = out_pts[0].reshape(-1, 2) * self.DSIZE
        if M_c2o is not None:
            lmk = _transform_pts(lmk, M_c2o)
        return lmk


# ---------------------------------------------------------------------------
# Appearance extractor (ONNX)
# ---------------------------------------------------------------------------

class AppearanceExtractorONNX:
    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def __call__(self, image: NDArray) -> NDArray:
        """image: (1, 3, 256, 256) float32 [0,1]"""
        return self.session.run(None, {'image': image})[0]


# ---------------------------------------------------------------------------
# Motion extractor (ONNX)
# ---------------------------------------------------------------------------

class MotionExtractorONNX:
    OUTPUT_NAMES = ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']

    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def __call__(self, image: NDArray) -> Dict[str, NDArray]:
        """image: (1, 3, 256, 256) float32 [0,1]"""
        out_list = self.session.run(None, {'image': image})
        outputs = {name: out_list[i] for i, name in enumerate(self.OUTPUT_NAMES)}
        outputs['exp'] = outputs['exp'].reshape(1, -1)
        outputs['kp'] = outputs['kp'].reshape(1, -1)
        return outputs


# ---------------------------------------------------------------------------
# HuBERT (ONNX)
# ---------------------------------------------------------------------------

class HubertONNX:
    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def forward_chunk(self, audio_chunk: NDArray) -> NDArray:
        return self.session.run(None, {'input_values': audio_chunk.reshape(1, -1)})[0]


# ---------------------------------------------------------------------------
# Wav2Feat – HuBERT feature extraction from waveform
# ---------------------------------------------------------------------------

class Wav2Feat:
    FEAT_DIM = 1024

    def __init__(self, hubert: HubertONNX):
        self.hubert = hubert

    def wav2feat(self, audio_16k: NDArray, chunksize: Tuple[int, int, int] = (3, 5, 2)) -> NDArray:
        """Offline full-utterance HuBERT feature extraction."""
        num_f = math.ceil(len(audio_16k) / 16000 * 25)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80

        speech_pad = np.concatenate([
            np.zeros((split_len - int(sum(chunksize[1:]) * 0.04 * 16000),), dtype=audio_16k.dtype),
            audio_16k,
            np.zeros((split_len,), dtype=audio_16k.dtype),
        ], 0)

        valid_feat_s = -sum(chunksize[1:]) * 2
        valid_feat_e = -chunksize[2] * 2

        i = 0
        res_lst = []
        while i < num_f:
            sss = int(i * 0.04 * 16000)
            eee = sss + split_len
            chunk = speech_pad[sss:eee]
            encoding = self.hubert.forward_chunk(chunk)
            valid_encoding = encoding[valid_feat_s:valid_feat_e]
            valid_feat = valid_encoding.reshape(chunksize[1], 2, 1024).mean(1)
            res_lst.append(valid_feat)
            i += chunksize[1]

        ret = np.concatenate(res_lst, 0)
        return ret[:num_f]


# ---------------------------------------------------------------------------
# Condition handler – merge audio features with emotion/eye/shape conditions
# ---------------------------------------------------------------------------

def _get_emo_avg(idx=6):
    emo = np.zeros(8, dtype=np.float32)
    if isinstance(idx, (list, tuple)):
        for i in idx:
            emo[i] = 8
    else:
        emo[idx] = 8
    return softmax(emo)


class ConditionHandler:
    def __init__(self, use_emo=True, use_sc=True, use_eye_open=True, use_eye_ball=True, seq_frames=80):
        self.use_emo = use_emo
        self.use_sc = use_sc
        self.use_eye_open = use_eye_open
        self.use_eye_ball = use_eye_ball
        self.seq_frames = seq_frames

    def setup(self, source_info: Dict, emo=4, eye_f0_mode=False):
        self.eye_f0_mode = eye_f0_mode
        self.x_s_info_0 = source_info['x_s_info_lst'][0]

        if self.use_sc:
            self.sc = source_info['sc']
            self.sc_seq = np.stack([self.sc] * self.seq_frames, 0)

        if self.use_eye_open:
            self.eye_open_lst = np.concatenate(source_info['eye_open_lst'], 0)
            self.num_eye_open = len(self.eye_open_lst)
            if self.num_eye_open == 1 or self.eye_f0_mode:
                self.eye_open_seq = np.stack([self.eye_open_lst[0]] * self.seq_frames, 0)
            else:
                self.eye_open_seq = None

        if self.use_eye_ball:
            self.eye_ball_lst = np.concatenate(source_info['eye_ball_lst'], 0)
            self.num_eye_ball = len(self.eye_ball_lst)
            if self.num_eye_ball == 1 or self.eye_f0_mode:
                self.eye_ball_seq = np.stack([self.eye_ball_lst[0]] * self.seq_frames, 0)
            else:
                self.eye_ball_seq = None

        if self.use_emo:
            emo_avg = _get_emo_avg(emo)
            self.emo_seq = np.stack([emo_avg] * self.seq_frames, 0)

    def __call__(self, aud_feat: NDArray, idx: int = 0) -> NDArray:
        frame_num = len(aud_feat)
        parts = [aud_feat]
        if self.use_emo:
            if len(self.emo_seq) == frame_num:
                parts.append(self.emo_seq)
            else:
                parts.append(np.stack([self.emo_seq[0]] * frame_num, 0))
        if self.use_eye_open:
            if self.eye_open_seq is not None and len(self.eye_open_seq) == frame_num:
                parts.append(self.eye_open_seq)
            else:
                idxs = [_mirror_index(max(i, 0), self.num_eye_open) for i in range(idx, idx + frame_num)]
                parts.append(self.eye_open_lst[idxs])
        if self.use_eye_ball:
            if self.eye_ball_seq is not None and len(self.eye_ball_seq) == frame_num:
                parts.append(self.eye_ball_seq)
            else:
                idxs = [_mirror_index(max(i, 0), self.num_eye_ball) for i in range(idx, idx + frame_num)]
                parts.append(self.eye_ball_lst[idxs])
        if self.use_sc:
            if len(self.sc_seq) == frame_num:
                parts.append(self.sc_seq)
            else:
                parts.append(np.stack([self.sc] * frame_num, 0))
        return np.concatenate(parts, -1).astype(np.float32)


def _mirror_index(index, size):
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1


# ---------------------------------------------------------------------------
# Motion format conversion
# ---------------------------------------------------------------------------

_KS_SHAPE_MAP = [
    ('scale', (1, 1), 1),
    ('pitch', (1, 66), 66),
    ('yaw',   (1, 66), 66),
    ('roll',  (1, 66), 66),
    ('t',     (1, 3), 3),
    ('exp',   (1, 63), 63),
    ('kp',    (1, 63), 63),
]


def _motion_dic2arr(dic, ignore_keys=()):
    arr = []
    for k, _, ds in _KS_SHAPE_MAP:
        if k not in dic or k in ignore_keys:
            continue
        v = dic[k].reshape(ds)
        if k == 'scale':
            v = v - 1
        arr.append(v)
    return np.concatenate(arr, -1)


def _motion_arr2dic(arr):
    dic = {}
    s = 0
    for k, ds, ss in _KS_SHAPE_MAP:
        v = arr[s:s + ss].reshape(ds)
        if k == 'scale':
            v = v + 1
        dic[k] = v
        s += ss
        if s >= len(arr):
            break
    return dic


# ---------------------------------------------------------------------------
# LMDM – Landmark Diffusion Model (ONNX)
# ---------------------------------------------------------------------------

def _make_beta(n_timestep, cosine_s=8e-3):
    import torch as _torch
    timesteps = (_torch.arange(n_timestep + 1, dtype=_torch.float64) / n_timestep + cosine_s)
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = _torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    return np.clip(betas.numpy(), 0, 0.999)


def _make_beta_np(n_timestep=1000, cosine_s=8e-3):
    """Pure numpy version of the cosine beta schedule."""
    steps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
    alphas_bar = np.cos(steps / (1 + cosine_s) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return np.clip(betas, 0, 0.999)


class LMDM_ONNX:
    def __init__(self, model_path: str, motion_feat_dim: int = 265,
                 audio_feat_dim: int = 1059, seq_frames: int = 80):
        self.session = _create_session(model_path)
        self.motion_feat_dim = motion_feat_dim
        self.seq_frames = seq_frames
        self.n_timestep = 1000

        betas = _make_beta_np(self.n_timestep).astype(np.float64)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self._sampling_timesteps = None
        self._time_pairs = None
        self._schedules: Dict[str, List] = {}

    def setup(self, sampling_timesteps: int = 50):
        if self._sampling_timesteps == sampling_timesteps:
            return
        self._sampling_timesteps = sampling_timesteps
        total = self.n_timestep
        times_f = np.linspace(-1, total - 1, sampling_timesteps + 1)
        times = list(reversed(times_f.astype(int).tolist()))
        self._time_pairs = list(zip(times[:-1], times[1:]))

        shape = (1, self.seq_frames, self.motion_feat_dim)
        tc_list, asq_list, sig_list, c_list, noise_list = [], [], [], [], []
        for t, t_next in self._time_pairs:
            tc_list.append(np.full((1,), t, dtype=np.int64))
            if t_next < 0:
                continue
            a = float(self.alphas_cumprod[t])
            a_next = float(self.alphas_cumprod[t_next])
            sigma = math.sqrt((1 - a / a_next) * (1 - a_next) / (1 - a))
            c = math.sqrt(1 - a_next - sigma ** 2)
            asq_list.append(np.float32(math.sqrt(a_next)))
            sig_list.append(np.float32(sigma))
            c_list.append(np.float32(c))
            noise_list.append(np.random.randn(*shape).astype(np.float32))
        self._schedules = {
            'time_cond': tc_list, 'alpha_next_sqrt': asq_list,
            'sigma': sig_list, 'c': c_list, 'noise': noise_list,
        }

    def _one_step(self, x, cond_frame, cond, time_cond):
        pred = self.session.run(None, {
            'x': x, 'cond_frame': cond_frame, 'cond': cond, 'time_cond': time_cond,
        })
        return pred[0], pred[1]  # pred_noise, x_start

    def sample(self, kp_cond: NDArray, aud_cond: NDArray) -> NDArray:
        """Run DDIM sampling. kp_cond: (1, D), aud_cond: (1, T, D_cond)"""
        kp_cond = np.asarray(kp_cond, dtype=np.float32)
        aud_cond = np.asarray(aud_cond, dtype=np.float32)
        x = np.random.randn(1, self.seq_frames, self.motion_feat_dim).astype(np.float32)
        i = 0
        for _, t_next in self._time_pairs:
            tc = self._schedules['time_cond'][i]
            pred_noise, x_start = self._one_step(x, kp_cond, aud_cond, tc)
            if t_next < 0:
                x = np.asarray(x_start, dtype=np.float32)
                break
            asq = self._schedules['alpha_next_sqrt'][i]
            c = self._schedules['c'][i]
            sig = self._schedules['sigma'][i]
            noise = self._schedules['noise'][i]
            x = (x_start * asq + c * pred_noise + sig * noise).astype(np.float32)
            i += 1
        return x


# ---------------------------------------------------------------------------
# Audio2Motion – runs LMDM over sliding windows to produce per-frame motion
# ---------------------------------------------------------------------------

class Audio2Motion:
    def __init__(self, lmdm: LMDM_ONNX):
        self.lmdm = lmdm
        self.seq_frames = lmdm.seq_frames

    def setup(self, x_s_info: Dict, overlap: int = 10, sampling_timesteps: int = 50,
              smo_k: int = 3):
        self.overlap = overlap
        self.valid_clip_len = self.seq_frames - overlap
        self.smo_k = smo_k
        self.fuse_length = overlap
        self.fuse_alpha = (np.arange(self.fuse_length, dtype=np.float32)
                           .reshape(1, -1, 1) / self.fuse_length)
        kp_source = _motion_dic2arr(x_s_info, ignore_keys={'kp'})[None].astype(np.float32)
        self.s_kp_cond = kp_source.copy().reshape(1, -1)
        self.kp_cond = self.s_kp_cond.copy()
        self.lmdm.setup(sampling_timesteps)

    def run_offline(self, aud_cond_all: NDArray) -> NDArray:
        """aud_cond_all: (N_frames, cond_dim). Returns (1, N_frames, motion_dim)."""
        num_frames = len(aud_cond_all)
        idx = 0
        res_kp_seq = None
        bar = tqdm(desc='[ditto] LMDM diffusion', total=math.ceil(num_frames / self.valid_clip_len))
        while idx < num_frames:
            bar.update()
            aud_cond = aud_cond_all[idx:idx + self.seq_frames][None]
            if aud_cond.shape[1] < self.seq_frames:
                pad = np.stack([aud_cond[:, -1]] * (self.seq_frames - aud_cond.shape[1]), 1)
                aud_cond = np.concatenate([aud_cond, pad], 1)
            pred_kp_seq = self.lmdm.sample(self.kp_cond, aud_cond)
            if res_kp_seq is None:
                res_kp_seq = pred_kp_seq
                res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])
            else:
                res_kp_seq = self._fuse(res_kp_seq, pred_kp_seq)
                s = res_kp_seq.shape[1] - self.valid_clip_len - self.fuse_length
                e = res_kp_seq.shape[1] - self.valid_clip_len + 1
                res_kp_seq = self._smo(res_kp_seq, s, e)
            new_idx = res_kp_seq.shape[1] - self.overlap
            self.kp_cond = res_kp_seq[:, new_idx - 1].astype(np.float32)
            idx += self.valid_clip_len
        bar.close()
        res_kp_seq = res_kp_seq[:, :num_frames]
        res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])
        return res_kp_seq

    def cvt_fmt(self, res_kp_seq: NDArray) -> List[Dict]:
        """Convert (1, N, dim) motion tensor to list of dicts."""
        seq = res_kp_seq[0]
        return [_motion_arr2dic(seq[i]) for i in range(seq.shape[0])]

    def _fuse(self, res, pred):
        fl = self.fuse_length
        r1_s = res.shape[1] - fl
        r2_s = self.seq_frames - self.valid_clip_len - fl
        r2_e = self.seq_frames - self.valid_clip_len
        r1 = res[:, r1_s:r1_s + fl]
        r2 = pred[:, r2_s:r2_e]
        r_fuse = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha
        res[:, r1_s:r1_s + fl] = r_fuse
        return np.concatenate([res, pred[:, r2_e:]], 1)

    def _smo(self, seq, s, e):
        if self.smo_k <= 1:
            return seq
        new = seq.copy()
        n = seq.shape[1]
        hk = self.smo_k // 2
        for i in range(s, min(e, n)):
            ss = max(0, i - hk)
            ee = min(n, i + hk + 1)
            seq[:, i, :202] = np.mean(new[:, ss:ee, :202], axis=1)
        return seq


# ---------------------------------------------------------------------------
# Stitch network (ONNX)
# ---------------------------------------------------------------------------

class StitchNetworkONNX:
    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def __call__(self, kp_source: NDArray, kp_driving: NDArray) -> NDArray:
        return self.session.run(None, {
            'kp_source': kp_source, 'kp_driving': kp_driving,
        })[0]


# ---------------------------------------------------------------------------
# Warp network (ONNX)
# ---------------------------------------------------------------------------

class WarpNetworkONNX:
    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def __call__(self, feature_3d: NDArray, kp_source: NDArray, kp_driving: NDArray) -> NDArray:
        """feature_3d: (1,32,16,64,64), kp_*: (1,21,3)"""
        return self.session.run(None, {
            'feature_3d': feature_3d, 'kp_source': kp_source, 'kp_driving': kp_driving,
        })[0]


# ---------------------------------------------------------------------------
# Decoder (ONNX)
# ---------------------------------------------------------------------------

class DecoderONNX:
    def __init__(self, model_path: str):
        self.session = _create_session(model_path)

    def __call__(self, feature: NDArray) -> NDArray:
        pred = self.session.run(None, {'feature': feature})[0]
        return np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255


# ---------------------------------------------------------------------------
# Motion stitching helpers (pure numpy)
# ---------------------------------------------------------------------------

def _bin66_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = np.arange(66, dtype=np.float32)
        pred_soft = softmax(pred, axis=1)
        return np.sum(pred_soft * idx, axis=1) * 3 - 97.5
    return pred


def _get_rotation_matrix(pitch_, yaw_, roll_):
    x = np.atleast_1d(pitch_ / 180 * np.pi).astype(np.float32).reshape(-1, 1)
    y = np.atleast_1d(yaw_ / 180 * np.pi).astype(np.float32).reshape(-1, 1)
    z = np.atleast_1d(roll_ / 180 * np.pi).astype(np.float32).reshape(-1, 1)
    bs = x.shape[0]
    ones = np.ones((bs, 1), dtype=np.float32)
    zeros = np.zeros((bs, 1), dtype=np.float32)
    rot_x = np.concatenate([ones, zeros, zeros, zeros, np.cos(x), -np.sin(x),
                            zeros, np.sin(x), np.cos(x)], 1).reshape(bs, 3, 3)
    rot_y = np.concatenate([np.cos(y), zeros, np.sin(y), zeros, ones, zeros,
                            -np.sin(y), zeros, np.cos(y)], 1).reshape(bs, 3, 3)
    rot_z = np.concatenate([np.cos(z), -np.sin(z), zeros, np.sin(z), np.cos(z),
                            zeros, zeros, zeros, ones], 1).reshape(bs, 3, 3)
    rot = rot_z @ rot_y @ rot_x
    return rot.transpose(0, 2, 1)


def _transform_keypoint(kp_info: Dict) -> NDArray:
    kp = kp_info['kp']
    pitch = _bin66_to_degree(kp_info['pitch'])
    yaw = _bin66_to_degree(kp_info['yaw'])
    roll = _bin66_to_degree(kp_info['roll'])
    t, exp, scale = kp_info['t'], kp_info['exp'], kp_info['scale']
    bs = kp.shape[0]
    num_kp = kp.shape[1] // 3 if kp.ndim == 2 else kp.shape[1]
    rot = _get_rotation_matrix(pitch, yaw, roll)
    kp_t = np.matmul(kp.reshape(bs, num_kp, 3), rot) + exp.reshape(bs, num_kp, 3)
    kp_t *= scale[..., None]
    kp_t[:, :, 0:2] += t[:, None, 0:2]
    return kp_t


def _mix_s_d_info(x_s_info, x_d_info, use_d_keys, d0=None):
    """Merge source and driven motion, keeping source for keys not in use_d_keys."""
    if d0 is not None:
        if isinstance(use_d_keys, dict):
            x_d_info = {
                k: x_s_info[k] + (v - d0[k]) * use_d_keys.get(k, 1)
                for k, v in x_d_info.items()
            }
        else:
            x_d_info = {k: x_s_info[k] + (v - d0[k]) for k, v in x_d_info.items()}
    for k, v in x_s_info.items():
        if k not in x_d_info or k not in use_d_keys:
            x_d_info[k] = v
    return x_d_info


def _fix_exp_for_video(x_d_info, x_s_info):
    """For video source: only use driven lips, keep source eyes and other expression."""
    _lip = [6, 12, 14, 17, 19, 20]
    a1 = np.zeros((21, 3), dtype=np.float32)
    a1[_lip] = 1
    a1 = a1.reshape(1, -1)
    a2 = 1 - a1
    x_d_info['exp'] = x_d_info['exp'] * a1 + x_s_info['exp'] * a2
    return x_d_info


# ---------------------------------------------------------------------------
# PutBack – composite rendered face onto original frame
# ---------------------------------------------------------------------------

def _get_mask(W, H, ratio_w=0.9, ratio_h=0.9):
    w, h = int(W * ratio_w), int(H * ratio_h)
    x1, x2 = (W - w) // 2, (W - w) // 2 + w
    y1, y2 = (H - h) // 2, (H - h) // 2 + h
    mask = np.ones((H, W), dtype=np.float32)

    def _grad(rows, cols):
        r = np.linspace(0, 0, cols)[None, :]
        c = np.linspace(0, 1, rows)[:, None]
        return np.sqrt(r ** 2 + c ** 2).astype(np.float32)

    mask[0:y1, x1:x2] = np.sqrt(np.linspace(0, 0, w)[None, :] ** 2 +
                                  np.linspace(0, 1, y1)[:, None] ** 2).clip(0, 1)
    mask[y2:H, x1:x2] = np.sqrt(np.linspace(0, 0, w)[None, :] ** 2 +
                                  np.linspace(1, 0, H - y2)[:, None] ** 2).clip(0, 1)
    mask[y1:y2, 0:x1] = np.sqrt(np.linspace(0, 1, x1)[None, :] ** 2 +
                                  np.linspace(0, 0, h)[:, None] ** 2).clip(0, 1)
    mask[y1:y2, x2:W] = np.sqrt(np.linspace(1, 0, W - x2)[None, :] ** 2 +
                                  np.linspace(0, 0, h)[:, None] ** 2).clip(0, 1)
    # corners
    for ys, ye, xs, xe, rx, ry in [
        (0, y1, 0, x1, (1, 0), (1, 0)),
        (0, y1, x2, W, (0, 1), (1, 0)),
        (y2, H, 0, x1, (1, 0), (0, 1)),
        (y2, H, x2, W, (0, 1), (0, 1)),
    ]:
        rh, rw = ye - ys, xe - xs
        if rh > 0 and rw > 0:
            row = np.linspace(*rx, rw)[None, :]
            col = np.linspace(*ry, rh)[:, None]
            mask[ys:ye, xs:xe] = 1 - np.clip(np.sqrt(row ** 2 + col ** 2), 0, 1)
    return mask


class PutBack:
    def __init__(self):
        self.mask_float = _get_mask(512, 512, 0.9, 0.9)

    def __call__(self, frame_rgb: NDArray, render_img: NDArray, M_c2o: NDArray) -> NDArray:
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(self.mask_float, M_c2o[:2, :], (w, h),
                                     flags=cv2.INTER_LINEAR).clip(0, 1)
        frame_warped = cv2.warpAffine(render_img.astype(np.float32), M_c2o[:2, :],
                                      (w, h), flags=cv2.INTER_LINEAR)
        result = (mask_warped[..., None] * frame_warped +
                  (1 - mask_warped[..., None]) * frame_rgb.astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Source2Info – face detection → crop → feature extraction for one frame
# ---------------------------------------------------------------------------

class Source2Info:
    def __init__(self, face_det: InsightFaceDetONNX, lmk106: Landmark106ONNX,
                 lmk203: Landmark203ONNX, app_ext: AppearanceExtractorONNX,
                 mot_ext: MotionExtractorONNX):
        self.face_det = face_det
        self.lmk106 = lmk106
        self.lmk203 = lmk203
        self.app_ext = app_ext
        self.mot_ext = mot_ext

    @staticmethod
    def _img_to_bchw256(img_crop_rgb):
        rgb_256 = cv2.resize(img_crop_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        return (rgb_256.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)

    def _crop(self, img_rgb, last_lmk=None, **kw):
        if last_lmk is None:
            det, _ = self.face_det.detect(img_rgb)
            if len(det) == 0:
                return None
            boxes = det[np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))]
            lmk_for_track = self.lmk106.get(img_rgb, boxes[0])
        else:
            lmk_for_track = last_lmk

        crop_dct = crop_image(img_rgb, lmk_for_track, dsize=self.lmk203.DSIZE,
                              scale=1.5, vy_ratio=-0.1, flag_do_rot=False)
        lmk203 = self.lmk203.run(crop_dct['img_crop'], crop_dct['M_c2o'])

        ret = crop_image(img_rgb, lmk203, dsize=512,
                         scale=kw.get('crop_scale', 2.3),
                         vx_ratio=kw.get('crop_vx_ratio', 0),
                         vy_ratio=kw.get('crop_vy_ratio', -0.125),
                         flag_do_rot=kw.get('crop_flag_do_rot', True))
        return ret['img_crop'], ret['M_c2o'], lmk203

    def __call__(self, img_rgb, last_lmk=None, **kw):
        result = self._crop(img_rgb, last_lmk, **kw)
        if result is None:
            return None
        img_crop, M_c2o, lmk203 = result
        rgb_256 = self._img_to_bchw256(img_crop)
        kp_info = self.mot_ext(rgb_256)
        f_s = self.app_ext(rgb_256)

        # Default eye values (skipping blaze_face + face_mesh for simplicity)
        eye_open = np.array([[0.8, 0.8]], dtype=np.float32)
        eye_ball = np.zeros((1, 6), dtype=np.float32)

        return {
            'x_s_info': kp_info, 'f_s': f_s, 'M_c2o': M_c2o,
            'eye_open': eye_open, 'eye_ball': eye_ball, 'lmk203': lmk203,
        }


# ---------------------------------------------------------------------------
# Avatar registrar – process all source frames
# ---------------------------------------------------------------------------

def _smooth_x_s_info_lst(info_list, smo_k=13):
    if len(info_list) <= 1:
        return info_list
    keys = info_list[0].keys()
    N = len(info_list)
    smo = {}
    for k in keys:
        arr = np.stack([info_list[i][k] for i in range(N)], 0)
        half = smo_k // 2
        smoothed = arr.copy()
        for i in range(N):
            s, e = max(0, i - half), min(N, i + half + 1)
            smoothed[i] = arr[s:e].mean(0)
        smo[k] = smoothed
    return [{k: smo[k][i] for k in keys} for i in range(N)]


def register_avatar(source2info: Source2Info, img_rgb_list: List[NDArray],
                    smo_k: int = 13, **crop_kw) -> Dict:
    """Register source avatar from a list of RGB frames."""
    source_info = {
        'x_s_info_lst': [], 'f_s_lst': [], 'M_c2o_lst': [],
        'eye_open_lst': [], 'eye_ball_lst': [],
    }
    last_lmk = None
    for rgb in tqdm(img_rgb_list, desc='[ditto] Registering avatar'):
        info = source2info(rgb, last_lmk, **crop_kw)
        if info is None:
            continue
        for k in ['x_s_info', 'f_s', 'M_c2o', 'eye_open', 'eye_ball']:
            source_info[f'{k}_lst'].append(info[k])
        last_lmk = info['lmk203']

    if not source_info['x_s_info_lst']:
        raise RuntimeError('No face detected in any source frame')

    if len(source_info['x_s_info_lst']) > 1 and smo_k > 1:
        source_info['x_s_info_lst'] = _smooth_x_s_info_lst(source_info['x_s_info_lst'], smo_k)

    source_info['sc'] = source_info['x_s_info_lst'][0]['kp'].flatten()
    source_info['is_image_flag'] = len(img_rgb_list) == 1
    source_info['img_rgb_lst'] = img_rgb_list
    return source_info


# ===========================================================================
# DittoPipeline – top-level orchestrator
# ===========================================================================

class DittoPipeline:
    """Full Ditto ONNX inference pipeline for audio-driven talking head."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._models_loaded = False
        self.source_info: Optional[Dict] = None
        self.x_d_info_list: Optional[List[Dict]] = None
        self.putback = PutBack()
        self.num_source_frames = 0

    def load_models(self):
        """Load all ONNX models. Call once after models are downloaded."""
        p = lambda name: os.path.join(self.model_dir, f'{name}.onnx')
        self.face_det = InsightFaceDetONNX(p('insightface_det'))
        self.lmk106 = Landmark106ONNX(p('landmark106'))
        self.lmk203 = Landmark203ONNX(p('landmark203'))
        self.app_ext = AppearanceExtractorONNX(p('appearance_extractor'))
        self.mot_ext = MotionExtractorONNX(p('motion_extractor'))
        self.hubert = HubertONNX(p('hubert'))
        self.lmdm = LMDM_ONNX(p('lmdm_v0.4_hubert'))
        self.stitch_net = StitchNetworkONNX(p('stitch_network'))
        self.warp_net = WarpNetworkONNX(p('warp_network_ori'))
        self.decoder = DecoderONNX(p('decoder'))

        self.source2info = Source2Info(self.face_det, self.lmk106, self.lmk203,
                                       self.app_ext, self.mot_ext)
        self.wav2feat = Wav2Feat(self.hubert)
        self.audio2motion = Audio2Motion(self.lmdm)
        self.condition_handler = ConditionHandler()
        self._models_loaded = True
        print('[ditto] All ONNX models loaded successfully')

    def prepare(self, img_rgb_list: List[NDArray], audio_16k: NDArray,
                sampling_timesteps: int = 50, emo: int = 4):
        """Pre-compute all motion parameters from source frames + audio.

        Args:
            img_rgb_list: Source video frames as RGB uint8 arrays.
            audio_16k: Audio waveform at 16 kHz, float32.
            sampling_timesteps: LMDM diffusion steps (lower = faster, less quality).
            emo: Emotion index (0-7). 4 = neutral.
        """
        if not self._models_loaded:
            raise RuntimeError('Models not loaded. Call load_models() first.')

        # 1. Register source avatar
        self.source_info = register_avatar(self.source2info, img_rgb_list)
        self.num_source_frames = len(self.source_info['x_s_info_lst'])
        is_image = self.source_info['is_image_flag']

        # 2. Audio → HuBERT features
        print('[ditto] Extracting HuBERT audio features…')
        aud_feat = self.wav2feat.wav2feat(audio_16k)
        num_output_frames = len(aud_feat)
        print(f'[ditto] Audio → {num_output_frames} frames @ 25fps')

        # 3. Setup condition handler
        self.condition_handler.setup(self.source_info, emo=emo,
                                     eye_f0_mode=not is_image)

        # 4. Build full conditioning sequence
        aud_cond_all = self.condition_handler(aud_feat, 0)

        # 5. Audio → motion via LMDM diffusion
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(x_s_info_0, overlap=10,
                                sampling_timesteps=sampling_timesteps, smo_k=3)
        res_kp_seq = self.audio2motion.run_offline(aud_cond_all)

        # 6. Convert to per-frame motion dicts
        self.x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)

        # 7. Setup motion stitch state for video mode
        x_s_info = self.source_info['x_s_info_lst'][0]
        if is_image:
            self._use_d_keys = ('exp', 'pitch', 'yaw', 'roll', 't')
            self._drive_eye = True
        else:
            self._use_d_keys = ('exp',)
            self._drive_eye = False
        self._relative_d = True
        self._d0 = None
        self._flag_stitching = True

        # Pre-compute source keypoints for image mode
        if is_image:
            self._x_s_static = _transform_keypoint(x_s_info)
        else:
            self._x_s_static = None

        print(f'[ditto] Pipeline ready: {num_output_frames} output frames')
        return num_output_frames

    def render_frame(self, frame_number: int) -> Optional[NDArray]:
        """Render a single output frame using pre-computed motion.

        Args:
            frame_number: Output frame index (0-based).

        Returns:
            Composited RGB frame, or None if frame_number is out of range.
        """
        if self.x_d_info_list is None or self.source_info is None:
            return None
        if frame_number >= len(self.x_d_info_list):
            return None

        # Source frame index (mirror-loop for video)
        src_idx = _mirror_index(frame_number, self.num_source_frames)
        x_s_info = self.source_info['x_s_info_lst'][src_idx]
        x_d_info = self.x_d_info_list[frame_number]

        # Relative motion: driven = source + (driven - d0)
        if self._relative_d and self._d0 is None:
            self._d0 = copy.deepcopy(self.x_d_info_list[0])

        x_d_info = _mix_s_d_info(x_s_info, x_d_info, self._use_d_keys, self._d0)

        # For video: only drive lips, keep source eyes
        if not self._drive_eye:
            x_d_info = _fix_exp_for_video(x_d_info, x_s_info)

        # Transform keypoints
        if self._x_s_static is not None:
            x_s = self._x_s_static
        else:
            x_s = _transform_keypoint(x_s_info)
        x_d = _transform_keypoint(x_d_info)

        # Stitch
        if self._flag_stitching:
            x_d = self.stitch_net(
                x_s.astype(np.float32),
                x_d.astype(np.float32),
            )

        # Warp
        f_s = self.source_info['f_s_lst'][src_idx]
        f_3d = self.warp_net(
            f_s.astype(np.float32),
            x_s.astype(np.float32),
            x_d.astype(np.float32),
        )

        # Decode
        render_img = self.decoder(f_3d.astype(np.float32))

        # Putback onto original frame
        frame_rgb = self.source_info['img_rgb_lst'][src_idx]
        M_c2o = self.source_info['M_c2o_lst'][src_idx]
        result = self.putback(frame_rgb, render_img, M_c2o)
        return result

    def teardown(self):
        """Release memory."""
        self.source_info = None
        self.x_d_info_list = None
        self._d0 = None
        self._models_loaded = False
