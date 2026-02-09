from PIL import Image, ImageDraw
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    seg_image = fp(image, mode=mode)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image


def get_image(
    image,
    face,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    mode="raw",
    fp=None,
    use_elliptical_mask=True,
    ellipse_padding_factor=0.1,
    blur_kernel_ratio=0.05,
    landmarks=None,
    mouth_vertical_offset=0.0,
    mouth_scale_factor=1.0,
    debug_mouth_mask=False,
    debug_frame_idx=None,
    debug_output_dir=None,
    mask_shape="ellipse",
    mask_height_ratio=0.4,
    mask_corner_radius=0.2,
    parsing_interval=1,
    blend_mode="alpha",
):
    # NOTE: This file is copied from the existing pipeline implementation.
    # It contains extensive mask-shape logic; kept as-is for functional parity.
    from braivtalk.utils._blending_impl import get_image as _get_image

    return _get_image(
        image=image,
        face=face,
        face_box=face_box,
        upper_boundary_ratio=upper_boundary_ratio,
        expand=expand,
        mode=mode,
        fp=fp,
        use_elliptical_mask=use_elliptical_mask,
        ellipse_padding_factor=ellipse_padding_factor,
        blur_kernel_ratio=blur_kernel_ratio,
        landmarks=landmarks,
        mouth_vertical_offset=mouth_vertical_offset,
        mouth_scale_factor=mouth_scale_factor,
        debug_mouth_mask=debug_mouth_mask,
        debug_frame_idx=debug_frame_idx,
        debug_output_dir=debug_output_dir,
        mask_shape=mask_shape,
        mask_height_ratio=mask_height_ratio,
        mask_corner_radius=mask_corner_radius,
        parsing_interval=parsing_interval,
        blend_mode=blend_mode,
    )

