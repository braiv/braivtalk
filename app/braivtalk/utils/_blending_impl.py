from PIL import Image, ImageDraw
import numpy as np
import cv2
import copy


def _color_transfer_lab(source_rgb, target_rgb, mask=None):
    """Reinhard LAB color transfer: match source color stats to target.

    Adjusts the mean and standard deviation of each LAB channel in *source*
    to match those of *target*, so the pasted region has the same skin tone,
    brightness, and color temperature as the surrounding original face.

    Args:
        source_rgb: The AI-generated face region (uint8 RGB numpy array).
        target_rgb: The original face region (uint8 RGB numpy array).
        mask: Optional uint8 grayscale mask; only pixels > 127 contribute
              to the target statistics (allows ignoring background).

    Returns:
        Color-corrected source as uint8 RGB numpy array.
    """
    if source_rgb.size == 0 or target_rgb.size == 0:
        return source_rgb

    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    if mask is not None and mask.shape[:2] == tgt_lab.shape[:2]:
        roi = mask > 127
        if roi.sum() < 10:
            # Not enough pixels to compute meaningful stats
            return source_rgb
        tgt_pixels = tgt_lab[roi]
    else:
        tgt_pixels = tgt_lab.reshape(-1, 3)

    src_pixels = src_lab.reshape(-1, 3)

    for ch in range(3):
        s_mean, s_std = src_pixels[:, ch].mean(), src_pixels[:, ch].std() + 1e-6
        t_mean, t_std = tgt_pixels[:, ch].mean(), tgt_pixels[:, ch].std() + 1e-6
        src_lab[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)

# Module-level cache for face parsing results.
# BiSeNet output barely changes between consecutive frames, so we can
# safely reuse the mask for several frames and only refresh periodically.
_parsing_cache = {
    "mask_small": None,
    "call_count": 0,
}


def reset_parsing_cache():
    """Reset the parsing cache (call between tasks/videos)."""
    _parsing_cache["mask_small"] = None
    _parsing_cache["call_count"] = 0


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None, use_elliptical_mask=True, ellipse_padding_factor=0.1, blur_kernel_ratio=0.05, landmarks=None, mouth_vertical_offset=0.0, mouth_scale_factor=1.0, debug_mouth_mask=False, debug_frame_idx=None, debug_output_dir=None, mask_shape="ellipse", mask_height_ratio=0.4, mask_corner_radius=0.2, parsing_interval=1, blend_mode="alpha"):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。
    Enhanced with landmark-based surgical mouth positioning for improved accuracy.

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 
        use_elliptical_mask (bool): 是否使用椭圆形掩码而不是矩形掩码。
        ellipse_padding_factor (float): 椭圆掩码的内边距因子，控制椭圆相对于面部边界的大小。
        blur_kernel_ratio (float): 高斯模糊核大小比例，用于平滑掩码边缘。
        landmarks (list): YOLOv8 facial landmarks [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]
        mouth_vertical_offset (float): Vertical offset for mouth positioning (positive = lower, negative = higher)
        mouth_scale_factor (float): Scale factor for mouth size matching (1.0 = exact YOLOv8 size, >1.0 = larger, <1.0 = smaller)
        debug_mouth_mask (bool): Save debug outputs for troubleshooting
        debug_frame_idx (int): Frame index for debug file naming
        debug_output_dir (str): Directory to save debug outputs
        mask_shape (str): Shape of blending mask ("ellipse", "triangle", "rounded_triangle", "wide_ellipse", "ultra_wide_ellipse")
        mask_height_ratio (float): Height ratio for mask relative to mouth width (0.3-0.8)
        mask_corner_radius (float): Corner radius for rounded shapes (0.0-0.5)
        blend_mode (str): "alpha" for fast alpha compositing (default) or "poisson" for cv2.seamlessClone

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # Face parsing with caching: only run BiSeNet every parsing_interval frames.
    # The parsing mask (face skin vs. background) barely changes between consecutive
    # frames, so reusing it is safe and gives a major speedup.
    expected_size = (x1 - x, y1 - y)
    run_fresh = (
        _parsing_cache["mask_small"] is None
        or parsing_interval <= 1
        or _parsing_cache["call_count"] % parsing_interval == 0
    )

    if run_fresh:
        mask_image_parsed = face_seg(face_large, mode=mode, fp=fp)
        mask_small = mask_image_parsed.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        _parsing_cache["mask_small"] = mask_small
    else:
        mask_small = _parsing_cache["mask_small"]
        # Resize cached mask if face bbox dimensions changed slightly
        if mask_small.size != expected_size:
            mask_small = mask_small.resize(expected_size, Image.BILINEAR)

    _parsing_cache["call_count"] += 1
    
    # Create mask with surgical precision using landmarks if available
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    
    if landmarks is not None and len(landmarks) >= 5:
        # SURGICAL POSITIONING: Use YOLOv8 landmarks for precise mouth region
        left_eye, right_eye, nose_tip, left_mouth, right_mouth = landmarks
        
        # Calculate mouth-specific region for surgical precision
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        
        # Enhanced mouth corner analysis for better positioning
        mouth_corner_height_diff = abs(left_mouth[1] - right_mouth[1])
        mouth_angle = mouth_corner_height_diff / max(mouth_width, 1)  # Prevent division by zero
        
        # Calculate nose-to-mouth distance for proportional sizing
        nose_to_mouth_dist = abs(nose_tip[1] - mouth_center_y)
        
        # Create landmark-based elliptical mask focused on mouth region
        face_width = x1 - x
        face_height = y1 - y
        
        # Convert global landmarks to local face coordinates
        local_mouth_center_x = mouth_center_x - x
        local_mouth_center_y = mouth_center_y - y
        local_left_mouth_x = left_mouth[0] - x
        local_right_mouth_x = right_mouth[0] - x
        local_mouth_y = left_mouth[1] - y  # Use left mouth Y (they should be similar)
        
        # Apply vertical offset for fine-tuning mouth position
        # Positive offset moves mouth down, negative moves it up
        offset_pixels = mouth_vertical_offset * face_height  # Convert ratio to pixels
        local_mouth_center_y += offset_pixels
        local_mouth_y += offset_pixels
        
        # Create surgical mouth mask
        mouth_mask = Image.new('L', (face_width, face_height), 0)
        draw = ImageDraw.Draw(mouth_mask)
        
        # DYNAMIC MOUTH SIZING: Match AI mouth to original YOLOv8 detected mouth size
        # Base mouth region on actual detected mouth width with scale factor
        base_mouth_width = mouth_width * mouth_scale_factor  # Apply user-defined scaling
        mouth_region_width = base_mouth_width * (1.0 + ellipse_padding_factor * 2)  # Add padding
        
        # ADVANCED MASK SHAPES: Calculate dimensions based on mask shape
        if mask_shape == "ultra_wide_ellipse":
            # Ultra wide ellipse: MAXIMUM coverage for challenging cases
            mouth_region_width = mouth_region_width * 1.8  # 80% wider than standard
            mouth_region_height = base_mouth_width * mask_height_ratio * 1.8  # Maximum height
        elif mask_shape == "wide_ellipse":
            # Wide ellipse: MUCH wider and taller for maximum lip coverage
            mouth_region_width = mouth_region_width * 1.4  # 40% wider than standard
            mouth_region_height = base_mouth_width * mask_height_ratio * 1.6  # Extra height
        elif mask_shape in ["triangle", "rounded_triangle"]:
            # Triangle shapes: height based on natural face geometry
            mouth_region_height = base_mouth_width * mask_height_ratio * 1.2
        else:
            # Standard ellipse: original calculation
            mouth_region_height = base_mouth_width * mask_height_ratio
        
        # Alternative height calculation using nose-mouth distance (more conservative)
        nose_based_height = nose_to_mouth_dist * 0.8
        
        # Use the larger of the two height calculations for better coverage
        mouth_region_height = max(mouth_region_height, nose_based_height)
        
        # Ensure minimum size for very small faces (safety net)
        mouth_region_width = max(mouth_region_width, face_width * 0.25)
        mouth_region_height = max(mouth_region_height, face_height * 0.15)
        
        # ------------------------------------------------------------------
        # CHIN-AWARE ASYMMETRIC POSITIONING
        # ------------------------------------------------------------------
        # The mask must cover from just above the upper lip down to the chin.
        # YOLO doesn't provide chin landmarks, but facial proportions give us
        # a reliable estimate: chin_bottom ~= mouth_center + nose_to_mouth * 1.2
        #
        # Instead of centering the ellipse on the mouth (which leaves the chin
        # exposed), we compute explicit top/bottom boundaries:
        #   top  = mouth_center - (small portion above upper lip)
        #   bottom = estimated chin position (+ padding)
        #
        # The ellipse center is then shifted downward to match these bounds.
        # ------------------------------------------------------------------
        estimated_chin_y = local_mouth_center_y + nose_to_mouth_dist * 1.2
        # Small overshoot below the chin to guarantee full coverage
        chin_padding = nose_to_mouth_dist * 0.15
        desired_bottom = estimated_chin_y + chin_padding

        # Top of the mask: just above the upper lip area
        # (~40% of nose-to-mouth distance above mouth center)
        desired_top = local_mouth_center_y - nose_to_mouth_dist * 0.4

        # Use the larger of: shape-based height vs chin-aware height
        chin_aware_height = desired_bottom - desired_top
        if chin_aware_height > mouth_region_height:
            mouth_region_height = chin_aware_height

        # Compute asymmetric mask bounds (biased downward toward chin)
        # The ellipse center shifts down so bottom reaches the chin.
        mask_center_y = (desired_top + desired_bottom) / 2
        # But never shift higher than the original mouth center
        mask_center_y = max(mask_center_y, local_mouth_center_y)

        mask_left = local_mouth_center_x - mouth_region_width / 2
        mask_top = mask_center_y - mouth_region_height / 2
        mask_right = local_mouth_center_x + mouth_region_width / 2
        mask_bottom = mask_center_y + mouth_region_height / 2
        
        # Ensure mask stays within face bounds
        mask_left = max(0, mask_left)
        mask_top = max(0, mask_top)
        mask_right = min(face_width, mask_right)
        mask_bottom = min(face_height, mask_bottom)
        
        # DRAW MASK BASED ON SHAPE
        if mask_shape == "ellipse":
            # Standard ellipse
            draw.ellipse([mask_left, mask_top, mask_right, mask_bottom], fill=255)
            
        elif mask_shape == "ultra_wide_ellipse":
            # Ultra wide ellipse - maximum coverage for challenging cases
            draw.ellipse([mask_left, mask_top, mask_right, mask_bottom], fill=255)
            
        elif mask_shape == "wide_ellipse":
            # Wide ellipse - much wider and taller
            draw.ellipse([mask_left, mask_top, mask_right, mask_bottom], fill=255)
            
        elif mask_shape == "triangle":
            # Upside-down triangle (natural face shape)
            triangle_points = [
                (local_mouth_center_x, mask_top),  # Top center point
                (mask_left, mask_bottom),          # Bottom left
                (mask_right, mask_bottom)          # Bottom right
            ]
            draw.polygon(triangle_points, fill=255)
            
        elif mask_shape == "rounded_triangle":
            # Upside-down triangle with rounded corners (most natural)
            # Create triangle path with rounded corners
            
            # Create a temporary high-res mask for smooth curves
            temp_size = (int(face_width * 2), int(face_height * 2))
            temp_mask = Image.new('L', temp_size, 0)
            temp_draw = ImageDraw.Draw(temp_mask)
            
            # Scale coordinates for high-res
            scale = 2.0
            t_center_x = local_mouth_center_x * scale
            t_center_y = local_mouth_center_y * scale
            t_width = mouth_region_width * scale
            t_height = mouth_region_height * scale
            
            # Calculate rounded triangle points
            corner_radius = mask_corner_radius * min(t_width, t_height) * 0.5
            
            # Top point (mouth center, slightly above)
            top_x = t_center_x
            top_y = mask_top * scale
            
            # Bottom corners
            left_x = mask_left * scale
            right_x = mask_right * scale
            bottom_y = mask_bottom * scale
            
            # Draw rounded triangle using multiple shapes
            # Main triangle body
            triangle_points = [
                (top_x, top_y + corner_radius),
                (left_x + corner_radius, bottom_y - corner_radius),
                (right_x - corner_radius, bottom_y - corner_radius)
            ]
            temp_draw.polygon(triangle_points, fill=255)
            
            # Add rounded corners
            # Top corner
            temp_draw.ellipse([top_x - corner_radius, top_y, 
                             top_x + corner_radius, top_y + corner_radius * 2], fill=255)
            
            # Bottom left corner
            temp_draw.ellipse([left_x, bottom_y - corner_radius * 2,
                             left_x + corner_radius * 2, bottom_y], fill=255)
            
            # Bottom right corner  
            temp_draw.ellipse([right_x - corner_radius * 2, bottom_y - corner_radius * 2,
                             right_x, bottom_y], fill=255)
            
            # Scale back down and paste
            temp_mask = temp_mask.resize((face_width, face_height), Image.LANCZOS)
            mouth_mask.paste(temp_mask, (0, 0))
            
        elif mask_shape == "dynamic_contour":
            # DYNAMIC CONTOUR: Follow natural face geometry using landmarks
            # This creates a mask that follows the jawline and chin contours
            
            # Calculate face geometry from landmarks
            face_center_x = float((left_eye[0] + right_eye[0]) / 2)
            face_center_y = float((left_eye[1] + right_eye[1]) / 2)
            
            # Calculate face dimensions
            eye_distance = float(abs(right_eye[0] - left_eye[0]))
            face_width_estimate = float(eye_distance * 2.2)  # Typical face width ratio
            
            # Calculate jawline points based on face geometry
            jaw_width = float(face_width_estimate * 0.8)  # Jawline is narrower than face
            chin_y = float(local_mouth_center_y + nose_to_mouth_dist * 1.2)  # Chin below mouth
            
            # Create dynamic contour points
            contour_points = []
            
            # Top arc (above mouth, following upper lip curve)
            top_y = float(local_mouth_center_y - mouth_region_height * 0.3)
            for i in range(11):  # 11 points for smooth curve
                angle = (i / 10.0) * np.pi  # 0 to π
                x = local_mouth_center_x + (mouth_region_width / 2) * np.cos(angle)
                y = top_y - (mouth_region_height * 0.1) * np.sin(angle)  # Slight curve
                contour_points.append((int(x), int(y)))
            
            # Right side (following jawline)
            right_jaw_x = float(local_mouth_center_x + jaw_width / 2)
            for i in range(5):  # 5 points down the right jaw
                t = i / 4.0  # 0 to 1
                x = right_jaw_x - (jaw_width * 0.1) * t  # Slight inward curve
                y = top_y + (chin_y - top_y) * t
                contour_points.append((int(x), int(y)))
            
            # Bottom arc (chin contour)
            chin_width = float(jaw_width * 0.6)  # Chin is narrower than jaw
            for i in range(11):  # 11 points for chin curve
                angle = np.pi * (i / 10.0)  # π to 0 (right to left)
                x = local_mouth_center_x + (chin_width / 2) * np.cos(angle)
                y = chin_y - (mouth_region_height * 0.2) * abs(np.sin(angle))  # Rounded chin
                contour_points.append((int(x), int(y)))
            
            # Left side (following jawline)
            left_jaw_x = float(local_mouth_center_x - jaw_width / 2)
            for i in range(5):  # 5 points up the left jaw
                t = (4 - i) / 4.0  # 1 to 0
                x = left_jaw_x + (jaw_width * 0.1) * t  # Slight inward curve
                y = top_y + (chin_y - top_y) * t
                contour_points.append((int(x), int(y)))
            
            # Draw the dynamic contour polygon
            if len(contour_points) > 2:
                # Ensure all coordinates are integers for PIL compatibility
                int_contour_points = [(int(x), int(y)) for x, y in contour_points]
                draw.polygon(int_contour_points, fill=255)
                
                # Add debug info for dynamic contour
                if debug_mouth_mask:
                    print(f"Dynamic contour: {len(contour_points)} points, jaw_width={jaw_width:.1f}px, chin_y={chin_y:.1f}px")
        
        # Combine ellipse with BiSeNet face-parsing mask.
        # The ellipse defines the mouth region of interest but may not reach the
        # chin.  BiSeNet provides a pixel-accurate face boundary (class 1=skin
        # includes the chin).  We use np.maximum (union) so that BiSeNet can
        # EXTEND the mask to the chin where the ellipse falls short.
        # The upper_boundary_ratio step later crops the top of the mask, so
        # using union here won't bleed into the forehead/eye area.
        mouth_array = np.array(mouth_mask)
        mask_small_array = np.array(mask_small)
        # Ensure same size (BiSeNet mask covers face bbox = mouth_mask canvas)
        if mask_small_array.shape != mouth_array.shape:
            mask_small_resized = cv2.resize(
                mask_small_array, (mouth_array.shape[1], mouth_array.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            mask_small_resized = mask_small_array

        combined_mask = np.maximum(mouth_array, mask_small_resized)
        final_face_mask = Image.fromarray(combined_mask)
        
        # Paste the surgical landmark-based mask with bounds checking
        paste_x = x - x_s
        paste_y = y - y_s
        
        # Ensure paste coordinates are valid
        if paste_x >= 0 and paste_y >= 0:
            # Check if the mask fits within the target image
            target_w, target_h = mask_image.size
            mask_w, mask_h = final_face_mask.size
            
            if paste_x + mask_w <= target_w and paste_y + mask_h <= target_h:
                mask_image.paste(final_face_mask, (paste_x, paste_y))
            else:
                # Crop the mask to fit if it's too large
                crop_w = min(mask_w, target_w - paste_x)
                crop_h = min(mask_h, target_h - paste_y)
                if crop_w > 0 and crop_h > 0:
                    cropped_mask = final_face_mask.crop((0, 0, crop_w, crop_h))
                    mask_image.paste(cropped_mask, (paste_x, paste_y))
        else:
            print(f"WARNING: Invalid paste coordinates ({paste_x}, {paste_y}) - skipping mask paste")
        
        # Only log surgical positioning every 50 frames to avoid console spam
        if _parsing_cache["call_count"] % 50 == 1:
            offset_info = f", offset {mouth_vertical_offset:+.2f}" if mouth_vertical_offset != 0.0 else ""
            scale_info = f", scale {mouth_scale_factor:.2f}" if mouth_scale_factor != 1.0 else ""
            print(f"Surgical positioning: mouth center ({mouth_center_x:.1f}, {mouth_center_y + offset_pixels:.1f}), width {mouth_width:.1f}px->{base_mouth_width:.1f}px{offset_info}{scale_info}")
        
    elif use_elliptical_mask:
        # Fallback: Create elliptical mask for more natural blending (original method)
        face_width = x1 - x
        face_height = y1 - y
        
        # Create elliptical mask for the face region
        ellipse_mask = Image.new('L', (face_width, face_height), 0)
        draw = ImageDraw.Draw(ellipse_mask)
        
        # Calculate padding to make ellipse smaller than face bounds
        padding_w = int(face_width * ellipse_padding_factor)
        padding_h = int(face_height * ellipse_padding_factor)
        
        # Draw ellipse (white = include area, black = exclude)
        draw.ellipse([padding_w, padding_h, face_width - padding_w, face_height - padding_h], fill=255)
        
        # Apply the face parsing mask to the elliptical mask (intersection)
        ellipse_array = np.array(ellipse_mask)
        mask_small_array = np.array(mask_small)
        combined_mask = np.minimum(ellipse_array, mask_small_array)
        final_face_mask = Image.fromarray(combined_mask)
        
        # Paste the combined elliptical + parsing mask with bounds checking
        paste_x = x - x_s
        paste_y = y - y_s
        
        # Ensure paste coordinates are valid
        if paste_x >= 0 and paste_y >= 0:
            # Check if the mask fits within the target image
            target_w, target_h = mask_image.size
            mask_w, mask_h = final_face_mask.size
            
            if paste_x + mask_w <= target_w and paste_y + mask_h <= target_h:
                mask_image.paste(final_face_mask, (paste_x, paste_y))
            else:
                # Crop the mask to fit if it's too large
                crop_w = min(mask_w, target_w - paste_x)
                crop_h = min(mask_h, target_h - paste_y)
                if crop_w > 0 and crop_h > 0:
                    cropped_mask = final_face_mask.crop((0, 0, crop_w, crop_h))
                    mask_image.paste(cropped_mask, (paste_x, paste_y))
        else:
            print(f"Warning: Invalid paste coordinates ({paste_x}, {paste_y}) - skipping elliptical mask paste")
        
        if _parsing_cache["call_count"] % 50 == 1:
            print(f"Fallback: elliptical mask (no landmarks available)")
    else:
        # Original rectangular mask behavior with bounds checking
        paste_x = x - x_s
        paste_y = y - y_s
        paste_x1 = x1 - x_s
        paste_y1 = y1 - y_s
        
        # Ensure paste coordinates are valid
        if paste_x >= 0 and paste_y >= 0 and paste_x1 > paste_x and paste_y1 > paste_y:
            target_w, target_h = mask_image.size
            if paste_x1 <= target_w and paste_y1 <= target_h:
                mask_image.paste(mask_small, (paste_x, paste_y, paste_x1, paste_y1))
            else:
                print(f"Warning: Rectangular mask bounds ({paste_x}, {paste_y}, {paste_x1}, {paste_y1}) exceed target size ({target_w}, {target_h})")
        else:
            print(f"Warning: Invalid rectangular mask coordinates ({paste_x}, {paste_y}, {paste_x1}, {paste_y1}) - skipping mask paste")
        if _parsing_cache["call_count"] % 50 == 1:
            print(f"Basic: rectangular mask")
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # --- Step 1: Erode top/sides to keep the blend inside the AI face ---
    mask_np = np.array(modified_mask_image)
    erode_size = max(3, int(blur_kernel_ratio * ori_shape[0] * 0.4) // 2 * 2 + 1)
    eroded = cv2.erode(
        mask_np,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
        iterations=1,
    )

    # --- Step 2: Chin overspill -- push bottom edge PAST chin line ---
    # Find the vertical midpoint of actual mask content (not the canvas).
    rows_with_content = np.where(mask_np.max(axis=1) > 0)[0]
    if len(rows_with_content) > 0:
        content_mid_y = (rows_with_content[0] + rows_with_content[-1]) // 2
    else:
        content_mid_y = mask_np.shape[0] // 2

    # Dilate only the lower half of the mask content (chin/jaw area).
    # Use a vertically-biased kernel that's narrow horizontally (hugs the
    # face contour) but tall vertically (pushes well past the chin).
    chin_push = erode_size * 2 + max(6, int(height * 0.05))
    chin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, erode_size // 2), chin_push * 2 + 1)
    )
    # Compose: use eroded mask on top, but dilated-from-original on bottom
    # so the chin gets MORE coverage, not less.
    bottom_original = mask_np[content_mid_y:, :]
    bottom_dilated = cv2.dilate(bottom_original, chin_kernel, iterations=1)
    mask_np = eroded.copy()
    mask_np[content_mid_y:, :] = np.maximum(eroded[content_mid_y:, :], bottom_dilated)

    # --- Step 3: Distance-transform feathering ---
    # Instead of a fixed Gaussian blur (which creates a uniform thin feather
    # zone), use distance-transform to compute each pixel's distance to the
    # mask edge.  Normalise over a feather_width band to create a smooth
    # gradient that perfectly follows the contour shape.
    feather_width = max(15, int(ori_shape[0] * 0.06))  # ~6% of face width
    # Binary mask for distance transform (foreground = mask > 0)
    binary = (mask_np > 127).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    # Normalise: 0 at edge -> 1.0 at feather_width pixels inside
    alpha = np.clip(dist / feather_width, 0.0, 1.0)
    # Apply a smooth ease curve (hermite / smoothstep) for natural falloff
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    mask_array = (alpha * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    
    # --- Color matching: transfer original face color stats onto AI face ---
    # This eliminates the visible edge caused by brightness/tone mismatch.
    try:
        original_face_crop = np.array(body.crop((x, y, x1, y1)))  # RGB
        ai_face_arr = np.array(face)  # RGB
        if original_face_crop.shape[:2] == ai_face_arr.shape[:2]:
            ai_face_arr = _color_transfer_lab(ai_face_arr, original_face_crop)
            face = Image.fromarray(ai_face_arr)
    except Exception:
        pass  # Silently fall back to unmatched face on error

    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    # DEBUG: Save debug outputs if requested
    if debug_mouth_mask and debug_frame_idx is not None and debug_output_dir is not None:
        import os
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # Create a visualization showing the mask overlay
        mask_debug = np.array(mask_image)
        original_full = np.array(body)[:, :, ::-1]  # Full original image
        full_mask = np.zeros((original_full.shape[0], original_full.shape[1]), dtype=np.uint8)
        
        # Place the mask in the correct position on the full image
        x_s, y_s, x_e, y_e = crop_box
        mask_h, mask_w = mask_debug.shape
        
        # Ensure coordinates are within bounds
        img_h, img_w = original_full.shape[:2]
        y_start = max(0, y_s)
        x_start = max(0, x_s)
        y_end = min(img_h, y_s + mask_h)
        x_end = min(img_w, x_s + mask_w)
        
        # Calculate the corresponding mask region
        mask_y_start = max(0, -y_s)
        mask_x_start = max(0, -x_s)
        mask_y_end = mask_y_start + (y_end - y_start)
        mask_x_end = mask_x_start + (x_end - x_start)
        
        # Only assign if we have valid regions
        if y_end > y_start and x_end > x_start and mask_y_end > mask_y_start and mask_x_end > mask_x_start:
            full_mask[y_start:y_end, x_start:x_end] = mask_debug[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        
        # Apply colormap to the full-size mask
        mask_colored = cv2.applyColorMap(full_mask, cv2.COLORMAP_JET)
        
        # Now both images have the same dimensions
        overlay_vis = cv2.addWeighted(original_full, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(f"{debug_output_dir}/frame_{debug_frame_idx:06d}_mask_overlay.png", overlay_vis)
        
        print(f"Debug saved: frame {debug_frame_idx} -> {debug_output_dir}/")
    
    if blend_mode == "poisson":
        # --- Poisson blending via cv2.seamlessClone ---
        # Solves gradient-domain equations for mathematically optimal edge
        # blending.  Slower than alpha compositing but produces the best
        # results for difficult lighting / color conditions.
        try:
            src_bgr = np.array(face_large)[:, :, ::-1]  # RGB -> BGR
            dst_bgr = np.array(body)[:, :, ::-1]
            poisson_mask = np.array(mask_image)

            # seamlessClone needs the source to be placed at a center point
            # in destination coordinates.
            cx = crop_box[0] + src_bgr.shape[1] // 2
            cy = crop_box[1] + src_bgr.shape[0] // 2

            # Clamp center to destination bounds
            dst_h, dst_w = dst_bgr.shape[:2]
            cx = max(src_bgr.shape[1] // 2, min(cx, dst_w - src_bgr.shape[1] // 2 - 1))
            cy = max(src_bgr.shape[0] // 2, min(cy, dst_h - src_bgr.shape[0] // 2 - 1))

            result_bgr = cv2.seamlessClone(
                src_bgr, dst_bgr, poisson_mask, (cx, cy), cv2.NORMAL_CLONE
            )
            return result_bgr  # Already BGR, matches expected return format
        except Exception as e:
            # Fall back to alpha blending if Poisson fails
            if _parsing_cache["call_count"] % 50 == 1:
                print(f"Poisson blending failed, falling back to alpha: {e}")
            body.paste(face_large, crop_box[:2], mask_image)
    else:
        # --- Standard alpha compositing ---
        body.paste(face_large, crop_box[:2], mask_image)

    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box

