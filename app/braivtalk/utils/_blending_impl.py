from PIL import Image, ImageDraw
import numpy as np
import cv2
import copy


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


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None, use_elliptical_mask=True, ellipse_padding_factor=0.1, blur_kernel_ratio=0.05, landmarks=None, mouth_vertical_offset=0.0, mouth_scale_factor=1.0, debug_mouth_mask=False, debug_frame_idx=None, debug_output_dir=None, mask_shape="ellipse", mask_height_ratio=0.4, mask_corner_radius=0.2):
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

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    # Create mask with surgical precision using landmarks if available
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    
    if landmarks is not None and len(landmarks) >= 5:
        # 🎯 SURGICAL POSITIONING: Use YOLOv8 landmarks for precise mouth region
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
        
        # 🎯 DYNAMIC MOUTH SIZING: Match AI mouth to original YOLOv8 detected mouth size
        # Base mouth region on actual detected mouth width with scale factor
        base_mouth_width = mouth_width * mouth_scale_factor  # Apply user-defined scaling
        mouth_region_width = base_mouth_width * (1.0 + ellipse_padding_factor * 2)  # Add padding
        
        # 🎭 ADVANCED MASK SHAPES: Calculate dimensions based on mask shape
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
        
        # Calculate mask bounds centered on actual mouth position
        mask_left = local_mouth_center_x - mouth_region_width / 2
        mask_top = local_mouth_center_y - mouth_region_height / 2
        mask_right = local_mouth_center_x + mouth_region_width / 2
        mask_bottom = local_mouth_center_y + mouth_region_height / 2
        
        # Ensure mask stays within face bounds
        mask_left = max(0, mask_left)
        mask_top = max(0, mask_top)
        mask_right = min(face_width, mask_right)
        mask_bottom = min(face_height, mask_bottom)
        
        # 🎨 DRAW MASK BASED ON SHAPE
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
            top_y = (local_mouth_center_y - mouth_region_height / 2) * scale
            
            # Bottom corners
            left_x = (local_mouth_center_x - mouth_region_width / 2) * scale
            right_x = (local_mouth_center_x + mouth_region_width / 2) * scale
            bottom_y = (local_mouth_center_y + mouth_region_height / 2) * scale
            
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
            # 🎯 DYNAMIC CONTOUR: Follow natural face geometry using landmarks
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
                    print(f"🎯 Dynamic contour: {len(contour_points)} points, jaw_width={jaw_width:.1f}px, chin_y={chin_y:.1f}px")
        
        # Apply face parsing mask for additional refinement
        mouth_array = np.array(mouth_mask)
        mask_small_array = np.array(mask_small)
        combined_mask = np.minimum(mouth_array, mask_small_array)
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
            print(f"⚠️ Warning: Invalid paste coordinates ({paste_x}, {paste_y}) - skipping mask paste")
        
        offset_info = f", offset {mouth_vertical_offset:+.2f}" if mouth_vertical_offset != 0.0 else ""
        scale_info = f", scale {mouth_scale_factor:.2f}" if mouth_scale_factor != 1.0 else ""
        print(f"Surgical positioning: mouth center ({mouth_center_x:.1f}, {mouth_center_y + offset_pixels:.1f}), width {mouth_width:.1f}px→{base_mouth_width:.1f}px{offset_info}{scale_info}")
        
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
        print(f"Basic: rectangular mask")
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(blur_kernel_ratio * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    # 🔍 DEBUG: Save debug outputs if requested
    if debug_mouth_mask and debug_frame_idx is not None and debug_output_dir is not None:
        import os
        os.makedirs(debug_output_dir, exist_ok=True)
        
        # Save the isolated AI face region
        face_debug = np.array(face)[:, :, ::-1]  # Convert to BGR for OpenCV
        cv2.imwrite(f"{debug_output_dir}/frame_{debug_frame_idx:06d}_ai_face.png", face_debug)
        
        # Save the mask
        mask_debug = np.array(mask_image)
        cv2.imwrite(f"{debug_output_dir}/frame_{debug_frame_idx:06d}_mask.png", mask_debug)
        
        # Save the original face region for comparison
        original_face_region = np.array(body.crop((x, y, x1, y1)))[:, :, ::-1]
        cv2.imwrite(f"{debug_output_dir}/frame_{debug_frame_idx:06d}_original_face.png", original_face_region)
        
        # Create a visualization showing the mask overlay
        # Create a full-size mask for visualization
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
        
        print(f"Debug saved: frame {debug_frame_idx} → {debug_output_dir}/")
    
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

