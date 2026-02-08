import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from braivtalk.utils.blending import get_image
from braivtalk.utils.face_parsing import FaceParsing
from braivtalk.utils.audio_processor import AudioProcessor
from braivtalk.utils.utils import get_file_type, get_video_fps, datagen_enhanced, load_all_model

from braivtalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, read_video_frames, coord_placeholder
from braivtalk.utils.parallel_io import ParallelFrameWriter

# Import GPEN-BFR face enhancer (optional)
try:
    from braivtalk.enhancers.gpen_bfr_enhancer import GPENBFREnhancer
    GPEN_BFR_AVAILABLE = True
except Exception as e:
    print("Warning: GPEN-BFR not available:", str(e))
    GPEN_BFR_AVAILABLE = False


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:  # v1
        fp = FaceParsing()

    # Initialize GPEN-BFR face enhancer if enabled
    gpen_bfr_enhancer = None
    if args.enable_gpen_bfr and GPEN_BFR_AVAILABLE:
        try:
            print("🎨 Initializing GPEN-BFR face enhancer...")
            gpen_bfr_enhancer = GPENBFREnhancer(
                model_path=args.gpen_bfr_model_path,
                device="auto",
                config_name=args.gpen_bfr_config,
            )
            print(f"GPEN-BFR initialized with config: {args.gpen_bfr_config}")
        except Exception as e:
            print(f"Warning: GPEN-BFR initialization failed: {e}")
            print("   Continuing without face enhancement...")
            gpen_bfr_enhancer = None
    elif args.enable_gpen_bfr and not GPEN_BFR_AVAILABLE:
        print("Warning: GPEN-BFR requested but not available. Install dependencies with:")
        print("   pip install onnx onnxruntime-gpu scipy psutil")
        print("   Continuing without face enhancement...")

    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)

    # Process each task
    for task_id in inference_config:
        try:
            # Get task configuration
            task_config = inference_config[task_id]
            video_path = task_config["video_path"]
            audio_path = task_config["audio_path"]
            if "result_name" in task_config:
                args.output_vid_name = task_config["result_name"]

            # 🎯 MERGE YAML PARAMETERS WITH COMMAND LINE ARGUMENTS
            mouth_params = [
                'ellipse_padding_factor', 'upper_boundary_ratio', 'expand_factor',
                'use_elliptical_mask', 'blur_kernel_ratio', 'mouth_vertical_offset',
                'mouth_scale_factor', 'debug_mouth_mask', 'mask_shape',
                'mask_height_ratio', 'mask_corner_radius',
            ]

            # YOLOv8 face selection parameters
            yolo_params = [
                'yolo_conf_threshold', 'yolo_temporal_weight', 'yolo_size_weight',
                'yolo_center_weight', 'yolo_max_face_jump', 'yolo_primary_face_lock_threshold',
                'yolo_primary_face_confidence_drop',
            ]

            for param in mouth_params + yolo_params:
                if param in task_config:
                    setattr(args, param, task_config[param])
                    print(f"Using YAML parameter: {param} = {task_config[param]}")
                    if param == 'mask_shape':
                        print(f"DEBUG: mask_shape set to: {args.mask_shape}")

            # REINITIALIZE FACE DETECTOR WITH YOLO PARAMETERS
            yolo_config_found = any(param in task_config for param in yolo_params)
            if yolo_config_found:
                from braivtalk.utils.preprocessing import init_face_detector
                print("Reinitializing face detector with YAML parameters...")
                init_face_detector(
                    use_yolo=True,
                    yolo_conf_threshold=getattr(args, 'yolo_conf_threshold', 0.5),
                    yolo_temporal_weight=getattr(args, 'yolo_temporal_weight', 0.25),
                    yolo_size_weight=getattr(args, 'yolo_size_weight', 0.30),
                    yolo_center_weight=getattr(args, 'yolo_center_weight', 0.20),
                    yolo_max_face_jump=getattr(args, 'yolo_max_face_jump', 0.3),
                    yolo_primary_face_lock_threshold=getattr(args, 'yolo_primary_face_lock_threshold', 10),
                    yolo_primary_face_confidence_drop=getattr(args, 'yolo_primary_face_confidence_drop', 0.8),
                )

            # Set bbox_shift based on version
            if args.version == "v15":
                bbox_shift = 0  # v15 uses fixed bbox_shift
            else:
                bbox_shift = task_config.get("bbox_shift", args.bbox_shift)

            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"

            # Create temporary directories
            temp_dir = os.path.join(args.result_dir, f"{args.version}")
            os.makedirs(temp_dir, exist_ok=True)

            # Set result save paths
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename + ".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)

            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)

            # Read frames from source (direct-to-memory for video, skips FFmpeg disk I/O)
            if get_file_type(video_path) == "video":
                frame_list, fps = read_video_frames(video_path)
            elif get_file_type(video_path) == "image":
                frame_list = [cv2.imread(video_path)]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                frame_list = read_imgs(input_img_list)
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

            # Extract audio features
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )

            # Preprocess: detect faces (or load cached coordinates)
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    # Handle both old format (just coords) and new format (coords + landmarks)
                    if isinstance(saved_data, tuple) and len(saved_data) == 2:
                        coord_list, landmarks_list = saved_data
                    else:
                        coord_list = saved_data
                        landmarks_list = [None] * len(saved_data)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, landmarks_list = get_landmark_and_bbox(
                    frame_list, bbox_shift, detection_interval=args.detection_interval
                )
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump((coord_list, landmarks_list), f)

            print(f"Number of frames: {len(frame_list)}")

            # Process each frame - Enhanced version that handles ALL frames
            input_latent_list = []
            passthrough_frames = {}
            processed_frame_count = 0

            for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder:
                    input_latent_list.append(None)
                    passthrough_frames[i] = frame
                    print(f"Frame {i}: No face detected - will use passthrough")
                else:
                    processed_frame_count += 1
                    x1, y1, x2, y2 = bbox
                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents = vae.get_latents_for_unet(crop_frame)
                    input_latent_list.append(latents)

            print(
                f"Enhanced processing: {processed_frame_count} frames with faces, "
                f"{len(passthrough_frames)} passthrough frames"
            )

            # Smooth first and last frames
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Enhanced batch inference with support for passthrough frames
            print("Starting enhanced inference")
            video_num = len(whisper_chunks)
            batch_size = args.batch_size

            gen = datagen_enhanced(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                coord_list_cycle=coord_list_cycle,
                frame_list_cycle=frame_list_cycle,
                passthrough_frames=passthrough_frames,
                coord_placeholder=coord_placeholder,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )

            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))

            for _, batch_data in enumerate(tqdm(gen, total=total)):
                if 'process_whisper_batch' in batch_data and 'process_latent_batch' in batch_data:
                    whisper_batch = batch_data['process_whisper_batch']
                    latent_batch = batch_data['process_latent_batch']

                    audio_feature_batch = pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=unet.model.dtype)

                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents)

                    # Optional GPEN-BFR enhancement
                    if gpen_bfr_enhancer is not None:
                        try:
                            enhanced_faces = []
                            for face_idx in range(recon.shape[0]):
                                face_image = recon[face_idx]
                                enhanced_face = gpen_bfr_enhancer.enhance_face(face_image)
                                enhanced_faces.append(enhanced_face)
                            recon = np.stack(enhanced_faces, axis=0)
                        except Exception as e:
                            print(f"Warning: GPEN-BFR enhancement failed for batch: {e}")
                            print("   Using original VAE output...")

                    process_idx = 0
                    for process_type in batch_data['process_types']:
                        if process_type == 'process':
                            res_frame_list.append(recon[process_idx])
                            process_idx += 1
                        else:
                            res_frame_list.append(None)
                else:
                    for _process_type in batch_data['process_types']:
                        res_frame_list.append(None)

            # Enhanced frame output with parallel I/O for better performance
            print("Processing enhanced frame output (lip-sync + passthrough + parallel I/O)")
            frame_writer = ParallelFrameWriter(max_workers=4)

            with ThreadPoolExecutor(max_workers=4) as io_executor:
                for i, res_frame in enumerate(tqdm(res_frame_list, desc="Processing frames")):
                    bbox = coord_list_cycle[i % (len(coord_list_cycle))]
                    ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])

                    if bbox == coord_placeholder or res_frame is None:
                        combine_frame = ori_frame
                        if i < 10:
                            print(f"Frame {i}: Using passthrough (no face)")
                    else:
                        x1, y1, x2, y2 = bbox
                        if args.version == "v15":
                            y2 = y2 + args.extra_margin
                            y2 = min(y2, ori_frame.shape[0])

                        try:
                            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))

                            frame_landmarks = None
                            if landmarks_list and i < len(landmarks_list):
                                frame_landmarks = landmarks_list[i % (len(landmarks_list))]

                            debug_output_dir = None
                            if args.debug_mouth_mask:
                                debug_base_dir = os.path.join(os.getcwd(), "debug_mouth_masks")
                                debug_output_dir = os.path.join(debug_base_dir, f"{args.version}_{output_basename}")
                                os.makedirs(debug_output_dir, exist_ok=True)
                                print(f"🔍 Debug outputs will be saved to: {debug_output_dir}")

                            combine_frame = get_image(
                                ori_frame,
                                res_frame,
                                [x1, y1, x2, y2],
                                upper_boundary_ratio=args.upper_boundary_ratio,
                                expand=args.expand_factor,
                                mode=args.parsing_mode,
                                fp=fp,
                                use_elliptical_mask=args.use_elliptical_mask,
                                ellipse_padding_factor=args.ellipse_padding_factor,
                                blur_kernel_ratio=args.blur_kernel_ratio,
                                landmarks=frame_landmarks,
                                mouth_vertical_offset=args.mouth_vertical_offset,
                                mouth_scale_factor=args.mouth_scale_factor,
                                debug_mouth_mask=args.debug_mouth_mask,
                                debug_frame_idx=i,
                                debug_output_dir=debug_output_dir,
                                mask_shape=args.mask_shape,
                                mask_height_ratio=args.mask_height_ratio,
                                mask_corner_radius=args.mask_corner_radius,
                            )
                        except Exception as e:
                            print(f"Frame {i}: Processing failed, using passthrough - {e}")
                            combine_frame = ori_frame

                    frame_writer.write_frame_async(io_executor, i, combine_frame, result_img_save_path)

                frame_writer.wait_for_completion()

            # Save prediction results
            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            result_img_save_path_fixed = result_img_save_path.replace("\\", "/")
            temp_vid_path_fixed = temp_vid_path.replace("\\", "/")

            if args.force_cpu_encoding or not args.gpu_encoding:
                encoders = [
                    {
                        "name": "CPU (libx264)",
                        "cmd": (
                            f"ffmpeg -y -v warning -r {fps} -f image2 "
                            f"-i {result_img_save_path_fixed}/%08d.png "
                            f"-vcodec libx264 -preset {args.encoding_preset} -crf {args.encoding_crf} "
                            f"-pix_fmt yuv420p {temp_vid_path_fixed}"
                        ),
                    }
                ]
            else:
                encoders = [
                    {
                        "name": "NVENC (NVIDIA)",
                        "cmd": (
                            f"ffmpeg -y -v warning -r {fps} -f image2 "
                            f"-i {result_img_save_path_fixed}/%08d.png "
                            f"-c:v h264_nvenc -preset {args.encoding_preset} -crf {args.encoding_crf} "
                            f"-pix_fmt yuv420p {temp_vid_path_fixed}"
                        ),
                    },
                    {
                        "name": "VCE (AMD)",
                        "cmd": (
                            f"ffmpeg -y -v warning -r {fps} -f image2 "
                            f"-i {result_img_save_path_fixed}/%08d.png "
                            f"-c:v h264_amf -rc cqp -qp {args.encoding_crf} "
                            f"-pix_fmt yuv420p {temp_vid_path_fixed}"
                        ),
                    },
                    {
                        "name": "Quick Sync (Intel)",
                        "cmd": (
                            f"ffmpeg -y -v warning -r {fps} -f image2 "
                            f"-i {result_img_save_path_fixed}/%08d.png "
                            f"-c:v h264_qsv -preset {args.encoding_preset} -crf {args.encoding_crf} "
                            f"-pix_fmt yuv420p {temp_vid_path_fixed}"
                        ),
                    },
                    {
                        "name": "CPU (libx264)",
                        "cmd": (
                            f"ffmpeg -y -v warning -r {fps} -f image2 "
                            f"-i {result_img_save_path_fixed}/%08d.png "
                            f"-vcodec libx264 -preset {args.encoding_preset} -crf {args.encoding_crf} "
                            f"-pix_fmt yuv420p {temp_vid_path_fixed}"
                        ),
                    },
                ]

            encoding_successful = False
            for encoder in encoders:
                print(f"Trying {encoder['name']} encoding...")
                try:
                    subprocess.run(encoder['cmd'], shell=True, check=True, capture_output=True, text=True)
                    print(f"Video encoding successful with {encoder['name']}")
                    encoding_successful = True
                    break
                except subprocess.CalledProcessError as e:
                    print(f"WARNING: {encoder['name']} failed: {e.stderr if e.stderr else str(e)}")
                    if encoder != encoders[-1]:
                        print("   Trying next encoding option...")
                        continue
                    else:
                        print("ERROR: All encoding options failed")
                        return

            if not encoding_successful:
                print("ERROR: Video generation failed with all encoding methods")
                return

            audio_path_fixed = audio_path.replace("\\", "/")
            output_vid_name_fixed = output_vid_name.replace("\\", "/")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path_fixed} -i {temp_vid_path_fixed} {output_vid_name_fixed}"
            print("Audio combination command:", cmd_combine_audio)
            try:
                subprocess.run(cmd_combine_audio, shell=True, check=True)
                print("Audio combination successful")
                print(f"Final video created: {output_vid_name_fixed}")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Audio combination failed: {e}")
                return

            # Clean up temporary files
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)

            if not args.saved_coord:
                os.remove(crop_coord_save_path)

            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=-7, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference (conservative optimization)")
    parser.add_argument("--detection_interval", type=int, default=1, help="Face detection interval: detect every N frames, interpolate the rest (1=every frame, 5=5x faster detection)")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")

    # Mouth Overlay Accuracy Parameters
    parser.add_argument("--ellipse_padding_factor", type=float, default=0.1, help="Ellipse padding factor for mouth mask (smaller = larger mouth coverage, 0.06-0.15)")
    parser.add_argument("--upper_boundary_ratio", type=float, default=0.5, help="Upper boundary ratio for face replacement (smaller = more face coverage, 0.3-0.7)")
    parser.add_argument("--expand_factor", type=float, default=1.5, help="Face crop expansion factor (larger = more context, 1.2-1.8)")
    parser.add_argument("--use_elliptical_mask", action="store_true", default=True, help="Use elliptical mask instead of rectangular (recommended)")
    parser.add_argument("--blur_kernel_ratio", type=float, default=0.05, help="Blur kernel size ratio for mask smoothing (0.02-0.08)")
    parser.add_argument("--mouth_vertical_offset", type=float, default=0.0, help="Vertical offset for mouth positioning (positive = lower, negative = higher, -0.05 to +0.05)")
    parser.add_argument("--mouth_scale_factor", type=float, default=1.0, help="Scale factor for mouth size matching (1.0 = exact YOLOv8 size, 1.1-1.3 = larger for better coverage)")
    parser.add_argument("--debug_mouth_mask", action="store_true", help="Save debug outputs: isolated AI mouth, mask, and overlay visualization")
    parser.add_argument("--mask_shape", type=str, default="ellipse", choices=["ellipse", "triangle", "rounded_triangle", "wide_ellipse", "ultra_wide_ellipse", "dynamic_contour"], help="Shape of the blending mask")
    parser.add_argument("--mask_height_ratio", type=float, default=0.4, help="Height ratio for mask relative to mouth width")
    parser.add_argument("--mask_corner_radius", type=float, default=0.2, help="Corner radius for rounded shapes")

    # GPEN-BFR Face Enhancement Parameters
    parser.add_argument("--enable_gpen_bfr", action="store_true", help="Enable GPEN-BFR face enhancement")
    parser.add_argument("--gpen_bfr_model_path", type=str, default="models/gpen_bfr/gpen_bfr_256.onnx", help="Path to GPEN-BFR model")
    parser.add_argument(
        "--gpen_bfr_config",
        type=str,
        default="CONSERVATIVE",
        choices=["NATURAL", "BALANCED", "QUALITY_FOCUSED", "CONSERVATIVE", "DRAMATIC", "SKIN_FOCUS", "DETAIL_ENHANCE", "LIPS_OPTIMIZED"],
        help="GPEN-BFR enhancement configuration preset",
    )

    # GPU Video Encoding Parameters
    parser.add_argument("--gpu_encoding", action="store_true", default=True, help="Enable GPU-accelerated video encoding (default: True)")
    parser.add_argument("--force_cpu_encoding", action="store_true", help="Force CPU encoding (overrides GPU encoding)")
    parser.add_argument(
        "--encoding_preset",
        type=str,
        default="fast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="Encoding preset (speed vs quality tradeoff)",
    )
    parser.add_argument("--encoding_crf", type=int, default=18, help="Constant Rate Factor for quality (lower = better quality, 0-51)")

    args = parser.parse_args()

    # Handle encoding preferences
    if args.force_cpu_encoding:
        args.gpu_encoding = False
        print("Forced CPU encoding enabled")
    elif args.gpu_encoding:
        print("GPU encoding enabled (will fallback to CPU if needed)")

    main(args)

