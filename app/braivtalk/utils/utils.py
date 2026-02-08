import os
import cv2
import torch

from braivtalk.models.vae import VAE
from braivtalk.models.unet import UNet, PositionalEncoding


def load_all_model(
    unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
    vae_type="sd-vae",
    unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
    device=None,
):
    vae = VAE(
        model_path=os.path.join("models", vae_type),
    )
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device,
    )
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        return "image"
    elif ext.lower() in [".avi", ".mp4", ".mov", ".flv", ".mkv"]:
        return "video"
    else:
        return "unsupported"


def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def datagen_enhanced(
    whisper_chunks,
    vae_encode_latents,
    coord_list_cycle,
    frame_list_cycle,
    passthrough_frames,
    coord_placeholder=(0.0, 0.0, 0.0, 0.0),
    batch_size=12,
    delay_frame=0,
    device="cuda:0",
):
    """
    Enhanced datagen that handles both lip-sync processing and passthrough frames.
    Returns batches with processing type information.
    """
    whisper_batch, latent_batch, frame_batch, process_type_batch = [], [], [], []

    for i, w in enumerate(whisper_chunks):
        idx = (i + delay_frame) % len(coord_list_cycle)
        coord = coord_list_cycle[idx]

        if coord == coord_placeholder:
            original_idx = idx % len(frame_list_cycle)
            passthrough_frame = frame_list_cycle[original_idx]

            whisper_batch.append(w)
            latent_batch.append(None)
            frame_batch.append(passthrough_frame)
            process_type_batch.append("passthrough")
        else:
            latent_idx = idx % len(vae_encode_latents)
            latent = vae_encode_latents[latent_idx]

            whisper_batch.append(w)
            latent_batch.append(latent)
            frame_batch.append(None)
            process_type_batch.append("process")

        if len(whisper_batch) >= batch_size:
            yield create_enhanced_batch(whisper_batch, latent_batch, frame_batch, process_type_batch, device)
            whisper_batch, latent_batch, frame_batch, process_type_batch = [], [], [], []

    if len(whisper_batch) > 0:
        yield create_enhanced_batch(whisper_batch, latent_batch, frame_batch, process_type_batch, device)


def create_enhanced_batch(whisper_batch, latent_batch, frame_batch, process_type_batch, device):
    process_items = [
        (w, l)
        for w, l, t in zip(whisper_batch, latent_batch, process_type_batch)
        if t == "process" and l is not None
    ]
    passthrough_items = [
        (w, f)
        for w, f, t in zip(whisper_batch, frame_batch, process_type_batch)
        if t == "passthrough" and f is not None
    ]

    batch_data = {
        "process_items": process_items,
        "passthrough_items": passthrough_items,
        "process_types": process_type_batch,
        "device": device,
    }

    if process_items:
        process_whispers, process_latents = zip(*process_items)
        batch_data["process_whisper_batch"] = torch.stack(list(process_whispers)).to(device)
        batch_data["process_latent_batch"] = torch.cat(list(process_latents), dim=0).to(device)

    return batch_data

