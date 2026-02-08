from diffusers import AutoencoderKL
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


class VAE:
    """
    VAE (Variational Autoencoder) wrapper used by the pipeline.
    """

    def __init__(self, model_path="./models/sd-vae-ft-mse/", resized_img=256, use_float16=False):
        self.model_path = model_path
        self.vae = AutoencoderKL.from_pretrained(self.model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)

        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False

        self.scaling_factor = self.vae.config.scaling_factor
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self._resized_img = resized_img
        self._mask_tensor = self.get_mask_tensor()

    def get_mask_tensor(self):
        mask_tensor = torch.zeros((self._resized_img, self._resized_img))
        mask_tensor[: self._resized_img // 2, :] = 1
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1
        return mask_tensor

    def preprocess_img(self, img_name, half_mask=False):
        window = []
        if isinstance(img_name, str):
            window_fnames = [img_name]
            for fname in window_fnames:
                img = cv2.imread(fname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self._resized_img, self._resized_img), interpolation=cv2.INTER_LANCZOS4)
                window.append(img)
        else:
            img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
            window.append(img)

        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))
        x = torch.squeeze(torch.FloatTensor(x))
        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        x = self.transform(x)

        x = x.unsqueeze(0)  # [1, 3, 256, 256]
        x = x.to(self.vae.device)
        return x

    def encode_latents(self, image):
        with torch.no_grad():
            init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents

    def decode_latents(self, latents):
        latents = (1 / self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[..., ::-1]  # RGB to BGR
        return image

    def get_latents_for_unet(self, img):
        ref_image = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref_image)
        ref_image = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref_image)
        latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
        return latent_model_input

