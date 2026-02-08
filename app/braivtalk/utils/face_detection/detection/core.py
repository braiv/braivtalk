import logging
import glob
from tqdm import tqdm
import numpy as np
import torch
import cv2


class FaceDetector(object):
    """An abstract class representing a face detector."""

    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose

        if verbose:
            if "cpu" in device:
                logger = logging.getLogger(__name__)
                logger.warning("Detection running on CPU, this may be potentially slow.")

        if "cpu" not in device and "cuda" not in device:
            if verbose:
                logger.error("Expected values for device are: {cpu, cuda} but got: %s", device)
            raise ValueError

    def detect_from_image(self, tensor_or_path):
        raise NotImplementedError

    def detect_from_directory(self, path, extensions=[".jpg", ".png"], recursive=False, show_progress_bar=True):
        if self.verbose:
            logger = logging.getLogger(__name__)

        if len(extensions) == 0:
            if self.verbose:
                logger.error("Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = "/**/*" if recursive else "/*"
        files = []
        for extension in extensions:
            files.extend(glob.glob(path + additional_pattern + extension, recursive=recursive))

        if self.verbose:
            logger.info("Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}
        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info("Running the face detector on image: %s", image_path)
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info("The detector was successfully run on all %s images", len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else cv2.imread(tensor_or_path)[..., ::-1]
        elif torch.is_tensor(tensor_or_path):
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError

