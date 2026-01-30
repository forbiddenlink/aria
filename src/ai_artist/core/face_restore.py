"""Face restoration using GFPGAN for fixing distorted/creepy faces."""

from pathlib import Path

from PIL import Image

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FaceRestorer:
    """Restore faces in images using GFPGAN.

    GFPGAN is a practical face restoration algorithm that excels at:
    - Fixing distorted facial features
    - Enhancing facial details
    - Correcting asymmetrical eyes
    - Improving overall face quality
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.restorer = None
        logger.info("face_restorer_initialized", device=device)

    def _load_model(self):
        """Lazy load GFPGAN model."""
        if self.restorer is not None:
            return True

        try:
            from gfpgan import GFPGANer

            # Default model path
            model_path = self.model_path or "models/gfpgan/GFPGANv1.4.pth"

            # Download model if not exists
            if not Path(model_path).exists():
                logger.info("downloading_gfpgan_model")
                import httpx

                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    Path(model_path).write_bytes(response.content)

            self.restorer = GFPGANer(
                model_path=model_path,
                upscale=1,  # Don't upscale, just restore
                arch="clean",
                channel_multiplier=2,
                device=self.device,
            )
            logger.info("gfpgan_model_loaded", path=model_path)
            return True

        except ImportError:
            logger.warning(
                "gfpgan_not_installed",
                message="Install with: pip install gfpgan",
                hint="Face restoration will be skipped",
            )
            return False
        except Exception as e:
            logger.error("gfpgan_load_failed", error=str(e))
            return False

    def restore(self, image: Image.Image) -> Image.Image:
        """Restore faces in an image.

        Args:
            image: PIL Image to process

        Returns:
            Image with restored faces (or original if no faces/error)
        """
        if not self._load_model():
            return image

        try:
            import cv2
            import numpy as np

            # Convert PIL to cv2 format (BGR)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Restore faces
            _, _, restored_img = self.restorer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )

            # Convert back to PIL (RGB)
            if restored_img is not None:
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(restored_rgb)
                logger.info("faces_restored")
                return result
            else:
                logger.debug("no_faces_detected")
                return image

        except Exception as e:
            logger.error("face_restoration_failed", error=str(e))
            return image

    def has_faces(self, image: Image.Image) -> bool:
        """Check if image contains faces.

        Args:
            image: PIL Image to check

        Returns:
            True if faces detected
        """
        try:
            import cv2
            import numpy as np

            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Use OpenCV face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            return len(faces) > 0

        except Exception:
            return False
