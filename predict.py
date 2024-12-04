from typing import List
from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

CACHE_DIR = "weights"  # Define a folder to cache model files
MODEL_NAME = "vidore/colpali-v1.2"  # Replace with your actual model identifier


class Predictor(BasePredictor):
    def setup(self):
        """
        This method is called once when the predictor is initialized.
        The model and preprocessor are loaded here.
        """
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        # Load the ColPali model
        self.model = ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=CACHE_DIR,  # Cache model files locally
            local_files_only=True
        ).eval()

        # Load the processor
        self.processor = ColPaliProcessor.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
        )
    
    def predict(
        self,
        image: Path = Input(
            description="List of image file paths to extract embeddings."
        ),
    ) -> List[List[float]]:
        """
        Perform inference. Extract image embeddings from the model.

        Parameters:
        - images: A list of image paths (file paths to images).

        Returns:
        - A nested list of image embeddings, one list per image.
        """
        # Load images from file paths as PIL images
        images = [image]
        pil_images = [Image.open(image_path).convert("RGB") for image_path in images]

        # Preprocess images
        batch_images = self.processor.process_images(pil_images).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)

        # Convert embeddings from Torch tensor to Python-native nested list
        return image_embeddings.cpu().tolist()
