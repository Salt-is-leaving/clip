from config import DEVICE, MODEL_NAME, PRETRAINED, BATCH_SIZE, CHECKPOINT_DIR
import logging
import config
import torch
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
import wandb
from tqdm import tqdm

# Logger call
logger = logging.getLogger(__name__)

class CLIPModel:
    def __init__(self):
        logger.info("Initializing CLIPModel")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = DEVICE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.load_model()  

    def load_model(self): 
        model_name = config.MODEL_NAME  # Changed to use config
        pretrained = config.PRETRAINED
        
        logger.info(f"Attempting to load model: {MODEL_NAME}")
        try:
            logger.info("Creating model and transforms...")
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device
            )
            logger.info("Model created successfully")

            logger.info("Getting tokenizer...")
            self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            logger.info("Tokenizer loaded successfully")
            logger.info(f"Moving model to {self.device}")

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model setup complete")

        except Exception as e:
            logger.error(f"Error during model loading: {str(e)}")
            raise

    def encode_text(self, text, batch_size=BATCH_SIZE):
        logger.info("Encoding text")
        if isinstance(text, str):
            text = [text]
        
        text_embeddings_list = []
        
        with torch.set_grad_enabled(False):  # Only disable gradients during inference
            for i in range(0, len(text), batch_size):
                text_batch = text[i:i + batch_size]
                text_tokens = self.tokenizer(text_batch).to(self.device)
                batch_text_embeddings = self.model.encode_text(text_tokens)
                text_embeddings_list.append(batch_text_embeddings.cpu().numpy())
            
        if len(text_embeddings_list) > 1:
            return np.concatenate(text_embeddings_list)
        return text_embeddings_list[0]
                
    def encode_image(self, image):
        logger.info("Encoding image")
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device) #add 1D to position 0
        logger.info(f"Processed image shape: {processed_image.shape}")

        with torch.set_grad_enabled(False):  # Only disable gradients during inference
            image_embedding = self.model.encode_image(processed_image)
         # Add shape verification, assert [1, 512]
            logger.info(f"Encoded image embedding shape: {image_embedding.shape}")
            image_embedding = image_embedding.cpu().numpy()
            image_embedding = image_embedding.squeeze()  #safer without arg,removes batch 1D if present
            assert image_embedding.shape[0] == 512, (
                f"Invalid embedding shape: {image_embedding.shape}. Expected (512,)"
            )
        return image_embedding
      
    def encode_batch(self, batch, batch_size=BATCH_SIZE):
        logger.info("Encoding batch")
        if isinstance(batch[0], (str, Path)) and Path(batch[0]).suffix.lower() in ['.jpg', '.png', '.jpeg']:
            return self.encode_image_batch(batch, batch_size)
        else:
            return self.encode_text_batch(batch, batch_size)

    def _encode_image_batch(self, batch, batch_size):
        logger.info("Encoding image batch")
        embeddings = []
        for i in tqdm(range(0, len(batch), batch_size), desc="Processing images"):
            batch_data = batch[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}")
            processed_images = torch.stack([
                self.preprocess(Image.open(img).convert('RGB')) for img in batch_data
            ]).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(processed_images).cpu().numpy()
                logger.info(f"Encoded image batch shape: {batch_embeddings.shape}")
                embeddings.append(batch_embeddings)
        return np.concatenate(embeddings)

    def encode_text_batch(self, batch, batch_size):
        logger.info("Encoding text batch")
        embeddings = []
        for i in range(0, len(batch), batch_size):
            batch_data = batch[i:i + batch_size]
            text_tokens = self.tokenizer(batch_data).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model.encode_text(text_tokens).cpu().numpy()
                logger.info(f"Encoded text batch shape: {batch_embeddings.shape}")
                embeddings.append(batch_embeddings)
        return np.concatenate(embeddings)
            
if __name__ == "__main__":
    logger.info("Starting model script")
    print("Available models:")
    print(open_clip.list_models())
    print("\nAvailable pretrained weights:")
    print(open_clip.list_pretrained())
    
    logger.info("\nTesting model initialization...")
    try:
        model = CLIPModel()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")