# train.py
import torch
import random
from tqdm import tqdm
from PIL import Image
from config import DEVICE, ETA, EPOCHS, BATCH_SIZE
from model import CLIPModel
from data_process import DataProcessor
from evaluate import Evaluator
from db import AnnoyDatabase  # Add this import
import logging

class Trainer:
    def __init__(self, model, device=DEVICE, logger=None):
        self.model = model # CLIPModel instance
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=ETA
        )
        self.logger = logger or logging.getLogger(__name__)  # Fallback if no logger provided
        self.annoy_db = None

    def set_annoy_db(self, annoy_db):
        """Initialize AnnoyDB instance for training"""
        if not isinstance(annoy_db, AnnoyDatabase):
            raise TypeError("Expected AnnoyDatabase instance")
        
        self.annoy_db = annoy_db
        self.logger.info(f"AnnoyDB initialized with status: {self.annoy_db.get_index_info()}")
    
    def compute_loss(self, image_embeddings, text_embeddings):
        """Compute contrastive loss between image and text embeddings"""
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.T) * 20.0  # Temperature scaling
        
        # Contrastive loss in both directions
        labels = torch.arange(len(logits)).to(self.device)
        loss = (torch.nn.functional.cross_entropy(logits, labels) + 
                torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        return loss
    
    def train_epoch(self, train_data):
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0
        num_batches = 0
        
        # Process data in batches
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            
            try: 
            #THIS VERIFICATION BLOCK:
                images = []
                texts = []
                for sample in batch:
                    # Verify and load image
                    image = sample['image']
                    if isinstance(image, str):
                        image = Image.open(image)
                    images.append(self.model.preprocess(image).to(self.device))
                    texts.append(sample['eng'])
                    
                # Stack verified images into tensor
                images = torch.stack(images)

                # Convert texts to tokens and move to DEVICE
                text_tokens = self.model.tokenizer(texts).to(self.device)
                
                # Rest of the training code remains the same...
                self.optimizer.zero_grad()
                image_embeddings = self.model.model.encode_image(images)
                text_embeddings = self.model.model.encode_text(text_tokens)
                
                loss = self.compute_loss(image_embeddings, text_embeddings)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else None
        
        def train(self, train_data, evaluator, test_data, save_checkpoint_fn, epochs=EPOCHS):
            """Main training loop with evaluation"""
        if not self.annoy_db:
            raise ValueError("AnnoyDB not set. Call set_annoy_db before training.")

            # Add batch size validation
        if len(train_data) < BATCH_SIZE:
            self.model.model.train()
            batch = train_data if isinstance(train_data, list) else [train_data]
            avg_loss = self.train_epoch(batch)
            return {'train_loss': avg_loss, 'batch_size': len(batch)}

        best_r1 = 0.0  # Best R@1 score
        validation_samples = 100  # Number of validation samples for interim evaluation
        
        for epoch in range(1, EPOCHS + 1):
            # Training phase
            avg_loss = self.train_epoch(train_data)
            if avg_loss is None:
                self.logger.error(f"Training epoch {epoch} failed")
                continue

            # Dynamic sampling for validation
            validation_data = random.sample(test_data, min(len(test_data), validation_samples))
            
            # Evaluation phase
            results = evaluator.evaluate_retrieval(validation_data)
            if not results or 'metrics' not in results:
                self.logger.error("Evaluation failed to return valid results")
                continue
                
            current_r1 = results['metrics']['text_to_image_top1']
           
            # Save checkpoint if best R@1 score
            if current_r1 > best_r1:
                best_r1 = current_r1
                # Validate embeddings before saving
                if 'embeddings' in results and all(key in results['embeddings'] 
                                             for key in ['image', 'text']):
                    save_checkpoint_fn(
                        model=self.model,
                        annoy_db=self.annoy_db, 
                        current_id=epoch,
                        embeddings={
                            'image': [(i, emb) for i, emb in enumerate(results['embeddings']['image'])],
                            'text': [(i, emb) for i, emb in enumerate(results['embeddings']['text'])]
                        }
                    )
                
            # Return metrics for logging
            metrics = {
                'epoch': epoch,
                'train_loss': avg_loss,
                'r1_score': current_r1,
                'best_r1': best_r1,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.logger.info(f"\nEpoch {epoch} Metrics: {metrics}")

        return metrics