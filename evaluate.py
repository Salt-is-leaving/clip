from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import wandb

from config import (
    DEVICE, BATCH_SIZE, VECTOR_DIM,
    LANGUAGES, CHECKPOINT_DIR, DB_PATH, INDEX_PATH
)
from model import CLIPModel
from data_process import DataProcessor
from db import ImageDatabase, AnnoyDatabase

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model, image_db, annoy_db):
        self.model = model
        self.image_db = image_db
        self.annoy_db = annoy_db
        self.logger = logging.getLogger(__name__)
        
    def evaluate_retrieval(self, test_data, k_values=[5]):
        """Evaluate image-text retrieval performance
        Args:
            test_data (list): List of test data items
            k_values (list): Top-k values for retrieval accuracy
        Returns:
            dict: Retrieval evaluation metrics
        """
        empty_result = {'metrics': {}, 'embeddings': {'image': [], 'text': []}}
        
        # Combined type and emptiness check
        if not isinstance(test_data, list) or not test_data:
            self.logger.error(f"Invalid or empty test_data: {type(test_data)}")
            return empty_result
            
        # Combined structure validation
        if not all(isinstance(item, dict) and all(key in item for key in ['image', 'eng']) 
                for item in test_data):
            self.logger.error("Invalid test data structure or missing required keys")
            return empty_result

        self.logger.info("Generating embeddings for test data...")
        results = {
            'embeddings': {'image': [], 'text': []},
            'modalities': [],
            'metrics': {}
        }

        # 1. Generate embeddings with indexing
        for idx, item in enumerate(tqdm(test_data, desc="Generating Embeddings")):
            try:
                image_emb = self.model.encode_image(item['image'])
                text_emb = self.model.encode_text(item['eng'])

                # Track embeddings with their modalities
                results['embeddings']['image'].append(image_emb)
                results['embeddings']['text'].append(text_emb)
                results['modalities'].append({
                    'image_id': f"img_{idx}",
                    'text_id': f"txt_{idx}"
                })
            
            except Exception as e:
                self.logger.error(f"Error processing item {idx+1}: {str(e)}")
                continue

        # Convert lists of embeddings to 2D numpy arrays
        # np.vstack throws an error if shapes are inconsistent, ensuring shape integrity
        try:
            results['embeddings']['image'] = np.vstack(results['embeddings']['image'])
            results['embeddings']['text'] = np.vstack(results['embeddings']['text'])
        except ValueError as ve:
            self.logger.error(f"Inconsistent embedding shapes: {ve}")
            return None

        # Compute cross-modal similarity matrices
        self.logger.info("Computing cross-modal similarities...")
        sim_matrix = cosine_similarity(
            results['embeddings']['text'],
            results['embeddings']['image']
        )

        # Evaluate retrieval performance with modality tracking
        for k in k_values:
            # Text-to-Image Retrieval
            t2i_acc = self.calculate_retrieval_accuracy(
                sim_matrix, k, 'text', 'image'
            )
            results['metrics'][f'text_to_image_top{k}'] = t2i_acc
            self.logger.info(f'Text-to-Image Top-{k} Accuracy: {t2i_acc:.4f}')

        # Log modality-aware metrics
        if wandb.run:
            wandb.log({
                **results['metrics'],
                'modality_distribution': {
                    'image_count': len(results['embeddings']['image']),
                    'text_count': len(results['embeddings']['text'])
                }
            })
        return results

    def calculate_retrieval_accuracy(self, similarity_matrix, k, input_modality, output_modality):
        """
        Calculate top-k retrieval accuracy from similarity matrix
        Args:
            similarity_matrix (numpy.ndarray): Similarity matrix
            k (int): Top-k value
        Returns:
            float: Retrieval accuracy
        """
        n = similarity_matrix.shape[0]
        correct = 0
        
        for i in range(n):
            top_indices = np.argsort(similarity_matrix[i])[-k:]
            if i in top_indices:
                correct += 1       
        return correct / n
    
        """
        Create a bar plot of retrieval performance
        Args:
            metrics (dict): Retrieval metrics
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.title("Retrieval Performance")
            plt.xlabel("Metrics")
            plt.ylabel("Top-k Accuracy")
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            plt.bar(metric_names, metric_values)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig('retrieval_performance.png')
            plt.close()
        
        except Exception as e:
            self.logger.warning(f"Could not create retrieval performance plot: {e}")
    def evaluate_zero_shot(self, test_data):
        """
        Evaluate zero-shot classification on test data.
        Args:
            test_data (list): List of test data items with 'category' key.
        Returns:
            float: Zero-shot classification accuracy.
        """
        self.logger.info("Starting zero-shot evaluation...")
    
        if not test_data or 'category' not in test_data[0]:
            self.logger.error("Test data missing or no category information")
            return None
        
        results = {'correct': 0, 'total': 0, 'per_category': {}}
        categories = list({item['category'] for item in test_data})  # Dynamically infer categories

        for item in tqdm(test_data, desc="Evaluating zero-shot"):
            try:
                image_emb = self.model.encode_image(item['image'])
                category_embs = [
                    self.model.encode_text(f"a photo of a {category}") for category in categories
                ]
                similarities = cosine_similarity(
                    image_emb.reshape(1, -1), np.vstack(category_embs)
                )[0]
                predicted_category = categories[np.argmax(similarities)]

                if predicted_category == item['category']:
                    results['correct'] += 1
                results['total'] += 1

            except Exception as e:
                self.logger.error(f"Error processing item: {e}")
                continue

        if results['total'] == 0:
            self.logger.error("No items were successfully processed")
            return 0

        accuracy = results['correct'] / results['total']
        self.logger.info(f"Zero-shot evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy
    
    def plot_retrieval_performance(self, metrics):
        """
        Create a simple bar plot of retrieval performance
        
        Args:
            metrics (dict): Dictionary containing retrieval metrics
        """
        try:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            save_path = CHECKPOINT_DIR / "retrieval_performance.png"

            plt.figure(figsize=(10, 6))
            plt.title('Retrieval Performance')
            
            # Plot bars
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            plt.bar(metric_names, metric_values)
            
            # Customize appearance
            plt.xlabel('Metrics')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save and log
            plt.savefig('retrieval_performance.png')
            plt.close()
            
            if wandb.run:
                wandb.log({"retrieval_plot": wandb.Image('retrieval_performance.png')})
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
    
    def plot_training_progress(self, metrics_history):
        """Plot training metrics over epochs
        Args:
            metrics_history: List of dicts containing per-epoch metrics
        """
        try:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CHECKPOINT_DIR / "training_progress.png"

            plt.figure(figsize=(15, 5))
            
            # Plot 1: Loss
            plt.subplot(131)
            epochs = [m['epoch'] for m in metrics_history]
            losses = [m['train_loss'] for m in metrics_history]
            plt.plot(epochs, losses, 'b-', label='Training Loss')
            plt.title('Training Loss vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Plot 2: Top-5 Accuracy
            plt.subplot(132)
            r5_scores = [m['r5_score'] for m in metrics_history]
            plt.plot(epochs, r5_scores, 'g-', label='Top-5 Score')
            plt.title('Top-5 Score vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True)
            plt.legend()
            
            # Plot 3: Learning Rate
            plt.subplot(133)
            lrs = [m['learning_rate'] for m in metrics_history]
            plt.plot(epochs, lrs, 'r-', label='Learning Rate')
            plt.title('Learning Rate vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            if wandb.run:
                wandb.log({"training_progress": wandb.Image(str(save_path))})
                
        except Exception as e:
            self.logger.error(f"Error creating training progress plot: {e}") 
def main():
    try:
        logger.info("Initializing evaluation components...")
        model = CLIPModel()
        processor = DataProcessor()
        image_db = ImageDatabase(DB_PATH)
        annoy_db = AnnoyDatabase(VECTOR_DIM, INDEX_PATH)
        evaluator = Evaluator(model, image_db, annoy_db)
        
        # Load test data and categories
        test_data = processor.process_dataset(split='test')
        if not test_data:
            logger.error("No test data found!")
            return
        
        retrieval_results = evaluator.evaluate_retrieval(test_data)
        zero_shot_accuracy = evaluator.evaluate_zero_shot(test_data)
        
        if retrieval_results and zero_shot_accuracy is not None:
            logger.info("Evaluation completed successfully")
        else:
            logger.error("Evaluation failed")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()