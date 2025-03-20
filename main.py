from config import (
    VECTOR_DIM, DB_PATH, INDEX_PATH, CHECKPOINT_DIR, WANDB_PROJECT, DATA_DIR,    
    CHECKPOINT_FREQUENCY, BATCH_SIZE, IMAGE_SIZE, DEVICE, ETA, EPOCHS
)

from model import CLIPModel, open_clip
from data_process import DataProcessor
from db import AnnoyDatabase, ImageDatabase
from search_handler import SearchHandler
from train import Trainer
from evaluate import Evaluator

from datetime import datetime
import os
import gc
from pathlib import Path
import numpy as np
import wandb
import torch
from tqdm import tqdm
import logging
import sys
import shutil

RESUME_TRAINING = True
BACKUP_CHECKPOINTS = True
CHECKPOINT_BACKUP_DIR = CHECKPOINT_DIR / "backups"
MAX_BACKUPS = 3 
MODEL_NAME = "open_clip"

    
# Centralized Logger config, wandb called in main()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CHECKPOINT_DIR / 'training.log')  # File output
        ]
    )
logger = logging.getLogger('clip_training')
logging.info("Starting CLIP training pipeline")

# global processor
processor = None
def initialize_directories(required=False):
    """Initialize required directories only if necessary"""
    try:
        # Data directory is always required
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create processed images directory
        (DATA_DIR / "processed_images").mkdir(parents=True, exist_ok=True)
        
        # Only create checkpoint dirs if explicitly required
        if required:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            if BACKUP_CHECKPOINTS:
                CHECKPOINT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False
    
def setup_cuda_environment():
    """Configure CUDA environment for better error handling and memory management"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
    if torch.cuda.is_available():
        # Enable memory tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Set device to current CUDA device
        torch.cuda.set_device(torch.cuda.current_device())

def recover_from_crash(model, checkpoint_dir, processor):
    """Attempt to recover from a crash by loading the latest checkpoint"""
    try:
        latest_checkpoint = max(checkpoint_dir.glob("checkpoint_*.pt"), default=None)
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint)
            model.model.load_state_dict(checkpoint['model_state'])
            last_processed_id = checkpoint['last_processed_id']
            logger.info(f"Recovered from checkpoint: {latest_checkpoint}")
            return last_processed_id
        return 0
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return 0

def handle_modality(model, data, modality):
    """
    Handle embedding generation based on modality type.
    Args:
        model (CLIPModel): The CLIP model instance
        data: Image or text data to embed
        modality (str): Type of data ('image' or 'text')
    Returns:
        numpy.ndarray: Generated embedding
    """
    try:
        if modality == 'image':
            embedding = model.encode_image(data)
        elif modality == 'text':
            embedding = model.encode_text(data)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
            
        # Validate embedding dimensions
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(embedding)}")

        # Reshape if necessary - ensure 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Shape validation after ensuring 2D
        if embedding.ndim != 2:
            raise ValueError(f"Invalid embedding dimensions: expected 2D array, got {embedding.ndim}D")
           
        if embedding.shape[1] != VECTOR_DIM:
            raise ValueError(f"Invalid embedding dimensions: expected shape (N, {VECTOR_DIM}), got {embedding.shape}")
            
        return embedding
        
    except Exception as e:
        logging.error(f"Embedding generation failed for {modality}: {e}")
        raise
def initialize_databases():
    """Initialize and verify database setup"""
    try:
        # Ensure parent directories exist
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Initialize databases
        image_db = ImageDatabase(DB_PATH)
        annoy_db = AnnoyDatabase(VECTOR_DIM, INDEX_PATH)
        
        # Verify database initialization
        if len(image_db) == 0:
            logger.info("New image database initialized")
        else:
            logger.info(f"Loaded existing image database with {len(image_db)} entries")
            
        # Verify Annoy indices
        if not annoy_db.index_built:
            logger.info("Building new Annoy indices...")
            annoy_db.build_index()
        else:
            logger.info("Loaded existing Annoy indices")      
        return image_db, annoy_db
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def save_checkpoint(model, annoy_db, current_id, embeddings):
    global processor 
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        if BACKUP_CHECKPOINTS:
            CHECKPOINT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{timestamp}.pt"
        logger.info(f"Saving checkpoint to: {checkpoint_path}")
        
         # Validate embedding format before processing
        for modality in ['image', 'text']:
            if modality not in embeddings:
                raise ValueError(f"Missing {modality} embeddings in input data")
            if not isinstance(embeddings[modality], list):
                raise ValueError(f"{modality} embeddings must be a list")
            if not all(isinstance(item, tuple) and len(item) == 2 for item in embeddings[modality]):
                raise ValueError(f"Invalid {modality} embedding format. Each item must be a (id, embedding) tuple")
        
        # Save embeddings with verification
        embedding_data = {
            'image_embeddings': [(id, emb) for id, emb in embeddings['image']],  # Store as tuples bs add_image_embedding() only accepts 3 parameters
            'text_embeddings': [(id, emb) for id, emb in embeddings['text']]
        }
        embedding_path = CHECKPOINT_DIR / f"embeddings_{timestamp}.pt"
        torch.save(embedding_data, embedding_path)
        
        # Verify embedding save
        if not embedding_path.exists():
            raise IOError("Embedding file not created successfully")
        
        # Save checkpoint with verification
        checkpoint_data = {
            'model_state': model.model.state_dict(),
            'last_processed_id': current_id,
            'timestamp': timestamp,
            'data_progress': {
                'train': processor.load_progress().get('train', {}),
                'test': processor.load_progress().get('test', {})
            }
        }
   
        torch.save(checkpoint_data, checkpoint_path)
        if not checkpoint_path.exists():
            raise IOError("Checkpoint file not created successfully")
            
        logger.info(f"Checkpoint saved. Last processed ID: {current_id}")
        
         # Create backups if enabled
        if BACKUP_CHECKPOINTS:
            backup_checkpoint = CHECKPOINT_BACKUP_DIR / f"backup_checkpoint_{timestamp}.pt"
            backup_embedding = CHECKPOINT_BACKUP_DIR / f"backup_embeddings_{timestamp}.pt"
            
            # Copy current files to backup
            shutil.copy2(checkpoint_path, backup_checkpoint)
            shutil.copy2(embedding_path, backup_embedding)
            
            # Cleanup old backups - keep only MAX_BACKUPS most recent pairs
            backup_checkpoints = sorted(CHECKPOINT_BACKUP_DIR.glob("backup_checkpoint_*.pt"))
            backup_embeddings = sorted(CHECKPOINT_BACKUP_DIR.glob("backup_embeddings_*.pt"))
            
            # Remove old backup pairs
            if len(backup_checkpoints) > MAX_BACKUPS:
                for old_ckpt, old_emb in zip(backup_checkpoints[:-MAX_BACKUPS], 
                                           backup_embeddings[:-MAX_BACKUPS]):
                    old_ckpt.unlink()
                    old_emb.unlink()
            
            logger.info(f"Backup created and old backups cleaned up")
        
        # Rebuild indices
        annoy_db.build_index()
        logger.info("Annoy index rebuilt and verified")
        
    except Exception as e:
        logger.error(f"Checkpoint saving failed: {e}")
        raise

def load_checkpoint(model, checkpoint_path):
    """Enhanced checkpoint loading with validation"""
    try:
        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting from scratch.")
            return 0, [], []
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Validate checkpoint data
        required_keys = ['model_state', 'last_processed_id', 'timestamp']
        if not all(key in checkpoint for key in required_keys):
            raise ValueError("Checkpoint missing required keys")
            
        model.model.load_state_dict(checkpoint['model_state'])
        logger.info("Model state loaded successfully")
        
        # Load and validate embeddings
        embedding_path = CHECKPOINT_DIR / f"embeddings_{checkpoint['timestamp']}.pt"
        if embedding_path.exists():
            embedding_data = torch.load(embedding_path)
            image_embeddings = embedding_data.get('image_embeddings', [])
            text_embeddings = embedding_data.get('text_embeddings', [])
            logger.info(f"Loaded {len(image_embeddings)} image and {len(text_embeddings)} text embeddings")
        else:
            logger.warning(f"Embedding file not found: {embedding_path}")
            image_embeddings = []
            text_embeddings = []
        
        return (
            checkpoint['last_processed_id'],
            image_embeddings,
            text_embeddings
        )
        
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {e}")
        raise

def process_batch(model, batch_data, image_db, annoy_db, search_handler):
    """Enhanced batch processing with modality handling"""
    processed_count = 0
    batch_embeddings = {'image': [], 'text': []}  # Reset for each batch
    
    try:
        for item in batch_data:
            # Generate embeddings using modality handler
            image_embedding = handle_modality(model, item['image'], 'image')
            text_embedding_eng = handle_modality(model, item['eng'], 'text')

            # Store in databases - use proper path from DataProcessor
            image_path = str(DATA_DIR / "processed_images" / f"{item['id']}.jpg")
            item_id = image_db.add_item(
                path=image_path,
                eng_caption=item['eng'],
                modality='image'
            )
            
            search_handler.add_image_embedding(item_id, image_embedding)
            search_handler.add_text_embedding(f"{item_id}_eng", text_embedding_eng)
            
            batch_embeddings['image'].append((item_id, image_embedding))
            batch_embeddings['text'].append((f"{item_id}_eng", text_embedding_eng))
            processed_count += 1
            
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise
        
    return processed_count, batch_embeddings


def cleanup_resources(image_db=None, wandb_run=None):
    """Cleanup function for graceful shutdown"""
    try:
        if image_db:
            logger.info("Closing image database connection...")
            image_db.close()
            
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception as cuda_error:
                logger.warning(f"CUDA cleanup warning: {cuda_error}")
      
        if wandb_run:
            try:
                wandb.finish()      
            except Exception as wandb_error:
                logger.warning(f"WandB cleanup warning: {wandb_error}")
         
         # Clean up empty checkpoint directories if requested
        if cleanup_checkpoints: # type: ignore
            if CHECKPOINT_DIR.exists() and not any(CHECKPOINT_DIR.iterdir()):
                try:
                    CHECKPOINT_DIR.rmdir()
                    if BACKUP_CHECKPOINTS and CHECKPOINT_BACKUP_DIR.exists():
                        CHECKPOINT_BACKUP_DIR.rmdir()
                except Exception as e:
                    logger.warning(f"Failed to remove empty checkpoint directories: {e}")
                    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def log_memory_stats():
    """Log CUDA memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"CUDA Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached")

def main():
    global processor
    image_db = None
    wandb_run = None
    saved_image_embeddings = []  
    saved_text_embeddings = []  
    
    try:
        # 1. Initialize components
        logger.info("Initializing components...")
        setup_cuda_environment() 
        wandb.init(project=WANDB_PROJECT,
            config={
                "learning_rate": ETA,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "model_name": MODEL_NAME,
                "pretrained": True  # or False, depending on your requirement
            }
        ) 

        model = CLIPModel()
        processor = DataProcessor(image_size=IMAGE_SIZE)
        image_db, annoy_db = initialize_databases()
        search_handler = SearchHandler(annoy_db, image_db)
        logger.info("Components initialized successfully.")
        
        # 2. Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = Evaluator(model, image_db, annoy_db)
        logger.info("Evaluator initialized.")

        # 3. Initialize trainer with loaded data
        logger.info("Initializing trainer...")
        trainer = Trainer(model=model, 
                          device=DEVICE,
                            logger=logger.getChild('trainer') 
                            )
        trainer.set_annoy_db(annoy_db)
        logger.info("Trainer initialized.")
        
        # 4. Load checkpoint and restore embeddings
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        latest_checkpoint = max(CHECKPOINT_DIR.glob("checkpoint_*.pt"), default=None)
        last_processed_id = 0
        
        if latest_checkpoint:
            last_processed_id, saved_image_embeddings, saved_text_embeddings = load_checkpoint(
                model, latest_checkpoint
            )
            # Restore embeddings to search handler 
            if saved_image_embeddings:
                for item_id, embedding in saved_image_embeddings:
                    search_handler.add_image_embedding(item_id, embedding)
            if saved_text_embeddings:
                for item_id, embedding in saved_text_embeddings:
                    search_handler.add_text_embedding(item_id, embedding)

        # 5. Load datasets ONCE after checkpoint
        logger.info("Loading LAION-COCO dataset...")
        train_data = processor.process_dataset(split='train')
        test_data = processor.process_dataset(split='test')

            # Add immediate feedback:
        if train_data:
            logger.info(f"Successfully loaded train data with {len(train_data)} samples")
        else:
            logger.error("Failed to load train data - returned None")
        if test_data:
            logger.info(f"Successfully loaded test data with {len(test_data)} samples")
        else:
            logger.error("Failed to load test data - returned None")

        # 6. Initialize embeddings for training loop
        current_embeddings = {
                'image': saved_image_embeddings,
                'text': saved_text_embeddings
            }

        # 7. Training loop

        logger.info("Starting training loop...")
        total_batches = len(train_data) // BATCH_SIZE
        logger.info(f"Will process {total_batches} batches with batch size {BATCH_SIZE}")

        total_processed = 0
        max_retries = 3
        
        for batch_start in tqdm(range(0, len(train_data), BATCH_SIZE), desc="Training batches"):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if batch_start < last_processed_id:
                        continue
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                
                    batch_end = min(batch_start + BATCH_SIZE, len(train_data))
                    batch = train_data[batch_start:batch_end]
                    logger.info(f"Processing batch of size {len(batch)}")

                    metrics = trainer.train(
                        train_data=batch,
                        evaluator=evaluator,
                        test_data=test_data,
                        save_checkpoint_fn=save_checkpoint
                    )
                    logger.info(f"Batch training completed. Metrics: {metrics if metrics else 'None'}")
                    processed_count, batch_embeddings = process_batch(
                        model, batch, image_db, annoy_db, search_handler
                    )
                    logger.info(f"Processed {processed_count} items in current batch")

                    if not metrics:
                        logger.error("Training returned no metrics")
                        continue

                    total_processed += processed_count
                    logger.info(f"Total processed so far: {total_processed}/{len(train_data)} samples")

                    current_embeddings['image'].extend(batch_embeddings['image'])
                    current_embeddings['text'].extend(batch_embeddings['text'])
            
                    if wandb.run:
                        wandb.log({
                            'batch_loss': metrics['train_loss'],
                            'processed_samples': total_processed
                        })
            
                    if total_processed % CHECKPOINT_FREQUENCY == 0:
                        logger.info(f"Saving checkpoint at {total_processed} samples...")
                        save_checkpoint(
                            model, annoy_db, batch_start + BATCH_SIZE, current_embeddings
                        )
                    break

                except RuntimeError as e:
                    if "CUDA" in str(e):
                        retry_count += 1
                        logger.warning(f"CUDA error occurred. Attempt {retry_count}/{max_retries}")
                        if retry_count < max_retries:
                            torch.cuda.empty_cache()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.reset_peak_memory_stats()
                            last_processed_id = recover_from_crash(model, CHECKPOINT_DIR, processor)
                            continue
                    raise

        # 8. Final evaluation
        logger.info("Training completed. Starting final evaluation...")
        results = evaluator.evaluate_retrieval(test_data)
        if results and 'metrics' in results:
            for metric, value in results['metrics'].items():
                logger.info(f"{metric}: {value:.4f}")
                if wandb.run:
                    wandb.log({f"final_{metric}": value})
        else:
            logger.error("Final evaluation failed to return valid results")
        
        # Final index build
        search_handler.build_index()
        logger.info(f"Pipeline completed. Total samples processed: {total_processed}")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        if wandb.run:
            wandb.alert(title="Pipeline Error", text=str(e))
        # Clean up resources and remove empty checkpoint dirs if initialization failed
        cleanup_resources(image_db, wandb_run, cleanup_checkpoints=True)
        raise
    finally:
        # Normal cleanup without removing checkpoint dirs
        cleanup_resources(image_db, wandb_run)