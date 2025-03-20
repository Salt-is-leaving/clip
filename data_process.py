from config import (
    LANGUAGES, DB_PATH, INDEX_PATH, IMAGE_SIZE, DATA_DIR, WDS_DIR, MAX_SAMPLES, CHECKPOINT_DIR, DEVICE, ETA, EPOCHS, BATCH_SIZE
)
from db import ImageDatabase
from pathlib import Path
import json
import torch
import tarfile
from io import BytesIO
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"

class DataProcessor:
    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.db = ImageDatabase(DB_PATH)
        self.processed_count = 0
        self.progress_file = CHECKPOINT_DIR / "processing_progress.json"
        self.images_dir = DATA_DIR / "processed_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def load_progress(self):
        """Load processing progress tracking"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    # Verify progress format
                    if isinstance(progress, dict) and all(key in progress for key in ['train', 'test']):
                        return progress
            except json.JSONDecodeError:
                pass
        # Initialize new progress tracking
        return {
            'train': {'completed_files': [], 'last_processed': None, 'total_items': 0},
            'test': {'completed_files': [], 'last_processed': None, 'total_items': 0}
        }
    
    def save_progress(self, split, tar_file=None, items_processed=0):
        """Save processing progress"""
        progress = self.load_progress()
        
        if tar_file:
            # Update progress for specific file
            if tar_file.name not in progress[split]['completed_files']:
                progress[split]['completed_files'].append(tar_file.name)
            progress[split]['last_processed'] = tar_file.name
            progress[split]['total_items'] += items_processed

        # Ensure directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save progress
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
            
    def process_tar_dataset(self, split='train', max_samples=MAX_SAMPLES):
        """First stage: Extract images and captions from tar files"""
        processed_samples = []
        total_items = 0

        tar_files = sorted(WDS_DIR.glob(f"{split}*.tar"))
        if not tar_files:
            logger.error(f"No {split} tar files found in {WDS_DIR}")
            return None
        
        # Load previous progress
        progress = self.load_progress()
        completed_files = set(progress[split]['completed_files'])
        
         # Create return structure to match training expectations
        dataset = []

        for tar_file in tar_files:
            if total_items >= max_samples:
                break
                
            # Skip completed files
            if tar_file.name in completed_files:
                logger.info(f"Skipping completed file: {tar_file.name}")
                continue
                
            logger.info(f"Processing {tar_file.name}")
            items_processed = self._process_single_tar(tar_file, max_samples - self.processed_count)
            
            if items_processed > 0:
                # Save progress after each file
                 # Transform items into training format
                for item in items_processed:
                    dataset.append({
                        'image': Image.open(item['path']),  # Load image from saved path
                        'eng': item['eng_caption'],
                        'id': item['id']
                    })
                total_items += len(items_processed)
                self.save_progress(split, tar_file, len(items_processed))
                
        logger.info(f"Processed {len(dataset)} items for {split}")
        return dataset
                
    def _process_single_tar(self, tar_file, remaining_samples):
        """Process single tar file extracting image/caption pairs"""
        items_processed = 0
        try:
            with tarfile.open(tar_file, 'r') as tar:
                # Get all JSON files (metadata)
                json_files = [f for f in tar.getmembers() if f.name.endswith('.json')]
                
                # Process each JSON and its corresponding image
                for json_file in tqdm(json_files[:remaining_samples]):
                    try:
                        # Extract metadata
                        meta_content = tar.extractfile(json_file).read().decode('utf-8')
                        metadata = json.loads(meta_content)
                        
                        # Get image filename
                        image_id = Path(json_file.name).stem
                        image_name = f"{image_id}.jpg"
                        
                        # Create permanent storage path
                        permanent_image_path = self.images_dir / image_name

                        # Skip if already processed
                        if permanent_image_path.exists():
                            logger.info(f"Image {image_name} already exists, skipping...")
                            continue

                        try:
                            # Extract and process image
                            image_member = tar.getmember(image_name)
                            image_data = tar.extractfile(image_member).read()
                            image = Image.open(BytesIO(image_data)).convert('RGB')
                            
                            # Basic validation
                            if min(image.size) < 10 or image.getbbox() is None:
                                continue
                                
                            # Resize and save permanently
                            image = image.resize((self.image_size, self.image_size))
                            image.save(permanent_image_path)
                            
                            # Get caption
                            caption = metadata.get('eng_caption', '')
                            if not caption and 'captions' in metadata:
                                for cap in metadata['captions']:
                                    if isinstance(cap, dict) and cap.get('language') == 'eng':
                                        caption = cap.get('text', '')
                                        break
                            
                            if not caption:
                                continue
                                
                            # Store in database with permanent path
                            self.db.add_item(
                                path=str(permanent_image_path),
                                eng_caption=caption,
                                modality='image'
                            )
                            
                            self.processed_count += 1
                            items_processed += 1
                            
                            if self.processed_count % 100 == 0:
                                print(f"Processed {self.processed_count} items")
                                
                        except KeyError:
                            continue  # Skip if image file not found
                            
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON
                        
        except Exception as e:
            print(f"Error processing {tar_file}: {e}")
            return items_processed
        
    def verify_processed_images(self):
        """Verify integrity of processed images and database entries"""
        verification_results = {
            'total_db_entries': 0,
            'existing_images': 0,
            'missing_images': [],
            'invalid_paths': [],
            'mismatched_sizes': []
        }
        
        # Check database entries
        items = self.db.get_items_by_modality('image')
        verification_results['total_db_entries'] = len(items)
        
        for item_id, path, caption in items:
            image_path = Path(path)
            
            # Check if path is valid
            if not path or not isinstance(path, (str, Path)):
                verification_results['invalid_paths'].append(item_id)
                continue
                
            # Check if file exists
            if not image_path.exists():
                verification_results['missing_images'].append((item_id, str(image_path)))
                continue
                
            try:
                # Check image integrity and size
                with Image.open(image_path) as img:
                    if img.size != (self.image_size, self.image_size):
                        verification_results['mismatched_sizes'].append(
                            (item_id, str(image_path), img.size)
                        )
                    verification_results['existing_images'] += 1
            except Exception as e:
                verification_results['invalid_paths'].append((item_id, str(image_path), str(e)))
                
        # Log verification results
        logger.info("=== Image Processing Verification Results ===")
        logger.info(f"Total database entries: {verification_results['total_db_entries']}")
        logger.info(f"Existing valid images: {verification_results['existing_images']}")
        
        if verification_results['missing_images']:
            logger.error(f"Found {len(verification_results['missing_images'])} missing images:")
            for item_id, path in verification_results['missing_images'][:10]:  # Show first 10
                logger.error(f"  Missing: ID {item_id} at {path}")
                
        if verification_results['invalid_paths']:
            logger.error(f"Found {len(verification_results['invalid_paths'])} invalid paths:")
            for item in verification_results['invalid_paths'][:10]:
                logger.error(f"  Invalid: {item}")
                
        if verification_results['mismatched_sizes']:
            logger.error(f"Found {len(verification_results['mismatched_sizes'])} size mismatches:")
            for item_id, path, size in verification_results['mismatched_sizes'][:10]:
                logger.error(f"  Mismatch: ID {item_id} at {path} has size {size}")
                
        # Return verification status
        is_valid = (
            len(verification_results['missing_images']) == 0 and
            len(verification_results['invalid_paths']) == 0 and
            len(verification_results['mismatched_sizes']) == 0 and
            verification_results['existing_images'] == verification_results['total_db_entries']
        )
        
        if is_valid:
            logger.info("Verification PASSED: All images valid and accounted for")
        else:
            logger.error("Verification FAILED: Found inconsistencies")
            
        return is_valid, verification_results

    def verify_and_clean(self, remove_invalid=False):
        """Verify and optionally clean up invalid entries"""
        is_valid, results = self.verify_processed_images()
        
        if not is_valid and remove_invalid:
            logger.info("Cleaning up invalid entries...")
            
            # Remove invalid database entries
            for item_id, _ in results['missing_images']:
                self.db.remove_item(item_id)  # You'll need to add this method to ImageDatabase
                
            # Remove mismatched images
            for item_id, path, _ in results['mismatched_sizes']:
                self.db.remove_item(item_id)
                Path(path).unlink(missing_ok=True)
                
            # Verify again after cleanup
            is_valid, new_results = self.verify_processed_images()
            logger.info("Cleanup completed. Re-verification " + 
                       ("PASSED" if is_valid else "FAILED"))
            
        return is_valid
            
if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor()
    processor.process_tar_dataset('train', max_samples=1000)
    print(f"Total items processed: {processor.processed_count}")