from config import DB_PATH, INDEX_PATH, VECTOR_DIM, N_TREES
from pathlib import Path
import sqlite3
from annoy import AnnoyIndex
import numpy as np
import uuid
import logging
import wandb

# Get logger from main
logger = logging.getLogger(__name__)

class ImageDatabase:
    def __init__(self, db_path):
        # Initialize SQLite
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.create_tables()

    def create_tables(self):
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS items (
                        id TEXT PRIMARY KEY,
                        path TEXT,
                        eng_caption TEXT,
                        modality TEXT NOT NULL
                    )
                """)
                
    def add_item(self, path, eng_caption, modality):
        """Add image or text metadata to the database"""
        try:
            item_id = str(uuid.uuid4())
            with self.conn:
                self.conn.execute(
                    "INSERT INTO items (id, path, eng_caption, modality) VALUES (?, ?, ?, ?)",
                    (item_id, path, eng_caption, modality)
                )
                return item_id
        except sqlite3.Error as e:
            logger.error(f"Error adding item to database: {str(e)}")
            raise

    def get_item(self, item_id):
        cursor = self.conn.execute(
            "SELECT path, eng_caption, modality FROM items WHERE id = ?",
            (item_id,)
        )
        return cursor.fetchone()

    def get_images_path(self, ids):
        """Get paths for multiple image IDs"""
        paths = {}
        for id_ in ids:
            cursor = self.conn.execute(
                "SELECT path FROM items WHERE id = ? AND modality = 'image'", 
                (id_,)
            )
            result = cursor.fetchone()
            paths[id_] = result[0] if result else None
        return paths

    def get_items_by_modality(self, modality):
            """Get all items of specific modality"""
            cursor = self.conn.execute(
                "SELECT id, path, eng_caption FROM items WHERE modality = ?",
                (modality,)
            )
            return cursor.fetchall()

    def __len__(self):
        cursor = self.conn.execute("SELECT COUNT(*) FROM items")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()

class AnnoyDatabase:
    def __init__(self, VECTOR_DIM, index_path):
        self.VECTOR_DIM = VECTOR_DIM
        self.index_path = Path(index_path)
        self.annoy_index = AnnoyIndex(VECTOR_DIM, 'angular') # Unified index for all embeddings
        self.index_built = False
        self.item_mapping = {} # Map item_id to index
   
        # Try to load existing indices
        try:
            self.load_index()
        except Exception as e:
            logger.warning(f"Creating new index: {e}")
            self.annoy_index = AnnoyIndex(VECTOR_DIM, 'angular')
        
    def get_index_info(self):
        """Get current state of the index"""
        return {
            'total_items': self.annoy_index.get_n_items(),
            'dimension': self.VECTOR_DIM,
            'built': self.index_built,
            'mapping_size': len(self.item_mapping)
        }
    
    def load_index(self):
        """Load existing Annoy index"""
        if not self.index_path.exists():
            raise FileNotFoundError("Index file not found")
        try:
            # Create a temporary index for reading
            temp_index = AnnoyIndex(self.VECTOR_DIM, 'angular')
            temp_index.load(str(self.index_path))
            
            # Transfer items to our writable index
            n_items = temp_index.get_n_items()
            for i in range(n_items):
                vector = temp_index.get_item_vector(i)
                self.annoy_index.add_item(i, vector)
            
            self.index_built = False  # Keep it false so we can still add items
            logger.info(f"Successfully loaded {n_items} items from existing index")
            
            # Clean up
            del temp_index
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def load_embeddings(self, embeddings_data):
        """Load multiple embeddings and rebuild index
        
        Args:
            embeddings_data (dict): Dictionary with modality keys ('image', 'text')
                                  containing lists of (item_id, embedding) tuples
        """
        if not isinstance(embeddings_data, dict):
            raise TypeError("Expected dictionary of embeddings")

        # Reset existing index state
        self.annoy_index = AnnoyIndex(self.VECTOR_DIM, 'angular')
        self.item_mapping = {}
        self.index_built = False
        
        try:
            # Process embeddings by modality
            for modality in ('image', 'text'):
                if modality in embeddings_data:
                    for item_id, embedding in embeddings_data[modality]:
                        self.add_embedding(item_id, embedding, modality)
            
            # Build index after adding all embeddings
            if self.annoy_index.get_n_items() > 0:
                self.build_index()

        except Exception as e:
            logger.error(f"Error loading embeddings batch: {e}")
            raise

    def add_embedding(self, item_id, embedding, embedding_type):
        """Add embedding with matching item_id from ImageDatabase"""
        if len(embedding) != self.VECTOR_DIM:
            raise ValueError(f"Expected {self.VECTOR_DIM} dimensions, got {len(embedding)}")
        
        try:
            # Add to Annoy index
            index = self.annoy_index.get_n_items()
            self.annoy_index.add_item(index, embedding)
            
            # Store mapping
            self.item_mapping[index] = {
                'item_id': item_id,
                'embedding_type': embedding_type
            }
            
            logger.info(f"Added {embedding_type} embedding for ID {item_id}")
            return index
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            raise
    
    def get_embedding(self, item_id):
        """Retrieve embedding by item_id"""
        for annoy_idx, info in self.item_mapping.items():
            if info['item_id'] == item_id:
                # Get the embedding vector from unified annoy index
                embedding = self.annoy_index.get_item_vector(annoy_idx)
                embedding_type = info['embedding_type']
                return embedding, embedding_type
        return None, None

    def build_index(self, n_trees=N_TREES):
        """Build and save Annoy index"""
        if not self.index_built:
            try:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.annoy_index.build(n_trees)
                self.annoy_index.save(str(self.index_path))
                self.index_built = True
                logger.info("Successfully built and saved index")
            except Exception as e:
                logger.error(f"Error building index: {e}")
                raise

    def get_similar_items(self, query_embedding, n_results=5):
        """Get similar items using Annoy index"""
        if not self.index_built:
            raise RuntimeError("Index not built yet")
            
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding, 
            n_results, 
            include_distances=True
        )
        
        results = []
        for idx, distance in zip(indices, distances):
            if idx in self.item_mapping:
                item_info = self.item_mapping[idx]
                results.append({
                    'item_id': item_info['item_id'],
                    'embedding_type': item_info['embedding_type'],
                    'distance': distance
                })
                
        return results
    
if __name__ == "__main__":
    try:
        logger.info("Testing unified ID and embedding...")
        image_db = ImageDatabase(DB_PATH)
        annoy_db = AnnoyDatabase(VECTOR_DIM, INDEX_PATH)

        # Add a sample image embedding
        test_embedding = np.random.rand(VECTOR_DIM).astype(np.float32)
        test_id = image_db.add_item("test.jpg", "Test image", "image")
        annoy_db.add_embedding(test_id, test_embedding, "image")

        # Build index
        annoy_db.build_index()

    
        # Test retrieval
        similar = annoy_db.get_similar_items(test_embedding)
        logger.info(f"Retrieved: {similar}")

        # Clean up
        image_db.close()
    except Exception as e:
        logger.error(f"Error: {e}")
