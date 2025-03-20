from config import VECTOR_DIM, N_TREES
from pathlib import Path
import wandb
from db import ImageDatabase, AnnoyDatabase
import sqlite3
from annoy import AnnoyIndex
import numpy as np

class SearchHandler:
    def __init__(self, annoy_db, image_db):
        self.annoy_db = annoy_db
        self.image_db = image_db
     
    def add_image_embedding(self, id, embedding):
        """Add image embedding vector to index"""
        return self.annoy_db.add_embedding(id, embedding, "image")
    
    def add_text_embedding(self, id, embedding):
        """Add text embedding vector to index"""
        return self.annoy_db.add_embedding(id, embedding, "text")
    
    def build_index(self, n_trees=100):
        """Build the index for searching"""
        self.annoy_db.build_index(n_trees)
    
    def search_similar_items(self, query_embedding, n_results=5, embedding_type=None):
        """Find n most similar items to query embedding"""
        results = self.annoy_db.get_similar_items(query_embedding, n_results)
        
        if embedding_type:
            # Filter by embedding type if specified
            results = [r for r in results if r['embedding_type'] == embedding_type]  
        return results
    
    def search_similar_images(self, query_embedding, n_results=5):
        """Find n most similar images to query embedding"""
        return self.search_similar_items(query_embedding, n_results, "image")
    
    def retrieve_similar_images(self, query_embedding, search_type='text', n_results=5):
        """
        Retrieve top similar images based on embedding similarity.
        """
        similar_items = self.search_similar_items(
            query_embedding, 
            n_results,
            "image" if search_type == 'image' else None
        )
        
        results = []
        for item in similar_items:
            item_metadata = self.image_db.get_item(item['item_id'])
            if item_metadata:
                path, caption, _ = item_metadata
                results.append((
                    item['item_id'],
                    1 - item['distance'],  # Convert distance to similarity
                    path
                )) 
        return results

    @staticmethod
    def get_all_ids_from_annoy(INDEX_PATH, VECTOR_DIM):
        """Fetch all IDs stored in Annoy index."""
        annoy_index = AnnoyIndex(VECTOR_DIM, 'angular')
        annoy_index.load(INDEX_PATH)
        return [i for i in range(annoy_index.get_n_items())]

    @staticmethod
    def get_all_ids_from_db(db_path):
        """Fetch all IDs stored in SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT id FROM items")  # Updated table name
        db_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return db_ids

    def validate_consistency(self, INDEX_PATH, db_path, VECTOR_DIM):
        """Validate consistency between Annoy index and SQLite database"""
        annoy_ids = self.get_all_ids_from_annoy(INDEX_PATH, VECTOR_DIM)
        db_ids = self.get_all_ids_from_db(db_path)

        missing_in_annoy = set(db_ids) - set(annoy_ids)
        missing_in_db = set(annoy_ids) - set(db_ids)
        return missing_in_annoy, missing_in_db