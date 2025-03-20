from pathlib import Path
import numpy as np
from PIL import Image
import logging
import argparse
import matplotlib.pyplot as plt
from config import DEVICE, VECTOR_DIM, DB_PATH, INDEX_PATH
from model import CLIPModel
from db import ImageDatabase, AnnoyDatabase
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class Prompter:
    def __init__(self):
        logger.info("Let your search begin!")
        self.model = CLIPModel()
        self.image_db = ImageDatabase(DB_PATH)
        self.annoy_db = AnnoyDatabase(VECTOR_DIM, INDEX_PATH)
        logger.info("Image Search System initialized successfully")

    def search_images(self, prompt, top_k=5):
        """
        Search for images matching the input prompt
        Args:
            prompt (str): Text prompt to search for
            top_k (int
            ): Number of top matches to return
        Returns:
            list: List of tuples (image_path, similarity_score)
        """
        try:
            # 1. Process and encode the text prompt
            logger.info(f"Processing prompt: {prompt}")
            text_embedding = self.model.encode_text(prompt)
            
            # 2. Reshape embedding if needed
            if text_embedding.ndim == 1:
                text_embedding = text_embedding.reshape(1, -1)
            
            # 3. Get similar images from Annoy index
            similar_results = self.annoy_db.get_similar_items(text_embedding.flatten(), top_k)
            results = []
            
            # 4. Process results and get metadata
            for item in similar_results:
                item_metadata = self.image_db.get_item(item['item_id'])
                if item_metadata:
                    path, _, _ = item_metadata  # Using proper tuple unpacking
                    # 5. Convert distance to similarity score
                    results.append((path, 1 - item['distance']))
            
            # 6. Return processed results
            return results
            
        except Exception as e:
            # 7. Error handling
            logger.error(f"Error during image search: {str(e)}")
            return []
        
    def display_results(self, results):
        """
        Display search results in a user-friendly format
        Args:
            results (list): List of (image_path, similarity) tuples
        """
        if not results:
            print("No results found.")
            return
            
        print("\nSearch Results:")
        print("-" * 50)
        for i, (path, similarity) in enumerate(results, 1):
            print(f"{i}. Image: {path}")
            print(f"   Similarity Score: {similarity:.4f}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Image Search System')
    parser.add_argument('--top_k', type=int, default=5, choices=[5], help='Number of top matches to return')
    args = parser.parse_args()
    
    prompter = Prompter()
    
    while True:
        try:
            # User input
            prompt = input("\nEnter your search prompt (or 'quit' to exit): ")
            
            if prompt.lower() == 'quit':
                print("Exiting search system...")
                break
                
            if not prompt.strip():
                print("Please enter a valid prompt.")
                continue
            
            # Search Text2Image
            results = prompter.search_images(prompt, args.top_k)
            
            # Display results
            prompter.display_results(results)
            
        except KeyboardInterrupt:
            print("\nExiting search system...")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()