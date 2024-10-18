import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Union, Any
from datetime import datetime
import json
import logging
from pathlib import Path

# TODO 1: Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

class PerformanceMetrics:
    """
    TODO 2: Implement comprehensive metrics tracking
    - Track query times
    - Monitor accuracy scores
    - Calculate cache hit rates
    - Store user interaction patterns
    """
    def __init__(self):
        self.query_times = []
        self.accuracy_scores = []
        self.cache_hits = 0
        self.total_queries = 0
        self.query_history = defaultdict(list)
        
    def add_metric(self, query_time: float, accuracy: float, query_text: str):
        self.query_times.append(query_time)
        self.accuracy_scores.append(accuracy)
        self.total_queries += 1
        self.query_history[query_text].append({
            'timestamp': datetime.now(),
            'query_time': query_time,
            'accuracy': accuracy
        })
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.query_times:
            return {
                'status': 'No metrics available'
            }
        
        return {
            'avg_query_time': sum(self.query_times) / len(self.query_times),
            'avg_accuracy': sum(self.accuracy_scores) / len(self.accuracy_scores),
            'cache_hit_rate': self.cache_hits / self.total_queries if self.total_queries > 0 else 0,
            'total_queries': self.total_queries,
            'top_queries': Counter(self.query_history.keys()).most_common(5)
        }

class VectorSearchEngine:
    """
    TODO 3: Implement main search engine class
    - Initialize models and collections
    - Handle different types of queries
    - Manage caching and batch processing
    """
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("image_collection")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.metrics = PerformanceMetrics()
        self.cache = {}
        
    @lru_cache(maxsize=100)
    def get_cached_embedding(self, query_text: str) -> np.ndarray:
        """
        TODO 4: Implement caching mechanism
        - Cache frequent queries
        - Track cache hits/misses
        - Implement cache invalidation strategy
        """
        inputs = self.processor(text=query_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs).numpy()
        self.metrics.cache_hits += 1
        return embedding

    def process_input(self, input_data: Union[str, Image.Image], input_type: str = "text") -> np.ndarray:
        """
        TODO 5: Implement multi-modal input processing
        - Handle both text and image inputs
        - Preprocess inputs appropriately
        - Return normalized embeddings
        """
        if input_type == "text":
            return self.get_cached_embedding(input_data)
        else:
            image = Image.open(input_data) if isinstance(input_data, str) else input_data
            inputs = self.processor(images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                return self.model.get_image_features(**inputs).numpy()

    def batch_search(self, queries: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        TODO 6: Implement batch processing
        - Process multiple queries efficiently
        - Optimize batch size
        - Return aggregated results
        """
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_embeddings = [self.get_cached_embedding(q).tolist() for q in batch]
            batch_results = self.collection.query(
                query_embeddings=batch_embeddings,
                n_results=1
            )
            results.extend(batch_results)
        return results

    def filter_results(self, results: Dict[str, Any], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        TODO 7: Implement result filtering
        - Filter based on similarity threshold
        - Add confidence scores
        - Sort by relevance
        """
        filtered_results = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            similarity_score = 1 - distance  # Convert distance to similarity
            if similarity_score > similarity_threshold:
                filtered_results.append({
                    'metadata': metadata,
                    'similarity': similarity_score,
                    'confidence': f"{similarity_score * 100:.2f}%"
                })
        return sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)

def create_enhanced_interface(search_engine: VectorSearchEngine) -> gr.Blocks:
    """
    TODO 8: Create enhanced Gradio interface
    - Add multiple tabs for different functions
    - Include analytics dashboard
    - Add batch processing interface
    """
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Advanced Vector Search using ChromaDB")
        
        with gr.Tabs():
            # Search Tab
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column():
                        query_input = gr.Textbox(
                            placeholder="Enter your query here",
                            label="Search Query"
                        )
                        image_input = gr.Image(
                            label="Or upload an image",
                            type="filepath"
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Search")
                            clear_btn = gr.Button("Clear")
                    
                    with gr.Column():
                        result_image = gr.Image(label="Result")
                        metrics_output = gr.JSON(label="Search Metrics")
            
            # Analytics Tab
            with gr.Tab("Analytics"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Analytics")
                    analytics_output = gr.JSON(label="Performance Analytics")
            
            # Batch Processing Tab
            with gr.Tab("Batch Search"):
                batch_input = gr.Textbox(
                    placeholder="Enter multiple queries (one per line)",
                    label="Batch Queries",
                    lines=5
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=16,
                    step=1,
                    label="Batch Size"
                )
                batch_submit = gr.Button("Process Batch")
                batch_results = gr.JSON(label="Batch Results")

        # Define interface functions
        def search(query_text, image_path):
            """
            TODO 9: Implement search functionality
            - Handle both text and image queries
            - Track performance metrics
            - Return formatted results
            """
            start_time = time.time()
            
            if query_text:
                embedding = search_engine.process_input(query_text, "text")
            elif image_path:
                embedding = search_engine.process_input(image_path, "image")
            else:
                return None, {"error": "No input provided"}
            
            results = search_engine.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1
            )
            
            filtered_results = search_engine.filter_results(results)
            query_time = time.time() - start_time
            
            # Update metrics
            search_engine.metrics.add_metric(
                query_time=query_time,
                accuracy=filtered_results[0]['similarity'] if filtered_results else 0,
                query_text=query_text or "image_query"
            )
            
            return (
                Image.open(filtered_results[0]['metadata']['image']) if filtered_results else None,
                {
                    'query_time': f"{query_time:.4f}s",
                    'confidence': filtered_results[0]['confidence'] if filtered_results else "N/A",
                    'performance_metrics': search_engine.metrics.get_summary()
                }
            )

        def process_batch(queries, batch_size):
            """
            TODO 10: Implement batch processing
            - Process multiple queries efficiently
            - Track batch performance
            - Return aggregated results
            """
            queries = [q.strip() for q in queries.split('\n') if q.strip()]
            start_time = time.time()
            results = search_engine.batch_search(queries, int(batch_size))
            total_time = time.time() - start_time
            
            return {
                'total_time': f"{total_time:.4f}s",
                'average_time_per_query': f"{total_time/len(queries):.4f}s",
                'results': results
            }

        def refresh_analytics():
            """
            TODO 11: Implement analytics refresh
            - Generate updated metrics
            - Calculate performance trends
            - Return formatted analytics
            """
            return search_engine.metrics.get_summary()

        # Connect interface functions
        submit_btn.click(
            search,
            inputs=[query_input, image_input],
            outputs=[result_image, metrics_output]
        )
        
        clear_btn.click(
            lambda: (None, None),
            outputs=[result_image, metrics_output]
        )
        
        refresh_btn.click(
            refresh_analytics,
            outputs=analytics_output
        )
        
        batch_submit.click(
            process_batch,
            inputs=[batch_input, batch_size],
            outputs=batch_results
        )

        return interface

def main():
    """
    TODO 12: Implement main application logic
    - Initialize search engine
    - Load initial data
    - Start the interface
    """
    # Initialize search engine
    search_engine = VectorSearchEngine()
    
    # Load initial image data
    image_paths = [
        "img/image-01.jpg",
        "img/image-02.jpg",
        "img/image-03.jpeg",
        "img/image-05.jpeg",
        'img/images.jpeg'
    ]
    
    # Process and store initial images
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = search_engine.processor(images=images, return_tensors="pt", padding=True)
    
    start_time = time.time()
    with torch.no_grad():
        image_embeddings = search_engine.model.get_image_features(**inputs).numpy()
    
    # Add embeddings to collection
    search_engine.collection.add(
        embeddings=[embedding.tolist() for embedding in image_embeddings],
        metadatas=[{"image": path} for path in image_paths],
        ids=[str(i) for i in range(len(image_paths))]
    )
    
    logging.info(f"Data ingestion completed in {time.time() - start_time:.4f} seconds")
    
    # Create and launch interface
    interface = create_enhanced_interface(search_engine)
    interface.launch()

if __name__ == "__main__":
    main()
