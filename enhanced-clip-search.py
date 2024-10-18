import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from collections import Counter
import numpy as np
import logging
from pathlib import Path

# TODO 1: Enhanced Logging Setup
# Add below code after imports:
'''
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
'''

# TODO 2: Add Performance Tracking Class
# Add this class before client initialization:
'''
class SearchMetrics:
    def __init__(self):
        self.query_times = []
        self.accuracy_scores = []
        self.total_queries = 0
        self.query_history = {}
    
    def add_query(self, query_text, query_time, accuracy):
        self.query_times.append(query_time)
        self.accuracy_scores.append(accuracy)
        self.total_queries += 1
        
        if query_text not in self.query_history:
            self.query_history[query_text] = []
        self.query_history[query_text].append({
            'time': query_time,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
'''

# Initialize client and models (keep existing code)
client = chromadb.Client()
collection = client.create_collection("image_collection")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# TODO 3: Add Caching Decorator
# Add this before image processing:
'''
@lru_cache(maxsize=100)
def get_cached_embedding(text_input):
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    with torch.no_grad():
        return model.get_text_features(**inputs).numpy()
'''

# TODO 4: Enhance Image Processing
# Modify the image processing section:
'''
def process_images(image_paths):
    start_time = time.time()
    results = []
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            inputs = processor(images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).numpy()
            results.append({
                'path': image_path,
                'embedding': embedding.tolist(),
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            results.append({
                'path': image_path,
                'status': 'error',
                'error': str(e)
            })
    
    processing_time = time.time() - start_time
    logger.info(f"Processed {len(results)} images in {processing_time:.2f} seconds")
    return results
'''

# TODO 5: Enhance Search Function
def search_image(query, metrics_tracker=None):
    """
    Add enhanced search with the following features:
    - Query validation
    - Error handling
    - Performance tracking
    - Confidence scoring
    - Result filtering
    """
    start_time = time.time()
    
    try:
        # Validate query
        if not query.strip():
            return None, "Please enter a search query", "", {}
        
        # Get embedding (use cached if text query)
        embedding = get_cached_embedding(query)
        
        # Perform search with multiple results
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=3  # Get top 3 results
        )
        
        # Process results
        processed_results = []
        for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            similarity = 1 - distance  # Convert distance to similarity
            processed_results.append({
                'image_path': metadata['image'],
                'similarity': similarity,
                'confidence': f"{similarity * 100:.2f}%"
            })
        
        # Sort by similarity
        processed_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get best match
        best_match = processed_results[0]
        result_image = Image.open(best_match['image_path'])
        
        # Calculate performance metrics
        query_time = time.time() - start_time
        if metrics_tracker:
            metrics_tracker.add_query(query, query_time, best_match['similarity'])
        
        # Prepare performance data
        performance_data = {
            'query_time': f"{query_time:.4f} seconds",
            'confidence': best_match['confidence'],
            'alternative_matches': processed_results[1:],
            'total_results_found': len(processed_results)
        }
        
        return result_image, performance_data
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return None, f"Error during search: {str(e)}", "", {}

# TODO 6: Enhanced Gradio Interface
def create_enhanced_interface():
    """
    Create an enhanced interface with:
    - Multiple tabs
    - Batch processing
    - Analytics dashboard
    - Error handling
    """
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        metrics = SearchMetrics()  # Initialize metrics tracker
        
        gr.Markdown("# Advanced Vector Search with CLIP and ChromaDB")
        
        with gr.Tabs():
            # Search Tab
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column():
                        query_input = gr.Textbox(
                            placeholder="Enter your search query",
                            label="Search Query"
                        )
                        submit_btn = gr.Button("Search")
                        clear_btn = gr.Button("Clear")
                    
                    with gr.Column():
                        result_image = gr.Image(label="Best Match")
                        performance_output = gr.JSON(label="Search Performance")
            
            # Analytics Tab
            with gr.Tab("Analytics"):
                refresh_btn = gr.Button("Refresh Analytics")
                analytics_output = gr.JSON(label="Search Analytics")
                
                def refresh_analytics():
                    return {
                        'total_queries': metrics.total_queries,
                        'avg_query_time': f"{np.mean(metrics.query_times):.4f}s",
                        'avg_accuracy': f"{np.mean(metrics.accuracy_scores):.2f}",
                        'top_queries': Counter(metrics.query_history.keys()).most_common(5)
                    }
            
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
                
                def process_batch(queries, batch_size):
                    queries = [q.strip() for q in queries.split('\n') if q.strip()]
                    results = []
                    start_time = time.time()
                    
                    for i in range(0, len(queries), batch_size):
                        batch = queries[i:i + batch_size]
                        for query in batch:
                            result, perf_data = search_image(query, metrics)
                            results.append({
                                'query': query,
                                'performance': perf_data
                            })
                    
                    total_time = time.time() - start_time
                    return {
                        'total_time': f"{total_time:.4f}s",
                        'avg_time_per_query': f"{total_time/len(queries):.4f}s",
                        'results': results
                    }

        # Connect interface functions
        submit_btn.click(
            fn=lambda q: search_image(q, metrics),
            inputs=query_input,
            outputs=[result_image, performance_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, None),
            outputs=[result_image, performance_output]
        )
        
        refresh_btn.click(
            fn=refresh_analytics,
            outputs=analytics_output
        )
        
        batch_submit.click(
            fn=process_batch,
            inputs=[batch_input, batch_size],
            outputs=batch_results
        )

    return interface

# TODO 7: Enhanced Main Function
def main():
    """
    Add enhanced main function with:
    - Error handling
    - Performance monitoring
    - Resource cleanup
    """
    try:
        logger.info("Initializing vector search application...")
        
        # Process initial images
        image_paths = [
            "img/image-01.jpg",
            "img/image-02.jpg",
            "img/image-03.jpeg",
            "img/image-04.jpeg",
            "img/images.jpeg"
        ]
        
        # Process and add images
        results = process_images(image_paths)
        successful_results = [r for r in results if r['status'] == 'success']
        
        # Add to collection
        if successful_results:
            collection.add(
                embeddings=[r['embedding'] for r in successful_results],
                metadatas=[{"image": r['path']} for r in successful_results],
                ids=[str(i) for i in range(len(successful_results))]
            )
            
        logger.info(f"Successfully initialized with {len(successful_results)} images")
        
        # Create and launch interface
        interface = create_enhanced_interface()
        interface.launch()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
