import chromadb
from chromadb.utils import embedding_functions
import base64
from PIL import Image
import io
import os
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import torch

class MultimediaVectorStore:
    def __init__(self, collection_name="multimedia_collection"):
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # Initialize embedding models
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create or get collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string for storage"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def decode_base64_to_image(self, base64_string):
        """Convert base64 string back to image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def get_text_embedding(self, text):
        """Generate embedding for text using Sentence Transformer"""
        return self.text_model.encode(text).tolist()

    def get_image_embedding(self, image_path):
        """Generate embedding for image using CLIP"""
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy().flatten().tolist()

    def add_text(self, text_content, metadata=None, id=None):
        """Add text content to the collection"""
        if id is None:
            id = str(hash(text_content))
        
        embedding = self.get_text_embedding(text_content)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text_content],
            metadatas=[{"type": "text", **(metadata or {})}],
            ids=[id]
        )
        return id

    def add_image(self, image_path, metadata=None, id=None):
        """Add image content to the collection"""
        if id is None:
            id = f"img_{os.path.basename(image_path)}"
            
        # Get image embedding
        embedding = self.get_image_embedding(image_path)
        
        # Convert image to base64 for storage
        image_base64 = self.encode_image_to_base64(image_path)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[image_base64],  # Store base64 string
            metadatas=[{"type": "image", "original_path": image_path, **(metadata or {})}],
            ids=[id]
        )
        return id

    def search(self, query, n_results=5, filter_type=None):
        """
        Search across all media types
        query can be either text or image_path
        """
        # Determine query type and get appropriate embedding
        if os.path.isfile(query) and query.lower().endswith(('.png', '.jpg', '.jpeg')):
            embedding = self.get_image_embedding(query)
        else:
            embedding = self.get_text_embedding(query)
        
        # Apply filter if specified
        where = {"type": filter_type} if filter_type else None
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where
        )
        
        return results

# Example usage
def main():
    # Initialize store
    store = MultimediaVectorStore()
    
    # Add some text content
    text_id = store.add_text(
        "A beautiful sunset over the ocean",
        metadata={"source": "description.txt"}
    )
    
    # Add an image
    image_id = store.add_image(
        "path/to/sunset.jpg",
        metadata={"photographer": "John Doe"}
    )
    
    # Search using text query
    text_results = store.search(
        "sunset by the sea",
        n_results=3
    )
    
    # Search using image query
    image_results = store.search(
        "path/to/query_image.jpg",
        n_results=3
    )
    
    # Search with type filter
    image_only_results = store.search(
        "sunset",
        n_results=3,
        filter_type="image"
    )

if __name__ == "__main__":
    main()
