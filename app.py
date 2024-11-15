import chromadb
import json
import base64
import io
import uuid
import time
import whisper
import gradio as gr
import models as md
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# Initialize models and client
client = chromadb.Client()
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # produces 384-dim embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # produces 512-dim embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = whisper.load_model("turbo")

# Create separate collections
text_collection = client.create_collection(
    name="dermatology_text_collection",
    metadata={"hnsw:space": "cosine"}
)

image_collection = client.create_collection(
    name="dermatology_image_collection",
    metadata={"hnsw:space": "cosine"}
)

def get_text_embedding(text):
    """Generate embedding for text content"""
    return text_model.encode(text).tolist()

def speech_to_text(audio_file_path: str) -> str:
    """
        Placeholder for speech-to-text conversion using Whisper
        Args:
            audio_file_path: Path to audio fie
        Returns:
            Transcribed text
    """
    #    Implement Whisper integration
    result = model.transcribe(audio_file_path)
    return result['text']

def calculate_accuracy(image_embedding, query_embedding):
    # Cosine similarity between query and image embeddings
    similarity = cosine_similarity([image_embedding],[query_embedding])[0][0]
    return similarity

def get_image_embedding(base64_string):
    """Generate embedding for image from base64 string"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    return image_features.detach().numpy().flatten().tolist()

def store_medical_case(case_data):
    """Store a single medical case with image and text"""
    base_id = str(uuid.uuid4())
    
    # Prepare text content
    text_content = f"Title: {case_data['metadata']['title']}\n\n"
    text_content += f"Description: {case_data['metadata']['description']}\n\n"
    text_content += f"Background: {case_data['metadata']['background']}"
    


    # Store text entry
    text_embedding = get_text_embedding(text_content)
    text_id = f"{base_id}_text"
    
    text_collection.add(
        embeddings=[text_embedding],
        documents=[text_content],
        metadatas=[{
            "type": "text",
            "title": case_data['metadata']['title'],
            "description": case_data['metadata']['description'],
            "background": case_data['metadata']['background'],
            "related_image_id": f"{base_id}_image",
        }],
        ids=[text_id]
    )
    
    # Store image entry
    image_embedding = get_image_embedding(case_data['img'])
    image_id = f"{base_id}_image"
    
    image_collection.add(
        embeddings=[image_embedding],
        documents=[case_data['img']],
        metadatas=[{
            "type": "image",
            "title": case_data['metadata']['title'],
            "description": case_data['metadata']['description'],
            "background": case_data['metadata']['background'],
            "related_text_id": text_id,
        }],
        ids=[image_id]
    )
    
    return text_id, image_id

def load_json_data(json_file_path):
    """
    Load and store medical cases from a JSON file into ChromaDB.
    
    Expected JSON format:
    [
        {
            "img": "base64_encoded_string",
            "metadata": {
                "title": "Case Title",
                "description": "Case Description",
                "background": "Patient Background"
            }
        },
        ...
    ]
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of medical cases")
            
        results = {
            'total_cases': len(data),
            'successful': [],
            'failed': [],
            'errors': []
        }
        # Measure image ingestion time
        start_ingestion_time = time.time()
    
        for index, item in enumerate(data):
            try:
                # Validate required fields
                if not all(key in item for key in ['img', 'metadata']):
                    raise ValueError(f"Case {index}: Missing required fields (img or metadata)")
                    
                if not all(key in item['metadata'] for key in ['title', 'description', 'background']):
                    raise ValueError(f"Case {index}: Missing required metadata fields")
                
                # Store the case
                text_id, image_id = store_medical_case(item)
                
                results['successful'].append({
                    'index': index,
                    'title': item['metadata']['title'],
                    'text_id': text_id,
                    'image_id': image_id
                })
        
            except Exception as e:
                results['failed'].append({
                    'index': index,
                    'title': item.get('metadata', {}).get('title', 'Unknown'),
                    'error': str(e)
                })
                results['errors'].append(str(e))
        # Measure total ingestion time
        end_ingestion_time = time.time()
        ingestion_time = end_ingestion_time - start_ingestion_time   
        
        # Add summary statistics
        results['success_count'] = len(results['successful'])
        results['failure_count'] = len(results['failed'])
        
        return results, ingestion_time
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading JSON file: {str(e)}")

def search_medical_cases(query, n_results=2, include_related=True, mode="text"):
    """
    Search function that can handle both text and image queries
    
    Args:
        query: Either a text string or base64 image string
        n_results: Number of results to return
        include_related: Whether to include related content
        mode: Either "text" or "image" to specify search type
    """
    if mode == "text":
        query_embedding = get_text_embedding(query)
        primary_collection = text_collection
        secondary_collection = image_collection
        related_id_field = 'related_image_id'
    elif mode == "image":
        query_embedding = get_image_embedding(query)
        primary_collection = image_collection
        secondary_collection = text_collection
        related_id_field = 'related_text_id'
    elif mode == 'speech':
        audio_text = speech_to_text(query)
        query_embedding = get_text_embedding(audio_text)
        primary_collection = text_collection
        secondary_collection = image_collection
        related_id_field = 'related_image_id'
    else:
        raise ValueError("Mode must be either 'text', 'Speech', 'image'")
        
    # Initial search
    results = primary_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if not include_related:
        return results
    
    # Get related content
    enhanced_results = []
    for idx, result_id in enumerate(results['ids'][0]):
        result_data = {
            'id': result_id,
            'document': results['documents'][0][idx],
            'metadata': results['metadatas'][0][idx],
            'distance': results['distances'][0][idx],
            'search_mode': mode
        }
        
        # Get related content ID
        related_id = result_data['metadata'].get(related_id_field)
        
        if related_id:
            related_content = secondary_collection.get(
                ids=[related_id]
            )
            result_data['related_content'] = {
                'document': related_content['documents'][0],
                'metadata': related_content['metadatas'][0]
            }
        
        enhanced_results.append(result_data)
    
    return enhanced_results

# Suggested queries
queries = [
    "Red, itchy skin",
    "Blistering rash",
    "Pustules on skin",
    "Scaling skin",
    "Skin inflammation",
    "Red patches",
    "Swollen skin",
    "Painful bumps",
    "Skin discoloration",
    "Open sores",
    # "Peeling skin",
    # "Itchy blisters",
    # "Cracked skin",
    # "Skin crusting",
    # "Fungal infection signs"
]

# Function to populate the query input box with the suggested query
def populate_query(suggested_query):
    return suggested_query

def create_interface():
    # Automatically load data at startup
    json_path = "data/cases_data.json"
    try:
        results, ingestion_time = load_json_data(json_path)
        ingestion_status = (
            f"**📊 Data Loaded**: {results['success_count']} cases in {ingestion_time:.3f}s\n"
            f"Successfully loaded: {results['success_count']} cases\n"
            f"Failed to load: {results['failure_count']} cases"
        )
    except Exception as e:
        ingestion_status = f"**📊 Data Loading Error**: {str(e)}"
    
    with gr.Blocks(theme=gr.themes.Soft(), css="""
        .center-text {
            text-align: center;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    """) as interface:
        # Header with title
        gr.Markdown("# DermaSeek", elem_classes='center-text')
        
        # Get the total number of items in collections
        total_text_items = len(text_collection.get()['ids'])
        total_image_items = len(image_collection.get()['ids'])
        
        # Performance Metrics Section
        with gr.Row():
            with gr.Column():
                gr.Markdown(ingestion_status)
            with gr.Column():
                query_time_display = gr.Markdown("**⚡ Query Time**: -- s")
            with gr.Column():
                gr.Markdown(f"**📑 Total Cases**: {total_text_items:,}")

        # Function to process and display results
        def format_results(results, query_time):
            if not results:
                return (
                    [],
                    "No results found.",
                    "**Match Score**: N/A",
                    f"**⚡ Query Time**: {query_time:.3f}s"
                )

            # Prepare gallery images
            gallery_images = []
            text_output = "### Case Summary\n\n"
            
            for idx, result in enumerate(results, 1):
                # Handle image display
                if result['search_mode'] == 'text':
                    if 'related_content' in result:
                        image_data = result['related_content']['document']
                        gallery_images.append(Image.open(io.BytesIO(base64.b64decode(image_data))))
                elif result['search_mode'] == 'image':
                    if 'related_content' in result:
                        image_data = result['document']
                        gallery_images.append(Image.open(io.BytesIO(base64.b64decode(image_data))))
                elif result['search_mode'] == 'speech':
                    if 'related_content' in result:
                        image_data = result['related_content']['document']
                        gallery_images.append(Image.open(io.BytesIO(base64.b64decode(image_data))))
                else:
                    return "Invalid Mode"

                # Format text information
                text_output += f"### CASE {idx}\n\n"
                text_output += f"**Title**: {result['metadata']['title']}\n\n"
                text_output += f"**Description**: {result['metadata']['description']}\n\n"
                text_output += f"**Background**: {result['metadata']['background']}\n\n"

                # Summarize Case
                summary_text, _ = md.summarize_case_ai(text_output)
                # summary_text, _ = md.summarize_case_hug(text_output) #Needs further processing

                # summary_text += f"\n\n**Match Score**: {1 - result['distance']:.2%}\n\n"

            return (
                gallery_images,
                summary_text,
                f"**TOP Match Score [CASE 1]**: {1 - results[0]['distance']:.2%}",
                f"**⚡ Query Time**: {query_time:.3f}s"
            )

        # Search handlers
        def text_search(query, n_results, include_related):
            if not query or query.strip() == "":
                return [], "No search query provided.", "**Match Score**: N/A", "**⚡ Query Time**: 0s"

            start_time = time.time()
            results = search_medical_cases(
                query=query,
                n_results=n_results,
                include_related=include_related,
                mode="text"
            )
            query_time = time.time() - start_time
            return format_results(results, query_time)

        def image_search(image_path, n_results, include_related):
            if not image_path:
                return [], "No image provided.", "**Match Score**: N/A", "**⚡ Query Time**: 0s"
            
            # Convert image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            start_time = time.time()
            results = search_medical_cases(
                query=image_data,
                n_results=n_results,
                include_related=include_related,
                mode="image"
            )
            query_time = time.time() - start_time
            return format_results(results, query_time)

        def speech_search(audio_path, n_results, include_related):
            if not audio_path:
                return [], "No audio provided.", "**Match Score**: N/A", "**⚡ Query Time**: 0s"
            
            start_time = time.time()
            results = search_medical_cases(
                query=audio_path,
                n_results=n_results,
                include_related=include_related,
                mode="speech"
            )
            query_time = time.time() - start_time
            return format_results(results, query_time)
        
        # Main content area
        with gr.Row():
            # Left Column - Search Inputs and Suggestions
            with gr.Column(scale=1):
                # Suggested Queries Section
                gr.Markdown("### 💡 Suggested Queries")
                with gr.Row():
                    query_suggestions = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[[q] for q in queries],
                        type="index"
                    )
                    
                # Search Type Tabs
                with gr.Tabs() as input_tabs:
                    # Text Search Tab
                    with gr.Tab("📝 Text Search"):
                        text_input = gr.Textbox(
                            label="Search medical cases",
                            placeholder="Describe the condition or symptoms..."
                        )
                        text_search_btn = gr.Button("Search", variant="primary")
                    
                    # Image Search Tab
                    with gr.Tab("🖼️ Image Search"):
                        image_input = gr.Image(
                            label="Upload a medical image",
                            type="filepath"
                        )
                        image_search_btn = gr.Button("Search", variant="primary")
                    
                    # Audio Search Tab
                    with gr.Tab("🎵 Speech Search"):
                        audio_input = gr.Audio(
                            label="Record or upload audio description",
                            type="filepath"
                        )
                        audio_search_btn = gr.Button("Search", variant="primary")
                
                # Controls
                with gr.Row():
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Number of results"
                    )
                    include_related = gr.Checkbox(
                        label="Include related content",
                        value=True
                    )

            # Right Column - Results
            with gr.Column(scale=1):
                gr.Markdown("### Search Results 🔍")
                with gr.Row():
                    # Results Gallery
                    results_gallery = gr.Gallery(
                        label="Image Results",
                        show_label=True,
                        columns=2,
                        height=400
                    )
                
                # Text Results
                results_text = gr.Markdown("", label="Case Details")
                
                # Performance Score
                accuracy_score = gr.Markdown("**Match Score**: --")

        # Function to handle suggested query selection
        def use_suggested_query(evt: gr.SelectData):
            selected_query = queries[evt.index]
            return selected_query

        # Connect suggested queries to text input
        query_suggestions.select(
            fn=use_suggested_query,
            outputs=[text_input]
        )

        # Connect the search buttons
        text_search_btn.click(
            fn=text_search,
            inputs=[text_input, num_results, include_related],
            outputs=[results_gallery, results_text, accuracy_score, query_time_display]
        )
        
        image_search_btn.click(
            fn=image_search,
            inputs=[image_input, num_results, include_related],
            outputs=[results_gallery, results_text, accuracy_score, query_time_display]
        )
        
        audio_search_btn.click(
            fn=speech_search,
            inputs=[audio_input, num_results, include_related],
            outputs=[results_gallery, results_text, accuracy_score, query_time_display]
        )

    return interface

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
