# Dermatology Medical Case Search Using ChromaDB

## Overview
This project implements an advanced image and data retrieval system for dermatology medical cases using ChromaDB. The primary goal is to enable users to efficiently search for dermatological cases by using text, image, or audio queries. This project uses data collected from the **DermNetNZ** case collection, providing a real-world solution for dermatological reference and research.

## Features
- **Case Retrieval**: Search by text, image, or audio descriptions.
- **Image and Text Embeddings**: Uses CLIP for image embeddings and Sentence Transformers for text embeddings.
- **Audio Transcription**: Converts spoken descriptions to text for case search.
- **Gradio Interface**: A user-friendly web interface for interactive queries.
- **ChromaDB Integration**: Efficient storage and retrieval with ChromaDB collections.

---

## Technologies and Libraries

- **Data Processing**: `requests`, `BeautifulSoup`, `json`, `base64`
- **Image and Text Embeddings**: `transformers`, `sentence_transformers`, `sklearn`
- **Audio Processing**: `whisper` model
- **Database**: `chromadb`
- **User Interface**: `gradio`
- **Utilities**: `uuid`, `PIL`, `io`, `time`

---

## Directory Structure
The project consists of the following primary files:

```plaintext
.
├── data
│   └── cases_data.json      # JSON file containing dermatology cases in a structured format
├── img
│   └── [case_title].jpg     # Folder containing downloaded case images
├── data_collection.py       # Script to collect data from DermNetNZ
├── app.py                   # Main application and Gradio interface
├── packages.txt             # Text containing required os package
├── requirements.txt         # Text containing required python libraries
└── README.md                # Project documentation
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Install required libraries:
```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GeamXD/Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB.git
   cd Advanced-Image-and-Data-Retrieval-System-Using-ChromaDB
   ```

2. Install dependencies:
   ```bash
   pip install requests beautifulsoup4 sentence-transformers transformers chromadb gradio
   ```

---

## Data Collection

### Running the Data Collection Script
The `data_collection.py` script scrapes dermatology cases from DermNetNZ, saves case images in base64 format, and stores each case with its title, description, and background in `data/cases_data.json`.

Run the script:
```bash
python data_collection.py
```

---

## Application Structure

### Main Application
The `app.py` file builds and runs the primary application, leveraging ChromaDB for fast retrieval of text and image data, alongside models for embeddings and transcription.

**Key Functions**:
- `get_text_embedding(text)`: Generates embeddings for text.
- `get_image_embedding(base64_string)`: Generates embeddings for images.
- `speech_to_text(audio_file_path)`: Transcribes audio descriptions to text.
- `store_medical_case(case_data)`: Stores a single medical case with image and text into ChromaDB.
- `search_medical_cases(query, n_results=2, include_related=True, mode="text")`: Searches medical cases in ChromaDB based on text, image, or audio queries.

---

## Running the Application

1. Start the Gradio interface:
   ```bash
   python app.py
   ```
2. **User Interface**: Access the application in your browser at `http://127.0.0.1:7860` (default Gradio URL).

### Gradio UI
The interface allows users to:
- Search for dermatological cases by text, image, or audio.
- View results with match scores and related case information.
- Check performance metrics like total cases, successful loads, and query times.

---

## Example Queries
Use the following queries in the interface to test various search functionalities:
- **Text Search**: "Red, itchy skin" or "Blistering rash"
- **Image Search**: Upload an image file of a dermatological case.
- **Audio Search**: Describe symptoms in audio format.

---
## Hosted On GradioSpace
[Check out the Demo](https://huggingface.co/spaces/geamxd/Dermatology-Case)

---

## Video Demo
[Watch the Demo on YouTube](??)


---


## Future Work
Potential improvements for this project include:
- Expanding the data collection to cover additional dermatology sources.
- Enhancing performance by optimizing embedding and search processes.
- Developing a more granular tagging system for detailed case classification.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contributors
- **Your Name** - Musa Godwin
- **Email** - musa.godwin8112@gmail.com