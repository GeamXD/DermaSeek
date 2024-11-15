import os
import time
from transformers import pipeline
from together import Together
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.oauth2 import service_account
from scipy.io import wavfile
import gradio as gr

available_languages = ["US English","English (India)", "English (UK)", "Arabic", "French (France)"]
select_language = available_languages[0]

language_dict = {
    "US English": 'en-US-Standard-C',
    "English (India)": 'en-IN-Standard-A',
    "English (UK)": 'en-GB-Standard-A',
    "Arabic": 'ar-XA-Standard-A',
    "French (France)": 'fr-FR-Standard-A'
}

# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
# Global variable to store the last generated audio path
last_generated_audio = None

def text_to_wav(available_language: str, text: str) -> str:
    """
    Convert text to speech using Google Cloud Text-to-Speech API and save as WAV file.
    
    Args:
        available_language (str): The language of choice
        text (str): The text to convert to speech
    
    Returns:
        str: Path to the generated audio file
    
    Raises:
        Exception: If credentials are not properly set up
    """
    # Initialize client
    client_tts = texttospeech.TextToSpeechClient(credentials=credentials)

    # Last generated audio
    global last_generated_audio
    output_path = "output/generated_speech.wav"
    last_generated_audio = output_path

    #
    voice_name = language_dict[available_language]

    # Extract language code from voice name (e.g., 'en-GB' from 'en-GB-Standard-A')
    language_code = "-".join(voice_name.split("-")[:2])

    # Create the synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )

    # Configure audio output
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Generate the speech
    response = client_tts.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )


    # Save the audio content
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{output_path}"')

    # Calculate the length of the generated speech
    samplerate, data = wavfile.read(output_path)
    speech_length = len(data) / samplerate
    return output_filename, speech_length