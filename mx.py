input_string = ('Here is a summary of the two case summaries:\n\n'
                'Case 1: Unwelcome Holiday Souvenir\n\n'
                'A 15-year-old boy developed an itchy, red rash after getting a beach "henna tattoo" in Thailand. '
                'The likely cause is an allergic reaction to paraphenylenediamine (PPD), a chemical in the tattoo dye. '
                'The reaction will settle with a topical steroid, but the boy is now sensitized to PPD and may react '
                'to other products containing the chemical, such as hair dye, sunscreen, or medications.\n\n'
                'Case 2: Streaky Summer Rash\n\n'
                'A 33-year-old man developed a phytophotodermatitis rash after using a strimmer, which spattered phototoxic sap onto his skin. '
                'The rash resulted in redness and blistering and was caused by exposure to sunlight. With photoprotection, the rash will heal, '
                'but may leave post-inflammatory pigmentation for several months.')

# Parsing and formatting
def format_case_summary(input_string):
    # Split into sections by double newlines
    sections = input_string.split("\n\n")
    # output = f"**{sections[0]}**\n\n"  # Add the introduction
    output = ''
    for section in sections[1:]:
        if section.startswith("Case"):
            # Add case title
            output += f"### {section}\n\n"
        else:
            # Format the case details
            output += f"- {section.strip()}\n\n"

    return output

formatted_summary = format_case_summary(input_string)
print(formatted_summary)




import gradio as gr
from TTS.api import TTS
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

model_name = TTS.list_models()[0]
tts = TTS(model_name)

available_speakers = [
    "Daisy Studious", "Sofia Hellen", "Asya Anara",
    "Eugenio Mataracı", "Viktor Menelaos", "Damien Black"      
]

available_languages = ["US English", "Spanish (LatAm)"]

selected_speaker = available_speakers[0]
selected_language = available_languages[0]

# Create the output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
# Global variable to store the last generated audio path and text
last_generated_audio = None
last_generated_text = ""

def trim_text(text, max_length=30):
    """
    Trim the text to a maximum length and add ellipsis if it exceeds the limit.
    """
    return text[:max_length] + "..." if len(text) > max_length else text


# Main Speech Synthesis Function
def generate_speech_with_timestamps(text, speaker, language):
    global last_generated_audio
    output_path = "output/generated_speech.wav"
    start_time = time.time()

    # Generate the speech and save it to a WAV file
    tts.tts_to_file(
        text=text,
        speaker=speaker,
        language='en' if language == "US English" else 'es',
        file_path=output_path
    )

    # TODO#9 - Managing Duration and Tracking Variables
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    last_generated_audio = output_path
    last_generated_text = text

    # Calculate the length of the generated speech
    samplerate, data = wavfile.read(output_path)
    speech_length = len(data) / samplerate

    # TODO#11 - Return Audio Information
    return output_path, len(text.split()), speaker, language, round(speech_length, 2), duration

# Waveform Function
def generate_waveform():
    # Initialize Global Variables and Input Validation
    global last_generated_audio, last_generated_text

    # Check if a valid audio file exists
    if not last_generated_audio or not os.path.exists(last_generated_audio):
        return None, "No valid audio file found to generate waveform."

    # Read Audio File and Create Time Axis
    samplerate, data = wavfile.read(last_generated_audio)
    time_axis = np.linspace(0, len(data) / samplerate, num=len(data))

    # Plot the Waveform with Custom Styling
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1E1E1E')  # Dark background

    # Plot the Waveform with Custom Styling
    ax.plot(time_axis, data, color='cyan', alpha=0.8, linewidth=1.2)

    # Styling grid and axes for a modern look
    ax.set_facecolor('#2E2E2E')  # Set darker plot background
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Add grid lines
    ax.spines['bottom'].set_color('white')  # Set bottom spine color to white
    ax.spines['left'].set_color('white')  # Set left spine color to white
    ax.tick_params(axis='x', colors='white')  # Set x-axis tick color
    ax.tick_params(axis='y', colors='white')  # Set y-axis tick color
    ax.set_xlabel("Time (seconds)", color='white')  # Label x-axis
    ax.set_ylabel("Amplitude", color='white')  # Label y-axis

    # Add a Title to the Plot
    # Trim long text for display in title
    trimmed_text = trim_text(last_generated_text)
    ax.set_title(f"Waveform for text input: '{trimmed_text}'", color='white', fontsize=14)

    # Save the waveform image
    waveform_image_path = "output/waveform.png"
    plt.savefig(waveform_image_path, transparent=True)
    plt.close()

    return waveform_image_path, "Waveform generated successfully!"

# Button Click Event Handler
def generate_speech(text, speaker, language):
    if not text:
        return None, "Please enter some text to generate speech.", "", gr.update(interactive=False)

    audio_path, word_count, speaker_name, lang, speech_length, duration = generate_speech_with_timestamps(text, speaker, language)

    # Format the text box content
    data_info = f"Word Count: {word_count}\nVoice: {speaker_name}\nLocalization: {lang}\nLength of Speech: {speech_length} seconds\nGeneration Duration: {duration} seconds"

    return audio_path, data_info, "Speech generation successful!", gr.update(interactive=True)

# Gradio Interface Setup
def setup_interface():
    with gr.Blocks() as app:
        # TODO#12 - Adding Title and Description
        gr.Markdown("# 🗣️ Text-to-Speech GenAI with Coqui TTS")
        gr.Markdown("Convert text to speech using Coqui TTS with support for different languages and speakers.")

        with gr.Row():
            with gr.Column():
                #13 - Creating Text Input
                text_input = gr.Textbox(label="Enter Text", placeholder="Type your text here...", lines=3)


                with gr.Row():
                    #14 - Creating Voice and Localization Options
                    speaker_dropdown = gr.Dropdown(choices=available_speakers, value=selected_speaker, label="Select Voice")
                    language_radio = gr.Radio(choices=available_languages, value=selected_language, label="Select Localization")

            with gr.Column():
                # TODO#15 - Displaying Data Information and Status
                data_info_display = gr.Textbox(label="Data Info", interactive=False, lines=5)
                status_message = gr.Textbox(label="Status", interactive=False)


        with gr.Row():
            with gr.Column():
                # TODO#16 - Adding Audio Output and "Generate Speech" Button
                audio_output = gr.Audio(label="Generated Speech", interactive=False)
                generate_button = gr.Button("Generate Speech")

            with gr.Column():
                # TODO#17 - Adding Waveform Display and "Generate Waveform" Button
                waveform_output = gr.Image(label="Waveform")
                generate_waveform_button = gr.Button("Generate Waveform", interactive=False)

        generate_button.click(
            generate_speech, 
            inputs=[text_input, speaker_dropdown, language_radio], 
            outputs=[audio_output, data_info_display, status_message, generate_waveform_button]
        )

        generate_waveform_button.click(
            generate_waveform, 
            outputs=[waveform_output, status_message]
        )

    return app

# TODO#18 - Launching the App
if __name__ == "__main__":
    app = setup_interface()
    app.launch(share=True)





    # def text_to_wav(voice_name: str, text: str, output_filename: str = None) -> str:
#     """
#     Convert text to speech using Google Cloud Text-to-Speech API and save as WAV file.
    
#     Args:
#         voice_name (str): The voice name (e.g., 'en-GB-Standard-A')
#         text (str): The text to convert to speech
#         output_filename (str, optional): Custom output filename. If None, uses voice name
    
#     Returns:
#         str: Path to the generated audio file
    
#     Raises:
#         Exception: If credentials are not properly set up
#     """
#     # Initialize client
#     client_tts = texttospeech.TextToSpeechClient(credentials=credentials)

#     # Extract language code from voice name (e.g., 'en-GB' from 'en-GB-Standard-A')
#     language_code = "-".join(voice_name.split("-")[:2])

#     # Create the synthesis input
#     synthesis_input = texttospeech.SynthesisInput(text=text)

#     # Configure voice parameters
#     voice = texttospeech.VoiceSelectionParams(
#         language_code=language_code,
#         name=voice_name
#     )

#     # Configure audio output
#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.LINEAR16
#     )

#     # Generate the speech
#     response = client_tts.synthesize_speech(
#         input=synthesis_input,
#         voice=voice,
#         audio_config=audio_config
#     )

#     # Generate output filename if not provided
#     if output_filename is None:
#         output_filename = f"{voice_name}.wav"
#     elif not output_filename.endswith('.wav'):
#         output_filename += '.wav'

#     # Save the audio content
#     with open(output_filename, "wb") as out:
#         out.write(response.audio_content)
#         print(f'Generated speech saved to "{output_filename}"')

#     return output_filename
