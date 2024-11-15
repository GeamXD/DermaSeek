import os
import time
from transformers import pipeline
from together import Together
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.oauth2 import service_account


# Loads the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load environment variables
load_dotenv()

# Creates a Together client
client = Together()

# Load GCP service account info
service_account_info = {
    "type": os.getenv("GCP_TYPE"),
    "project_id": os.getenv("GCP_PROJECT_ID"),
    "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),  # Ensure newline characters are preserved
    "client_email": os.getenv("GCP_CLIENT_EMAIL"),
    "client_id": os.getenv("GCP_CLIENT_ID"),
    "auth_uri": os.getenv("GCP_AUTH_URI"),
    "token_uri": os.getenv("GCP_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("GCP_CLIENT_CERT_URL")
}

# Create credentials
credentials = service_account.Credentials.from_service_account_info(service_account_info)

def summarize_case_ai(case: str) -> list[str, str]:
    """
    Summarize a case

    Args:
        case: string containing case to be summarized
    Returns:
        A list containing the summarized case and the time taken to summarize
    """
    start = time.time()
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[
          {"role": "system", "content": "You are a summarization assistant. Your goal is to provide clear, concise summaries that capture the main ideas, key points, and relevant details of the text while avoiding extraneous information. Aim for a summary that is informative but brief, adjusting the level of detail based on the text's length and complexity. Use neutral language and maintain clarity."},
          {"role": "user", "content": f"{case}"},
        ],
        max_tokens=256,
        temperature=0,
    )

    end = time.time()
    total_time = end - start
    return response.choices[0].message.content, total_time


def summarize_case_hug(case: str) -> list[str, str]:
    """
    Summarize a case

    Args:
        case: string containing case to be summarized
    Returns:
        A list containing the summarized case and the time taken to summarize
    """
    start = time.time()
    summary = summarizer(case, max_length=250, min_length=30, do_sample=False)
    end = time.time()
    total_time = end - start
    return summary, total_time


def text_to_wav(voice_name: str, text: str, output_filename: str = None) -> str:
    """
    Convert text to speech using Google Cloud Text-to-Speech API and save as WAV file.
    
    Args:
        voice_name (str): The voice name (e.g., 'en-GB-Standard-A')
        text (str): The text to convert to speech
        output_filename (str, optional): Custom output filename. If None, uses voice name
    
    Returns:
        str: Path to the generated audio file
    
    Raises:
        Exception: If credentials are not properly set up
    """
    # Initialize client
    client_tts = texttospeech.TextToSpeechClient(credentials=credentials)

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

    # Generate output filename if not provided
    if output_filename is None:
        output_filename = f"{voice_name}.wav"
    elif not output_filename.endswith('.wav'):
        output_filename += '.wav'

    # Save the audio content
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{output_filename}"')

    return output_filename

# Usage Example
if __name__ == '__main__':

    ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
        A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
        Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
        In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
        Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
        2010 marriage license application, according to court documents.
        Prosecutors said the marriages were part of an immigration scam.
        On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
        After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
        Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
        All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
        Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
        Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
        The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
        Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
        Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
        If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

    text_summary, time_taken = summarize_case_hug(ARTICLE)
 
    print(
        f""""
        summary text:
                {text_summary}
        time taken: {time_taken} seconds
        """
    )
