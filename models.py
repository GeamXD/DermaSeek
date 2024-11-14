import time
from transformers import pipeline
from google.cloud import texttospeech
from google.oauth2 import service_account

# Create credentials
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)


def summarize_case(case: str) -> list[str, str]:
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
# Loads the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

if __name__ == '__main__':
    text_summary, time_taken = summarize_case(ARTICLE)
    
    print(
        f""""
        summary text:
                {text_summary}
        time taken: {time_taken} seconds
        """
    )
