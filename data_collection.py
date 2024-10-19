import requests
from bs4 import BeautifulSoup
import base64
import json
import os

# Base URLs
BASE_URL = "https://dermnetnz.org"
CASES_URL = f"{BASE_URL}/cases"

def fetch_html(url):
    """Fetch HTML content from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f'Error fetching {url}: Status {response.status_code}')
        return None

def get_case_urls(soup):
    """Extract case URLs from the main cases page."""
    hrefs = [a['href'] for a in soup.select('body > section.cases > div > div.cases__wrap__grid.js-cases-grid a')]
    return [BASE_URL + href for href in hrefs]

def get_case_data(cases_url):
    """Fetch and parse case data from a case URL."""
    case_content = fetch_html(cases_url)
    if not case_content:
        return None
    
    cases_soup = BeautifulSoup(case_content, 'html.parser')

    # Extract case title and background
    case_title = cases_url.split('/')[-1]
    paragraphs = cases_soup.select('div.text-block__wrap p')
    case_background = paragraphs[-2].get_text() if len(paragraphs) >= 2 else "No background available"

    # Extract case description
    desc_items = cases_soup.select('div.accordion__wrap__item')
    case_description = " ".join(" ".join(item.stripped_strings) for item in desc_items)

    # Extract image URL
    img_tag = cases_soup.find('img', {'class': 'leftAlone'})
    img_url = BASE_URL + img_tag['src'] if img_tag else None

    return case_title, case_background, case_description, img_url

def download_image_and_convert_to_base64(img_url, case_title):
    """Download an image from the image URL."""
    if img_url:
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            with open(f"img/{case_title}.jpg", "wb") as file:
                file.write(img_response.content)
            print(f"Image for {case_title} downloaded successfully!")
            return base64.b64encode(img_response.content).decode('utf-8')
        else:
            print(f"Failed to download image from {img_url}: Status {img_response.status_code}")
    else:
        print(f"No image URL found for {case_title}")

def main():
    # Create a list to hold all case data
    all_cases_data = []

    # Fetch main cases page
    main_content = fetch_html(CASES_URL)
    if not main_content:
        return

    # Parse the main page and get case URLs
    soup = BeautifulSoup(main_content, 'html.parser')
    cases_urls = get_case_urls(soup)

    # Loop through cases_urls and process each case
    for case_url in cases_urls:  # Adjust this slice as needed
        case_data = get_case_data(case_url)
        if case_data:
            case_title, case_background, case_description, img_url = case_data
            # print(f"Title: {case_title}\nBackground: {case_background}\nDescription: {case_description}\n")
            print(f"Processing: {case_title}")
            # download_image(img_url, case_title)

            # Download the image and convert to Base64
            img_base64 = download_image_and_convert_to_base64(img_url, case_title)

            # Compile data into JSON format
            case_json = {
                "img": img_base64,
                "metadata": {
                    "title": case_title,
                    "description": case_description,
                    "background": case_background
                }
            }

            all_cases_data.append(case_json)

    # Write compiled data to a JSON file
    with open('data/cases_data.json', 'w') as json_file:
        json.dump(all_cases_data, json_file, indent=4)
        print("Data compiled and saved to cases_data.json.")

if __name__ == "__main__":
    main()
