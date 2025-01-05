#DATA COLLECTION
#Collecting the data
#!pip install beautifulsoup4==4.12.3 - Installed
#!pip install requests==2.31.0 -Installed
import requests
from bs4 import BeautifulSoup
import re

# URLs of the Wikipedia articles
urls = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    #"https://en.wikipedia.org/wiki/Apollo_program",
    #"https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    #"https://en.wikipedia.org/wiki/Mars_rover",  # Corrected link
    #"https://en.wikipedia.org/wiki/International_Space_Station",
    #"https://en.wikipedia.org/wiki/SpaceX",
    #"https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    #"https://en.wikipedia.org/wiki/Voyager_program",
    #"https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    #"https://en.wikipedia.org/wiki/Kepler_Space_Telescope",
    #"https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    #"https://en.wikipedia.org/wiki/Space_Shuttle",
    #"https://en.wikipedia.org/wiki/Artemis_program",
    #"https://en.wikipedia.org/wiki/Skylab",
    #"https://en.wikipedia.org/wiki/NASA",
    #"https://en.wikipedia.org/wiki/European_Space_Agency",
    #"https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    #"https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    #"https://en.wikipedia.org/wiki/New_Horizons",
    #"https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens",
    #"https://en.wikipedia.org/wiki/Curiosity_(rover)",
    #"https://en.wikipedia.org/wiki/Perseverance_(rover)",
    #"https://en.wikipedia.org/wiki/InSight",
    #"https://en.wikipedia.org/wiki/OSIRIS-REx",
    #"https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    #"https://en.wikipedia.org/wiki/BepiColombo",
    #"https://en.wikipedia.org/wiki/Juice_(spacecraft)",
    #"https://en.wikipedia.org/wiki/Solar_Orbiter",
    #"https://en.wikipedia.org/wiki/CHEOPS_(satellite)",
    #"https://en.wikipedia.org/wiki/Gaia_(spacecraft)"
]

#Preparing the data

def clean_text(content):
    # Remove references that usually appear as [1], [2], etc.
    content = re.sub(r'\[\d+\]', '', content)
    # Remove newline characters for clean text
    content = content.replace('\n', ' ')
    return content

def fetch_and_clean(url):
    # Fetch the content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main content of the article, ignoring side boxes and headers
    content = soup.find('div', {'class': 'mw-parser-output'})

    if not content:
        print(f"Main content not found for URL: {url}")
        return ""

    # Remove the bibliography section which generally follows a header like "References", "Bibliography"
    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)
        if section:
            # Remove all content from this section to the end of the document
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()

    # Extract and clean the text
    text = content.get_text(separator=' ', strip=True)
    text = clean_text(text)
    return text

# File to write the clean text
with open('llm.txt', 'w', encoding='utf-8') as file:
    for url in urls:
        clean_article_text = fetch_and_clean(url)
        file.write(clean_article_text + '\n')

print("Content written to llm.txt")

# data stored in llm.txt file

# Open the file and read the first 20 lines
with open('llm.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # Print the first 20 lines
    for line in lines[:20]:
        print(line.strip())
