# Installing Deeplake if not available
try:
    import deeplake
except ImportError:
    print("Deeplake not found, installing...")
    import os
    os.system("pip install deeplake==3.9.18")
    import deeplake
    print("Deeplake installed successfully!")

# Retrieving and setting the OpenAI API key
api_key_path = "C:/Users/srija/OneDrive/Desktop/ragprojectt/api_key.txt"
with open(api_key_path, "r") as f:
    API_KEY = f.readline().strip()

# Setting the OpenAI API key
import os
import openai
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieving and setting the Activeloop API token
activeloop_path = "C:/Users/srija/OneDrive/Desktop/ragprojectt/activeloop.txt"
with open(activeloop_path, "r") as f:
    ACTIVELOOP_TOKEN = f.readline().strip()

os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

# Installing Sentence-Transformers if not available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not found, installing...")
    os.system("pip install sentence-transformers==3.0.1")
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers installed successfully!")

# Loading the Vector Store
vector_store_path = "hub://srijayalavarthi254/text_embedding"

try:
    vector_store = deeplake.VectorStore(path=vector_store_path, token=ACTIVELOOP_TOKEN)
    print("Vector store loaded successfully!")
except Exception as e:
    print(f"Failed to load vector store: {e}")
    vector_store = None

# Function to generate embeddings for the given text using OpenAI API
def embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]  # Clean up the texts by removing new lines
    response = openai.Embedding.create(input=texts, model=model)
    return [data['embedding'] for data in response['data']]  # Return the embeddings

# Function to perform the search query on the vector store
def search_query(prompt, vector_store):
    # Embedding the query prompt first
    prompt_embedding = embedding_function(prompt)

    # Performing the search using embeddings
    search_results = vector_store.search(embedding=prompt_embedding[0], k=5)  # `embedding` instead of `query_embedding`
    return search_results

# Example: Use the function with a pre-set prompt
user_prompt = "Tell me about space exploration on the Moon and Mars."

# Perform the search
if vector_store:
    try:
        search_results = search_query(user_prompt, vector_store)

        # Print the search results
        print("Search Results:")
        for result in search_results:
            print(result)  # Customize how you want to display the results
    except Exception as e:
        print(f"An error occurred during search: {e}")
else:
    print("Vector store is not initialized. Please check your setup.")

# Function to wrap text to a specified width
def wrap_text(text, width=80):
    lines = []
    while len(text) > width:
        split_index = text.rfind(' ', 0, width)
        if split_index == -1:
            split_index = width
        lines.append(text[:split_index])
        text = text[split_index:].strip()
    lines.append(text)
    return '\n'.join(lines)
     
     
import textwrap

# Assuming the search results are ordered with the top result first
top_score = search_results['score'][0]
top_text = search_results['text'][0].strip()
top_metadata = search_results['metadata'][0]['source']

# Print the top search result
print("Top Search Result:")
print(f"Score: {top_score}")
print(f"Source: {top_metadata}")
print("Text:")
print(wrap_text(top_text))

#Augmented Input

augmented_input=user_prompt+" "+top_text
print(augmented_input)

#Generation and output

import openai
#from openai import OpenAI
import time

 # Start timing before the request
gpt_model = "gpt-4"  # Correct model name
start_time=time.time() #start timing before the request
def call_gpt4_with_full_text(itext):
    # Join all lines to form a single string
    text_input = '\n'.join(itext)
    prompt = f"Please elaborate on the following content:\n{text_input}"

    try:
        # Make the API call using the updated OpenAI client
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a space exploration expert."},
                {"role": "assistant", "content": "1. You can explain, read the input and answer in detail."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Adjust temperature as needed
        )

        # Return the response message content
        return response['choices'][0]['message']['content'].strip()

    except openai.error.AuthenticationError:
        return "Invalid API key. Please check your API key."
    except Exception as e:
        return str(e)


gpt4_response = call_gpt4_with_full_text(augmented_input)

response_time = time.time() - start_time  # Measure response time
print(f"Response Time: {response_time:.2f} seconds")  # Print response time

print(gpt_model, "Response:", gpt4_response)

#Formatted response

import textwrap
import re
from IPython.display import display, Markdown, HTML
import markdown

def print_formatted_response(response):
    # Check for markdown by looking for patterns like headers, bold, lists, etc.
    markdown_patterns = [
        r"^#+\s",           # Headers
        r"^\*+",            # Bullet points
        r"\*\*",            # Bold
        r"_",               # Italics
        r" ",               # Links
        r"-\s",             # Dashes used for lists
        r"\`\`\`"           # Code blocks
    ]

    # If any pattern matches, assume the response is in markdown
    if any(re.search(pattern, response, re.MULTILINE) for pattern in markdown_patterns):
        # Markdown detected, convert to HTML for nicer display
        html_output = markdown.markdown(response)
        display(HTML(html_output))  # Use display(HTML()) to render HTML in Colab
    else:
        # No markdown detected, wrap and print as plain text
        wrapper = textwrap.TextWrapper(width=80)
        wrapped_text = wrapper.fill(text=response)

        print("Text Response:")
        print("--------------------")
        print(wrapped_text)
        print("--------------------\n")

print_formatted_response(gpt4_response)
     

#Evaluating the output with Cosine Similarity

#with initial user prompt
   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

similarity_score = calculate_cosine_similarity(user_prompt, gpt4_response)

print(f"Cosine Similarity Score: {similarity_score:.3f}")

#with augmented user prompt
similarity_score = calculate_cosine_similarity(augmented_input, gpt4_response)

print(f"Cosine Similarity Score: {similarity_score:.3f}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]


similarity_score = calculate_cosine_similarity_with_embeddings(augmented_input, gpt4_response)
print(f"Cosine Similarity Score: {similarity_score:.3f}")
