#Installing the environment

try:
    import deeplake
except ImportError:
    import os
    import sys
    # Install deeplake if it's not already installed
    os.system(f"{sys.executable} -m pip install deeplake==3.9.18")
    import deeplake

print("Deeplake module imported successfully!")

#GitHub grequests.py
#Script to download files from the GitHub repository.

import subprocess

url = "https://raw.githubusercontent.com/Denis2054/RAG-Driven-Generative-AI/main/commons/grequests.py"
output_file = "grequests.py"

# Prepare the curl command
curl_command = [
    "curl",
    "-o", output_file,
    url
]

# Execute the curl command
try:
    subprocess.run(curl_command, check=True)
    print("Download successful.")
except subprocess.CalledProcessError:
    print("Failed to download the file.")

#!pip install openai==1.40.3

# For Google Colab and Activeloop(Deeplake library)
#This line writes the string "nameserver 8.8.8.8" to the file. This is specifying that the DNS server the system
#should use is at the IP address 8.8.8.8, which is one of Google's Public DNS servers.
#with open('/etc/resolv.conf', 'w') as file:
 #  file.write("nameserver 8.8.8.8")

#Retrieving and setting the OpenAI API key
f = open("C:/Users/srija/OneDrive/Desktop/ragprojectt/api_key.txt", "r")
API_KEY=f.readline().strip()
f.close()

#The OpenAI KeyActiveloop and OpenAI API keys
import os
import openai
os.environ['OPENAI_API_KEY'] =API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
     

#Retrieving and setting the Activeloop API token
f = open("C:/Users/srija/OneDrive/Desktop/ragprojectt/activeloop.txt", "r")
API_token=f.readline().strip()
f.close()
ACTIVELOOP_TOKEN=API_token
os.environ['ACTIVELOOP_TOKEN'] =ACTIVELOOP_TOKEN

#Embedding and Storage: populating the vector store
#Downloading and preparing the data

from grequests import download
source_text = "llm.txt"

directory = "Chapter02"
filename = "llm.txt"
download(directory, filename)
# Open the file and read the first 20 lines
with open('llm.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # Print the first 20 lines
    for line in lines[:20]:
        print(line.strip())

# Define the source file path
source_text = "llm.txt"

# Chunk size
CHUNK_SIZE = 1000

try:
    # Open the file and read its content
    with open(source_text, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk the text into smaller segments
    chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # Display the chunked data (optional: for debugging or visualization)
    print(f"Total chunks created: {len(chunked_text)}")
    for idx, chunk in enumerate(chunked_text[:5]):  # Print first 5 chunks as a sample
        print(f"\nChunk {idx + 1}:\n{chunk}")

except FileNotFoundError:
    print(f"Error: The file '{source_text}' was not found. Please make sure the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")

# Import necessary modules
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import deeplake.util

# Replace with your vector store path
vector_store_path = "hub://srijayalavarthi254/text_embedding"

try:
    # Attempt to load the vector store
    vector_store = VectorStore(path=vector_store_path)
    print("Vector store exists")
except FileNotFoundError:
    # Handle the case when the vector store does not exist
    print("Vector store does not exist. You can create it.")
    create_vector_store = True

#The embedding function

# Correcting the embedding function
def embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to a list
    texts = [t.replace("\n", " ") for t in texts]  # Remove newlines
    response = openai.Embedding.create(input=texts, model=model)  # Correct method
    return [item['embedding'] for item in response['data']]  # Extract embeddings

# Embedding and adding to the vector store
add_to_vector_store = True

if add_to_vector_store:
    # Read the source text file
    with open(source_text, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk the text into smaller segments
    CHUNK_SIZE = 1000
    chunked_text = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # Add to the vector store
    vector_store.add(
        text=chunked_text,
        embedding_function=embedding_function,
        embedding_data=chunked_text,
        metadata=[{"source": source_text}] * len(chunked_text)
    )

# Display vector store summary
print(vector_store.summary())

# Loading the dataset from the vector store
ds = deeplake.load(vector_store_path)
#link to see the dataset https://app.activeloop.ai/srijayalavarthi254/text_embedding

# Calculate and display dataset size in MB and GB
ds_size = ds.size_approx()
ds_size_mb = ds_size / 1048576  # Convert bytes to MB
ds_size_gb = ds_size / 1073741824  # Convert bytes to GB
print(f"Dataset size in megabytes: {ds_size_mb:.5f} MB")
print(f"Dataset size in gigabytes: {ds_size_gb:.5f} GB")
