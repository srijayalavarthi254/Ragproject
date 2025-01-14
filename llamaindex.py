# Install necessary packages
#!pip install llama-index-vector-stores-deeplake==0.1.6
#!pip install deeplake==3.9.18
#!pip install llama-index==0.10.64
#!pip install sentence-transformers==3.0.1

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import StorageContext
import os
import openai
import requests
import re
from bs4 import BeautifulSoup

# Check if Llama Index is working
print("Llama Index is working correctly!")

# Retrieve and set the OpenAI API key
f = open("C:/Users/srija/OneDrive/Desktop/ragprojectt/api_key.txt", "r")
API_KEY = f.readline().strip()
f.close()

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve and set the Activeloop API token
f = open("C:/Users/srija/OneDrive/Desktop/ragprojectt/activeloop.txt", "r")
API_token = f.readline().strip()
f.close()
ACTIVELOOP_TOKEN = API_token
os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

# Pipeline 1: Collecting and preparing the documents
urls = [
    "https://github.com/VisDrone/VisDrone-Dataset",
    "https://paperswithcode.com/dataset/visdrone",
    "https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf",
    # Add other URLs as needed
]

# Function to clean and extract content from fetched URLs
def clean_text(content):
    content = re.sub(r'\[\d+\]', '', content)  # Remove references
    content = re.sub(r'[^\w\s\.]', '', content)  # Remove punctuation (except periods)
    return content

def fetch_and_clean(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('div', {'class': 'mw-parser-output'}) or soup.find('div', {'id': 'content'})
        if content is None:
            return None

        for section_title in ['References', 'Bibliography', 'External links', 'See also', 'Notes']:
            section = content.find('span', id=section_title)
            while section:
                for sib in section.parent.find_next_siblings():
                    sib.decompose()
                section.parent.decompose()
                section = content.find('span', id=section_title)

        text = content.get_text(separator=' ', strip=True)
        text = clean_text(text)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None

# Directory to store the output files
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

# Fetch and save content from URLs
for url in urls:
    article_name = url.split('/')[-1].replace('.html', '')
    filename = os.path.join(output_dir, f"{article_name}.txt")

    clean_article_text = fetch_and_clean(url)
    if clean_article_text:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(clean_article_text)

print(f"Content written to files in the '{output_dir}' directory.")

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Adjust file extension as needed
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                document_text = file.read()
                documents.append(Document(text=document_text))
    return documents

# Load documents from the 'data' folder
documents = load_documents_from_directory('./data/')
print(documents[0].text if documents else "No documents found.")

# Step 2: Create and populate the Deep Lake Vector Store
vector_store_path = "hub://srijayalavarthi254/rag2"
dataset_path = "hub://srijayalavarthi254/rag2"

# Create a DeepLake VectorStore
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 3: Create an index over the documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("Index created successfully!")


# Import deeplake library
import deeplake

# Define your dataset path
dataset_path = "hub://srijayalavarthi254/rag2"  # Replace with your dataset path

# Load the dataset
ds = deeplake.load(dataset_path)

# Optionally, print some information about the dataset to confirm it's loaded
print("Dataset loaded successfully!")
print(ds)  # This will print the details of the dataset loaded


import json
import pandas as pd
import numpy as np

# Assuming 'ds' is your loaded Deep Lake dataset

# Create a dictionary to hold the data
data = {}

# Iterate through the tensors in the dataset
for tensor_name in ds.tensors:
    tensor_data = ds[tensor_name].numpy()

    # Check if the tensor is multi-dimensional
    if tensor_data.ndim > 1:
        # Flatten multi-dimensional tensors
        data[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        # Convert 1D tensors directly to lists and decode text
        if tensor_name == "text":
            data[tensor_name] = [t.tobytes().decode('utf-8') if t else "" for t in tensor_data]
        else:
            data[tensor_name] = tensor_data.tolist()

# Create a Pandas DataFrame from the dictionary
df = pd.DataFrame(data)
# Function to display a selected record
def display_record(record_number):
    record = df.iloc[record_number]
    display_data = {
        "ID": record.get("id", "N/A"),
        "Metadata": record.get("metadata", "N/A"),
        "Text": record.get("text", "N/A"),
        "Embedding": record.get("embedding", "N/A")
    }

    # Print the ID
    print("ID:")
    print(display_data["ID"])
    print()

    # Print the metadata in a structured format
    print("Metadata:")
    metadata = display_data["Metadata"]
    if isinstance(metadata, list):
        for item in metadata:
            for key, value in item.items():
                print(f"{key}: {value}")
            print()
    else:
        print(metadata)
    print()

    # Print the text
    print("Text:")
    print(display_data["Text"])
    print()

    # Print the embedding
    print("Embedding:")
    print(display_data["Embedding"])
    print()

# Function call to display a record
rec = 0  # Replace with the desired record number
display_record(rec)

# Ensure 'text' column is of type string
df['text'] = df['text'].astype(str)
# Create documents with IDs
documents = [Document(text=row['text'], doc_id=str(row['id'])) for _, row in df.iterrows()]

#Pipeline 3:Index-based RAG
#User input and RAG parameters

user_input="How does an object can be detected"

#similarity_top_k
k=3
#temperature
temp=0.1
#num_output
mt=1024

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]

#Vector store index query engine
from llama_index.core import VectorStoreIndex
vector_store_index = VectorStoreIndex.from_documents(documents)
print(type(vector_store_index))
vector_query_engine = vector_store_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
print(type(vector_query_engine))

#Query response and source
import pandas as pd
import textwrap

def index_query(input_query):
    response = vector_query_engine.query(input_query)

    # Optional: Print a formatted view of the response (remove if you don't need it in the output)
    print(textwrap.fill(str(response), 100))

    node_data = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        node_info = {
            'Node ID': node.id_,
            'Score': node_with_score.score,
            'Text': node.text
        }
        node_data.append(node_info)

    df = pd.DataFrame(node_data)

    # Instead of printing, return the DataFrame and the response object
    return df, response
import time
#start the timer
start_time = time.time()
df, response = index_query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(df.to_markdown(index=False, numalign="left", stralign="left"))  # Display
nodeid=response.source_nodes[0].node_id
nodeid
response.source_nodes[0].get_text()

#Optimized chunking
# Assuming you have the 'response' object from query_engine.query()

for node_with_score in response.source_nodes:
    node = node_with_score.node  # Extract the Node object from NodeWithScore
    chunk_size = len(node.text)
    print(f"Node ID: {node.id_}, Chunk Size: {chunk_size} characters")

#Performance metric

import numpy as np

def info_metrics(response):
  # Calculate the performance (handling None scores)
  scores = [node.score for node in response.source_nodes if node.score is not None]
  if scores:  # Check if there are any valid scores
      weights = np.exp(scores) / np.sum(np.exp(scores))
      perf = np.average(scores, weights=weights) / elapsed_time
  else:
      perf = 0  # Or some other default value if all scores are None

  average_score=np.average(scores, weights=weights)
  print(f"Average score: {average_score:.4f}")
  print(f"Query execution time: {elapsed_time:.4f} seconds")
  print(f"Performance metric: {perf:.4f}")

info_metrics(response)

#Tree index query engine

from llama_index.core import TreeIndex

# Create a Tree Index from the loaded documents
tree_index = TreeIndex.from_documents(documents)

# Print some details about the tree index
print("Tree Index created successfully!")
print(tree_index)
print(type(tree_index))

tree_query_engine = tree_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
import time
import textwrap
# Start the timer
start_time = time.time()
response = tree_query_engine.query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(textwrap.fill(str(response), 100))

#Performance metric
similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(response))
print(f"Cosine Similarity Score: {similarity_score:.3f}")
print(f"Query execution time: {elapsed_time:.4f} seconds")
performance=similarity_score/elapsed_time
print(f"Performance metric: {performance:.4f}")

#List index query engine
from llama_index.core import ListIndex
list_index = ListIndex.from_documents(documents)
print(type(list_index))

list_query_engine = list_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
#start the timer
start_time = time.time()
response = list_query_engine.query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(textwrap.fill(str(response), 100))

#Performance metric
similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(response))
print(f"Cosine Similarity Score: {similarity_score:.3f}")
print(f"Query execution time: {elapsed_time:.4f} seconds")
performance=similarity_score/elapsed_time
print(f"Performance metric: {performance:.4f}")

#Keyword index query index
from llama_index.core import KeywordTableIndex
keyword_index = KeywordTableIndex.from_documents(documents)
# Extract data for DataFrame
data = []
for keyword, doc_ids in keyword_index.index_struct.table.items():
    for doc_id in doc_ids:
        data.append({"Keyword": keyword, "Document ID": doc_id})

# Create the DataFrame
df = pd.DataFrame(data)
df

keyword_query_engine = keyword_index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
import time

# Start the timer
start_time = time.time()

# Execute the query (using .query() method)
response = keyword_query_engine.query(user_input)

# Stop the timer
end_time = time.time()

# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(textwrap.fill(str(response), 100))

#Performance metric
similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(response))
print(f"Cosine Similarity Score: {similarity_score:.3f}")
print(f"Query execution time: {elapsed_time:.4f} seconds")
performance=similarity_score/elapsed_time
print(f"Performance metric: {performance:.4f}")
