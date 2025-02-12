# Install the required libraries
# pip install llama-index-vector-stores-deeplake==0.1.6
# pip install llama-index-llms-openai==0.1.29

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import StorageContext
import os
import openai
import requests
import re
from bs4 import BeautifulSoup

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

import deeplake  # Correctly import Deep Lake

# Define the dataset path
dataset_path_llm = "hub://srijayalavarthi254/drone"

# Load the dataset
ds_llm = deeplake.load(dataset_path_llm)

# Limit to first 10 images for testing
limited_ds_llm = ds_llm[:10]

# Print dataset information
print(limited_ds_llm)

import json
import pandas as pd
import numpy as np

# Create a dictionary to hold the data
data_llm = {}

# Iterate through the tensors in the limited dataset
for tensor_name in limited_ds_llm.tensors:
    tensor_data = limited_ds_llm[tensor_name].numpy()

    # Check if the tensor is multi-dimensional
    if tensor_data.ndim > 1:
        # Flatten multi-dimensional tensors
        data_llm[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        # Convert 1D tensors directly to lists and decode text
        if tensor_name == "text":
            data_llm[tensor_name] = [t.tobytes().decode('utf-8') if t else "" for t in tensor_data]
        else:
            data_llm[tensor_name] = tensor_data.tolist()

# Create a Pandas DataFrame from the dictionary
df_llm = pd.DataFrame(data_llm)
df_llm

# Ensure 'text' column is of type string
df_llm['text'] = df_llm['text'].astype(str)

# Create documents with IDs
documents_llm = [Document(text=row['text'], doc_id=str(row['id'])) for _, row in df_llm.iterrows()]

#Initializing the LLM Query Engine
from llama_index.core import VectorStoreIndex
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)

vector_query_engine_llm = vector_store_index_llm.as_query_engine(similarity_top_k=2, temperature=0.1, num_output=1024)

# User input for multimodal RAG
user_input = "How do drones identify a truck?"

import time
import textwrap
#start the timer
start_time = time.time()
llm_response = vector_query_engine_llm.query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(textwrap.fill(str(llm_response), 100))

# Loading and visualizing the multimodal dataset

# Define the dataset path
dataset_path = 'hub://activeloop/visdrone-det-train'

# Load the dataset
ds = deeplake.load(dataset_path)

# Limit the dataset to the first 10 images for testing
limited_ds = ds[:10]

# Now you can perform your operations on `limited_ds` instead of the full dataset
limited_ds.summary()  # To check the details of the limited dataset
limited_ds.visualize()  # Visualize the smaller dataset

import pandas as pd

# Create an empty DataFrame with the defined structure
df = pd.DataFrame(columns=['image', 'boxes', 'labels'])

# Iterate through the samples using enumerate
for i, sample in enumerate(limited_ds):

    # Image data (choose either path or compressed representation)
    # df.loc[i, 'image'] = sample.images.path  # Store image path
    df.loc[i, 'image'] = sample.images.tobytes()  # Store compressed image data

    # Bounding box data (as a list of lists)
    boxes_list = sample.boxes.numpy(aslist=True)
    df.loc[i, 'boxes'] = [box.tolist() for box in boxes_list]

    # Label data (as a list)
    label_data = sample.labels.data()
    df.loc[i, 'labels'] = label_data['text']

df

labels_list = limited_ds.labels.info['class_names']
labels_list

# Selecting an image

# choose an image
ind = 0
image = limited_ds.images[ind].numpy()  # Fetch the first image and return a numpy array

import deeplake
from IPython.display import display
from PIL import Image
import cv2  # Import OpenCV

image = limited_ds.images[0].numpy()

# Convert from BGR to RGB (if necessary)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create PIL Image and display
img = Image.fromarray(image_rgb)
display(img)

# Adding Bounding boxes and saving the image

labels = limited_ds.labels[ind].data()  # Fetch the labels in the selected image
print(labels)

values = labels['value']
text_labels = labels['text']

# Determine the maximum text label length for formatting
max_text_length = max(len(label) for label in text_labels)

# Print the header
print(f"{'Index':<10}{'Label':<{max_text_length + 2}}")
print("-" * (10 + max_text_length + 2))  # Add a separator line

# Print the indices and labels in two columns
for index, label in zip(values, text_labels):
    print(f"{index:<10}{label:<{max_text_length + 2}}")

limited_ds.labels[ind].info['class_names']  # class names of the selected image


def display_image_with_bboxes(image_data, bboxes, labels, label_name, ind=0):
    #Displays an image with bounding boxes for a specific label.

    image_bytes = io.BytesIO(image_data)
    img = Image.open(image_bytes)

    # Extract class names specifically for the selected image
    class_names = limited_ds.labels[ind].info['class_names']

    # Filter for the specific label (or display all if class names are missing)
    if class_names is not None:
        try:
            label_index = class_names.index(label_name)
            relevant_indices = np.where(labels == label_index)[0]
        except ValueError:
            print(f"Warning: Label '{label_name}' not found. Displaying all boxes.")
            relevant_indices = range(len(labels))
    else:
        relevant_indices = []  # No labels found, so display no boxes

    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(bboxes):  # Enumerate over bboxes
        if idx in relevant_indices:   # Check if this box is relevant
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), label_name, fill="red")
    # Save the image
    save_path = "boxed_image.jpg"
    img.save(save_path)
    display(img)

import io
from PIL import ImageDraw
# Fetch labels and image data for the selected image
labels = limited_ds.labels[ind].data()['value']
image_data = limited_ds.images[ind].tobytes()
bboxes = limited_ds.boxes[ind].numpy()
ibox = "truck"  # class in image

# Display the image with bounding boxes for the label chosen
display_image_with_bboxes(image_data, bboxes, labels, label_name=ibox)

# Building a multimodal search engine

# The DataFrame is named 'df'
df['doc_id'] = df.index.astype(str)  # Create unique IDs from the row indices

# Create documents (extract relevant text for each image's labels)
documents = []
for _, row in df.iterrows():
    text_labels = row['labels']  # Each label is now a string
    text = " ".join(text_labels)  # Join text labels into a single string
    document = Document(text=text, doc_id=row['doc_id'])
    documents.append(document)

from llama_index.core import VectorStoreIndex
vector_store_index = VectorStoreIndex.from_documents(documents)

vector_store_index.index_struct   # Load the full dataset

vector_query_engine = vector_store_index.as_query_engine(similarity_top_k=1, temperature=0.1, num_output=1024)

import time
start_time = time.time()
response = vector_query_engine.query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")

print(textwrap.fill(str(response), 100))

from itertools import groupby

def get_unique_words(text):
    text = text.lower().strip()
    words = text.split()
    unique_words = [word for word, _ in groupby(sorted(words))]
    return unique_words

for node in response.source_nodes:
    print(node.node_id)
    # Get unique words from the node text:
    node_text = node.get_text()
    unique_words = get_unique_words(node_text)
    print("Unique Words in Node Text:", unique_words)


# deleting any image prevsiously saved
#!rm /content/boxed_image.jpg

import io
from PIL import Image

def process_and_display(response, df, ds, unique_words):
    """Processes nodes, finds corresponding images in dataset, and displays them with bounding boxes.

    Args:
        response: The response object containing source nodes.
        df: The DataFrame with doc_id information.
        ds: The dataset containing images, labels, and boxes.
        unique_words: The list of unique words for filtering.
    """

    for i, sample in enumerate(ds):
        for node in response.source_nodes:
            related_node_info = next(iter(node.node.relationships.values()))
            original_doc_id = related_node_info.node_id

            try:
                row_index = df[df['doc_id'] == original_doc_id].index[0]
            except IndexError:
                continue  # Skip if doc_id not found in the DataFrame

            if i == row_index:
                image_bytes = io.BytesIO(sample.images.tobytes())
                img = Image.open(image_bytes)

                labels = ds.labels[i].data()['value']
                image_data = ds.images[i].tobytes()
                bboxes = ds.boxes[i].numpy()
                ibox = unique_words[0]  # class in image

                display_image_with_bboxes(image_data, bboxes, labels, label_name=ibox)

# Assuming you have your 'response', 'df', 'ds', and 'unique_words' objects prepared:
process_and_display(response, df, ds, unique_words)


#Multimodal Modular Summary
#Assuming the processes have been activated.

from PIL import Image
def display_source_image(image_path):
  try:
    with Image.open(image_path) as img:
        display(img)
  except FileNotFoundError:
    print("Error: The image file was not found at the specified path.")
  except Exception as e:  # Catch other potential errors
    print(f"Error loading image: {e}")
     

# 1.user input=user_input
print(user_input)
# 2.LLM response
print(textwrap.fill(str(llm_response), 100))
# 3.Multimodal reponse
image_path = "/content/boxed_image.jpg"
display_source_image(image_path)

#Performance metric

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
     

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]


#LLM performance

llm_similarity_score = calculate_cosine_similarity_with_embeddings(user_input, str(llm_response))
print(user_input)
print(llm_response)
print(f"Cosine Similarity Score: {llm_similarity_score:.3f}")

#Multimodal performance

#Set Up OpenAI API Key

import openai

# Set the API key
api_key = ''  # Replace with your API key
#openai.api_key = API_KEY

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

#Encode the Image to Base64

import base64

# Path to your image
IMAGE_PATH = "C:/Users/srija/OneDrive/Desktop/ragprojectt/boxed_image.jpg"

# Encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Convert the image to Base64
base64_image = encode_image(IMAGE_PATH)
#Send the Image to OpenAI's GPT-4o

MODEL = "gpt-4o"

# Example unique word (Modify as needed)
unique_words = ["truck"]
u_word = unique_words[0]  # Example: "truck"
print(f"Unique Word: {u_word}")

# Send request to OpenAI API
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": f"You are a helpful assistant that analyzes images that contain {u_word}."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Analyze the following image, tell me if there is one {u_word} or more in the bounding boxes and analyze them:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.0,
)


# Extract and print the response
response_image = response.choices[0].message.content
print(response_image)

#Calculate Multimodal Similarity Score

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embedding1 =openai.embeddings.create(input=[text1], model="text-embedding-ada-002").data[0].embedding
    embedding2 =openai.embeddings.create(input=[text2], model="text-embedding-ada-002").data[0].embedding

    
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Example user input for comparison
user_input = "A truck is present in the image."
resp = user_input + " " + u_word + "s"

# Compute similarity
multimodal_similarity_score = calculate_cosine_similarity_with_embeddings(resp, str(response_image))
print(f"Cosine Similarity Score: {multimodal_similarity_score:.3f}")

#Compute Final Multimodal RAG Score

# Assuming llm_similarity_score is computed somewhere else
llm_similarity_score = 0.85  # Example score

# Compute final score
final_score = (llm_similarity_score + multimodal_similarity_score) / 2
print(f"Multimodal Modular RAG Score: {final_score:.3f}")
