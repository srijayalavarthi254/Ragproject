import os
import openai

# Read API key from file
try:
    with open(r"C:\Users\srija\OneDrive\Desktop\ragprojectt\api_key.txt", "r") as file:
        API_KEY = file.readline().strip()
except FileNotFoundError:
    print("Error: 'api_key.txt' file not found.")
    exit()

# Set the API key
os.environ['OPENAI_API_KEY'] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Test the API key with a valid call
try:
    engines = openai.Engine.list()
    print("API Key is working! Available engines:", [engine['id'] for engine in engines['data']])
except openai.error.AuthenticationError:
    print("Invalid API Key. Please check and update your API key.")
except Exception as e:
    print("An error occurred:", str(e))



import openai
import os

# Set your API key
API_KEY = 'ur api_key'  # Replace with your API key
openai.api_key = API_KEY

gptmodel = "gpt-4"  # Correct model name

def call_llm_with_full_text(itext):
    # Join all lines to form a single string
    text_input = '\n'.join(itext)
    prompt = f"Please elaborate on the following content:\n{text_input}"

    try:
        # Make the API call using the updated OpenAI client
        response = openai.ChatCompletion.create(
            model=gptmodel,
            messages=[
                {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
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

# Example usage
itext = ["Here is some text that you want to elaborate on circket.", "Please provide detailed explanations."]
result = call_llm_with_full_text(itext)
print(result)

# Formated response

import textwrap

def print_formatted_response(response):
    # Define the width for wrapping the text
    wrapper = textwrap.TextWrapper(width=80)  # Set to 80 columns wide, but adjust as needed
    wrapped_text = wrapper.fill(text=response)

    # Print the formatted response with a header and footer
    print("Response:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")

## The Data

import textwrap

# List of text records
db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    "It innovatively combines the capabilities of neural network-based language models with retrieval systems to enhance the generation of text, making it more accurate, informative, and contextually relevant.",
    "This methodology leverages the strengths of both generative and retrieval architectures to tackle complex tasks that require not only linguistic fluency but also factual correctness and depth of knowledge.",
    "At the core of Retrieval Augmented Generation (RAG) is a generative model, typically a transformer-based neural network, similar to those used in models like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers).",
    "This component is responsible for producing coherent and contextually appropriate language outputs based on a mixture of input prompts and additional information fetched by the retrieval component.",
    "Complementing the language model is the retrieval system, which is usually built on a database of documents or a corpus of texts.",
    "This system uses techniques from information retrieval to find and fetch documents that are relevant to the input query or prompt.",
    "The mechanism of relevance determination can range from simple keyword matching to more complex semantic search algorithms which interpret the meaning behind the query to find the best matches.",
    "This component merges the outputs from the language model and the retrieval system.",
    "It effectively synthesizes the raw data fetched by the retrieval system into the generative process of the language model.",
    "The integrator ensures that the information from the retrieval system is seamlessly incorporated into the final text output, enhancing the model's ability to generate responses that are not only fluent and grammatically correct but also rich in factual details and context-specific nuances.",
    "When a query or prompt is received, the system first processes it to understand the requirement or the context.",
    "Based on the processed query, the retrieval system searches through its database to find relevant documents or information snippets.",
    "This retrieval is guided by the similarity of content in the documents to the query, which can be determined through various techniques like vector embeddings or semantic similarity measures.",
    "The retrieved documents are then fed into the language model.",
    "In some implementations, this integration happens at the token level, where the model can access and incorporate specific pieces of information from the retrieved texts dynamically as it generates each part of the response.",
    "The language model, now augmented with direct access to retrieved information, generates a response.",
    "This response is not only influenced by the training of the model but also by the specific facts and details contained in the retrieved documents, making it more tailored and accurate.",
    "By directly incorporating information from external sources, Retrieval Augmented Generation (RAG) models can produce responses that are more factual and relevant to the given query.",
    "This is particularly useful in domains like medical advice, technical support, and other areas where precision and up-to-date knowledge are crucial.",
    "Retrieval Augmented Generation (RAG) systems can dynamically adapt to new information since they retrieve data in real-time from their databases.",
    "This allows them to remain current with the latest knowledge and trends without needing frequent retraining.",
    "With access to a wide range of documents, Retrieval Augmented Generation (RAG) systems can provide detailed and nuanced answers that a standalone language model might not be capable of generating based solely on its pre-trained knowledge.",
    "While Retrieval Augmented Generation (RAG) offers substantial benefits, it also comes with its challenges.",
    "These include the complexity of integrating retrieval and generation systems, the computational overhead associated with real-time data retrieval, and the need for maintaining a large, up-to-date, and high-quality database of retrievable texts.",
    "Furthermore, ensuring the relevance and accuracy of the retrieved information remains a significant challenge, as does managing the potential for introducing biases or errors from the external sources.",
    "In summary, Retrieval Augmented Generation represents a significant advancement in the field of artificial intelligence, merging the best of retrieval-based and generative technologies to create systems that not only understand and generate natural language but also deeply comprehend and utilize the vast amounts of information available in textual form.",
    "A RAG vector store is a database or dataset that contains vectorized data points."
]

# Join the list into one paragraph
paragraph = ' '.join(db_records)

# Use textwrap to wrap the paragraph at 80 characters
wrapped_text = textwrap.fill(paragraph, width=80)

# Print the wrapped text
print(wrapped_text)

# The Query

query = "define a rag store"

# Call the function and print the result
llm_response = call_llm_with_full_text(query)
print_formatted_response(llm_response)

#Part 2: Advanced Techniques and Evaluation
# 1.Retrieval Metrics
# Cosine Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate cosine similarity between two texts
def calculate_cosine_similarity(text1, text2):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',      # Remove common stop words
        use_idf=True,              # Use inverse document frequency
        norm='l2',                 # Normalize by L2 norm (unit vector length)
        ngram_range=(1, 2),        # Use unigrams and bigrams
        sublinear_tf=True,         # Apply sublinear TF scaling
        analyzer='word'            # Use word-level analysis
    )

    # Convert the texts into their TF-IDF representations
    tfidf = vectorizer.fit_transform([text1, text2])

    # Compute the cosine similarity between the two TF-IDF vectors
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

# Example texts for comparison
text1 = "Retrieval Augmented Generation combines retrieval with generation models."
text2 = "RAG models merge the strengths of retrieval and generation systems."

# Calculate and print the cosine similarity
similarity_score = calculate_cosine_similarity(text1, text2)
print(f"Cosine Similarity: {similarity_score}")

# Enhanced Similarity

import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmatized_words = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemmatized_words.append(token.lemma_)
    return lemmatized_words

def expand_with_synonyms(words):
    expanded_words = words.copy()
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return expanded_words

def calculate_enhanced_similarity(text1, text2):
    # Preprocess and tokenize texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Expand with synonyms
    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Count word frequencies
    freq1 = Counter(words1_expanded)
    freq2 = Counter(words2_expanded)

    # Create a set of all unique words
    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    # Create frequency vectors
    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    # Convert lists to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate cosine similarity
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cosine_similarity

#2.Naive RAG
# Keyword search and matching


def find_best_match_keyword_search(query, db_records):
    best_score = 0
    best_record = None

    # Split the query into individual keywords
    query_keywords = set(query.lower().split())

    # Iterate through each record in db_records
    for record in db_records:
        # Split the record into keywords
        record_keywords = set(record.lower().split())

        # Calculate the number of common keywords
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)

        # Update the best score and record if the current score is higher
        if current_score > best_score:
            best_score = current_score
            best_record = record

    return best_score, best_record

# Assuming 'query' and 'db_records' are defined in previous cells in your Colab notebook
best_keyword_score, best_matching_record = find_best_match_keyword_search(query, db_records)

print(f"Best Keyword Score: {best_keyword_score}")
print_formatted_response(best_matching_record)

# Metrics
# Cosine Similarity
score = calculate_cosine_similarity(query, best_matching_record)
print(f"Best Cosine Similarity Score: {score:.3f}")

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, response)
print(f"Enhanced Similarity:, {similarity_score:.3f}")


#Augmented input

augmented_input=query+ ": "+ best_matching_record
     
print_formatted_response(augmented_input)

#Generation

# Call the function and print the result
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)

#3.Advanced RAG
# 3.1.Vector search

#Search function

def find_best_match(text_input, records):
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

best_similarity_score, best_matching_record = find_best_match(query, db_records)
print_formatted_response(best_matching_record)
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, best_matching_record)
print(f"Enhanced Similarity:, {similarity_score:.3f}")


augmented_input=query+": "+best_matching_record
print_formatted_response(augmented_input)

# Call the function and print the result
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)

# 3.2.Index-based search

#Search Function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix

def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()  # Get the index of the highest similarity score
    best_score = similarities[0, best_index]
    return best_score, best_index

vectorizer, tfidf_matrix = setup_vectorizer(db_records)

best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
best_matching_record = db_records[best_index]

print_formatted_response(best_matching_record)

# Cosine Similarity
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")
print_formatted_response(best_matching_record)

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, response)
print(f"Enhanced Similarity:, {similarity_score:.3f}")


#Feature extraction

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)

    # Convert the TF-IDF matrix to a DataFrame for display purposes
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Display the DataFrame
    print(tfidf_df)

    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_vectorizer(db_records)

#Augmented input

augmented_input=query+": "+best_matching_record
print_formatted_response(augmented_input)

#Generation
# Call the function and print the result
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)

#4.Modular RAG

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalComponent:
    def __init__(self, method='vector'):
        self.method = method
        if self.method == 'vector' or self.method == 'indexed':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def fit(self, records):
      self.documents = records  # Initialize self.documents here
      if self.method == 'vector' or self.method == 'indexed':
        self.tfidf_matrix = self.vectorizer.fit_transform(records)

    def retrieve(self, query):
        if self.method == 'keyword':
            return self.keyword_search(query)
        elif self.method == 'vector':
            return self.vector_search(query)
        elif self.method == 'indexed':
            return self.indexed_search(query)

    def keyword_search(self, query):
        best_score = 0
        best_record = None
        query_keywords = set(query.lower().split())
        for index, doc in enumerate(self.documents):
            doc_keywords = set(doc.lower().split())
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords)
            if score > best_score:
                best_score = score
                best_record = self.documents[index]
        return best_record

    def vector_search(self, query):
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]

    def indexed_search(self, query):
        # Assuming the tfidf_matrix is precomputed and stored
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]

#Modular RAG Strategies

# Usage example
retrieval = RetrievalComponent(method='vector')  # Choose from 'keyword', 'vector', 'indexed'
retrieval.fit(db_records)
best_matching_record = retrieval.retrieve(query)

print_formatted_response(best_matching_record)

#Metrics

# Cosine Similarity
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")
print_formatted_response(best_matching_record)

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, response)
print("Enhanced Similarity:", similarity_score)

#Augmented Input

augmented_input=query+ " "+ best_matching_record
print_formatted_response(augmented_input)

#Generation

# Call the function and print the result
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
     
