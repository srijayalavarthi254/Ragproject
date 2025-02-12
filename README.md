

## **Building Index-Based RAG with LlamaIndex, Deep Lake and Open AI**

## **Overview**

Ragproject is a **Retrieval-Augmented Generation (RAG)** framework that combines Natural Language Processing (NLP) and machine learning techniques to provide enhanced text generation and search capabilities. The project integrates external knowledge bases to generate contextual responses, leveraging tools like OpenAI's GPT models, vector embeddings, and cosine similarity for text similarity evaluation.

---

## **Key Features**

- **Retrieval-augmented text generation:** Improves contextuality by combining retrieved data with the input query.
  
- **Efficient vector search:** Uses Deep Lake for efficient storage and retrieval of text embeddings.
  
- **Cosine similarity evaluation:** Measures the relevance of retrieved documents for improved accuracy.
  
- **Pre-trained transformer models for embeddings:** Supports sentence-transformers for document vectorization.
  
- **Flexible integration with external knowledge bases:** Enables integration with external data for enhanced query handling.

---

## **Installation**

To set up this project locally, follow the steps below.

### **Prerequisites:**

- Python 3.7 or higher
- `pip` (Python package installer)
- A valid OpenAI API key

### **Dependencies:**

--Install the required Python libraries by running the following commands:

    ->pip install openai
    
    ->pip install deeplake
    
    ->pip install sentence-transformers
    
    ->pip install scikit-learn
    
    ->pip install pandas
    
    ->pip install textwrap3 # Optional, for text formatting
    
    ->pip install markdown # Optional, for rendering markdown responses
    
    ->pip install nltk
    
    ->pip install spacy


**Additionally, download the language model for SpaCy:


python -m spacy download en_core_web_sm


Install additional required libraries:


    -> pip install llama-index-vector-stores-deeplake==0.1.6
    
    ->pip install deeplake==3.9.18
    
    ->pip install llama-index==0.10.64
    
    ->pip install sentence-transformers==3.0.1


## **Usage**

### **Clone the Repository:**


git clone https://github.com/yourusername/Ragproject.git
cd Ragproject


### **Add API Keys:**

- Save your OpenAI API key in a file named `api_key.txt`.
- Save your Activeloop token in a file named `activeloop.txt`.

Place these files in the root directory of the project:


Ragproject/
├── api_key.txt
├── activeloop.txt

### **Run the Project:**

To execute the main script:


python main.py


*For specific tasks, you can run other modules such as:


python augmented_generation.py
python datacollections.py
python embeddings_vector.py




## **Project Structure**

Below is the structure of the project:


Ragproject/
├── main.py                      # Main script for running the RAG model

├── augmented_generation.py       # Handles query augmentation and GPT-based generation

├── datacollections.py            # Data collection and preprocessing script

├── embeddings_vector.py          # Embedding generation and vector storage

├── llamaindex.py                 # Index-based implementation

├── multi_modular_Raag.py          # New modular RAG implementation

├── llm.txt                       # Logs or supporting notes for LLM queries

├── api_key.txt                   # Contains the OpenAI API key (DO NOT SHARE)

├── activeloop.txt                # Contains the Activeloop token

├── requirements.txt              # List of all required dependencies

├── README.md                     # Project documentation (this file)


## **New Feature: `multi_modular_rag.py`**

### **Overview:**

The `multi_modular_rag.py` file introduces a more modular approach to the Retrieval-Augmented Generation (RAG) pipeline. It integrates various modules for data collection, query augmentation, and document retrieval, providing a more structured way to manage and extend the core functionality of the RAG framework.

### **How It Works:**

1. **Data Retrieval**: 
   - The input query is processed, and relevant documents are retrieved based on similarity scores using cosine similarity.

2. **Query Augmentation**:
   - The retrieved documents are combined with the user’s query, enriching it with additional context for enhanced accuracy.

3. **Text Generation**:
   - The augmented query is passed to OpenAI’s GPT model (such as `text-davinci-003`) to generate a detailed, context-aware response.

### **Main Functions:**

- **`load_api_keys()`**: Loads API keys from files.
  
- **`preprocess_data(query)`**: Collects documents and generates embeddings.
  
- **`retrieve_and_augment(query)`**: Retrieves relevant documents using cosine similarity and augments the query using GPT.
  
- **`run_rag_pipeline(query)`**: The main pipeline that retrieves, augments, and generates a response using the RAG framework.

### **How to Use:**

To run the modular RAG pipeline, you can execute the `multi_modular_rag.py` file:


python multi_modular_rag.py


This will prompt you to enter a query, and the system will respond with a contextually enriched answer generated by GPT.


## **Data Retrieval & Augmented Generation**

### **Data Retrieval:**

1. **Embedding Queries**: 
   - Input queries are embedded using **sentence-transformers** or OpenAI’s embedding models.
   
2. **Retrieving Similar Vectors**:
   - Similar vectors are retrieved using **Deep Lake** for storing and fetching the embedded document vectors.

### **Augmented Generation:**

1. **Enhancing Context**: 
   - The retrieved data is combined with the user’s original query to provide enriched context for a more accurate response.

2. **GPT-Based Text Generation**: 
   - Using OpenAI's models (e.g., `text-davinci-003` or `gpt-4`), a detailed response is generated based on the augmented query.

---

## **Features**

- **Document Retrieval**: Collects and processes documents from URLs.
  
- **Vector Store**: Uses Deep Lake for storing document embeddings and efficient retrieval.
  
- **Indexing**: Generates and queries document indexes for fast retrieval of relevant data.
  
- **Search Engine**: Executes queries on documents using various indexing strategies:
  - **Vector Store**: Uses cosine similarity for document retrieval.
    
  - **Tree Index**: Organizes documents hierarchically for efficient searching.
    
  - **List Index**: Indexes documents in a list format for retrieval.
    
  - **Keyword Index**: Queries based on keywords for document matching.

- **Performance Metrics**:
  - Measures execution time and query performance, ensuring that the system operates efficiently under various conditions.

---

## **Query Engine**

Various query engines are available to search documents based on user input:

- **Vector Store Index**: Uses cosine similarity to retrieve documents based on vector proximity.
  
- **Tree Index**: Organizes documents hierarchically for optimized query searching.
  
- **List Index**: Indexes documents sequentially for retrieval.
  
- **Keyword Index**: Searches documents based on keyword matching.

---

## **Evaluation**

The **Ragproject** evaluates the relevance of the generated responses using **cosine similarity**, ensuring that the results are highly pertinent to the user's query.

---

## **Performance Evaluation**

The project tracks key performance metrics to gauge the effectiveness of the retrieval and generation process, including:

- **Average Query Execution Time**: Measures how long the system takes to process each query.
  
- **Cosine Similarity Score**: Evaluates the relevance of retrieved documents based on cosine similarity.
  
- **Performance Metric**: A composite metric that evaluates the query speed and relevance.



## **Conclusion**

This **RAG framework** integrates cutting-edge document retrieval and generation techniques, providing a flexible and efficient approach to process and query large document datasets. By combining deep learning models and search algorithms, the project is designed for high performance and scalability in handling a variety of query and document types.

