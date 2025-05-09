# AI Samples Repository

This repository contains basic AI samples demonstrating various techniques and workflows using Python. Each script focuses on a specific aspect of AI, such as embeddings, retrieval-augmented generation (RAG), and OpenAI API integration.

## Python Scripts

### 1. `open_ai_connection.py`

This script demonstrates how to establish a connection to OpenAI's API using an API key. It serves as a foundational example for integrating OpenAI's services into Python applications.

### 2. `embeddings.py`

This script computes text embeddings for a list of sentences using OpenAI's embedding model. It calculates cosine distances between the embeddings to measure semantic similarity and visualizes the results as a heatmap.

### 3. `index.py`

This script processes text files and creates a FAISS vector database for efficient similarity search. It uses the LlamaIndex framework with OpenAI's embedding model to generate vector representations of the text and stores them in the vector database. This is a key component for retrieval-augmented generation workflows.

### 4. `rag.py`

This script implements Retrieval-Augmented Generation (RAG) using LlamaIndex framework, FAISS vector database and OpenAI's models. It retrieves relevant context from the vector database and uses it to answer questions. The script demonstrates how to combine retrieval and generation for more accurate and context-aware responses.

## Setup

To set up the environment and install the required packages, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set the `OPENAI_API_KEY` environment variable with your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=your_api_key  # On Windows: set OPENAI_API_KEY=your_api_key
   ```

You are now ready to run the scripts in this repository. Each script is self-contained and can be executed independently.
