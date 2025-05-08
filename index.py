"""
index.py - process text files and create a FAISS vector database for RAG

reads text files from the 'data' directory, splits them into chunks generates embeddings using 
OpenAI, and stores them in a FAISS vector database that's saved to the 'vectordb' directory
"""
import sys
import os
import logging
from pathlib import Path
import faiss
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

SYS_SUCCESS = 0
SYS_FAILURE = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_db_dir(path):
    """ set up the directory for the vector database """
    vectordb_dir = Path(path)
    vectordb_dir.mkdir(exist_ok=True)
    return vectordb_dir


def read_documents(path):
    """ read documents from the data directory """
    data_dir = Path(path)
    if not data_dir.exists():
        logger.error("Data directory %s does not exist", data_dir)
        return None
    logger.info("Loading documents from %s", data_dir)
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    logger.info("Loaded %d documents", len(documents))
    return documents


def print_chunk_info(index):
    """ print some information about the chunks in the index: """
    print(f"chunks in the index: {len(index._docstore.docs)}")  # pylint: disable=W0212
    for i, (doc_id, doc_val) in enumerate(index._docstore.docs.items()): # pylint: disable=W0212
        if i < 4:
            print(f"{i}: {doc_id}\n{doc_val}")
            print(f"chunk len (bytes): {len(doc_val.text)}")
            print(f"chunk metadata: {doc_val.metadata}")
        else:
            break


def main():
    """ entry point for the script """
    documents = read_documents("data")
    if not documents:
        logger.error("No documents found in the data directory")
        sys.exit(SYS_FAILURE)
    vectordb_dir = setup_db_dir("vectordb")
    # global settings for the indexing:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(SYS_FAILURE)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
    text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=64)
    Settings.node_parser = text_splitter
    # initialize FAISS vector store
    dims = 3072  # OpenAI text-embedding-3-large dimensions
    faiss_index = faiss.IndexFlatL2(dims)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # this is using higher level abstraction for the vector store
    # for more control, see: https://docs.llamaindex.ai/en/stable/examples/low_level/ingestion/
    logger.info("Creating index from documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True
    )
    print_chunk_info(index)
    # save the index to disk
    logger.info("Saving index to %s", vectordb_dir)
    index.storage_context.persist(persist_dir=str(vectordb_dir))
    logger.info("Vector database creation completed successfully")
    sys.exit(SYS_SUCCESS)


if __name__ == "__main__":
    main()
