"""
rag.py - implement RAG using llama_index and OpenAI

reads questions from 'questions.txt', uses a FAISS vector database to retrieve
relevant context information, and then uses that context to answer each question using
OpenAI's LLM.
"""
import os
import logging
import sys
from pathlib import Path
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatMessage

SYS_SUCCESS = 0
SYS_FAILURE = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vector_db():
    """ load the FAISS vector database from disk """
    vectordb_dir = Path("vectordb")
    if not vectordb_dir.exists():
        logger.error("Vector database directory %s does not exist", vectordb_dir)
        logger.error("Please run index.py first to create the vector database")
        return None
    logger.info("Loading vector database from %s", vectordb_dir)
    storage_context = StorageContext.from_defaults(persist_dir=str(vectordb_dir))
    index = load_index_from_storage(storage_context)
    logger.info("Vector database loaded successfully")
    return index


def read_questions():
    """ assumes questions in questions.txt, with each question on a separate line """
    question_file = Path("questions.txt")
    if not question_file.exists():
        logger.error("Question file %s does not exist", question_file)
        return None
    with open(question_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    if not questions:
        logger.error("No questions found in the file")
        return None
    logger.info("Loaded %d questions from %s", len(questions), question_file)
    return questions


def retrieve_context(index, question, num_chunks=4):
    """ Retrieve context information for the given question using the vector store """
    if not index or not question:
        return None
    logger.info("Question: %s", question)
    # retrieve relevant chunks using the vector store index
    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=num_chunks,
    )
    retrieved_nodes = retriever.retrieve(question)
    return retrieved_nodes


def answer_question(index, question, num_chunks=4):
    """ use RAG to answer the question, manually builds the prompt with retrieved chunks """
    retrieved_nodes = retrieve_context(index, question, num_chunks)
    # Build context text from retrieved chunks
    context_text = ""
    for i, node in enumerate(retrieved_nodes):
        context_text += f"Chunk {i+1}:\n{node.text}\n\n"
    system_message = """You are a helpful assistant that answers questions based on the
     provided context. Use only the information from the context to answer the question.
    If the context doesn't contain the answer, say "I don't have enough information to answer this question."
    Be concise and clear in your answer."""
    user_message = f"""Context information:
    {context_text}
    Question: {question}
    Please answer the question based on the context provided above."""
    messages=[
      ChatMessage(role="system", content = system_message),
      ChatMessage(role="user", content = user_message)
    ]
    llm_response = Settings.llm.chat(messages)
    return llm_response.message.content, retrieved_nodes


def print_answer_and_sources(num, question, response, source_nodes):
    """ assuming all arguments are valid """
    print("\n" + "="*80)
    print(f"QUESTION {num+1}:")
    print(question)
    print("="*80)
    print("\nRETRIEVED CONTEXT:")
    print("-"*80)
    for j, node in enumerate(source_nodes):
        print(f"Chunk {j+1} (Score: {node.score:.4f}):")
        print(f"\n{node.text}\n")
        print("-"*80)
    print("\nANSWER:")
    print("="*80)
    print(response)
    print("="*80)
    print("\n\n" + "#"*100 + "\n\n")


def main():
    """ main function to run the RAG flow for each question """
    # assuming the OpenAI API key is set in the environment variable OPENAI_API_KEY:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(SYS_FAILURE)
    # global settings to be later used in the index and LLM:
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
    # assumes index is already created and stored in vectordb directory:
    index = load_vector_db()
    if not index:
        sys.exit(SYS_FAILURE)
    # read the questions from questions.txt file:
    questions = read_questions()
    if not questions:
        sys.exit(SYS_FAILURE)
    # for each question:
    for i, question in enumerate(questions):
        logger.info("Processing question %d of %d", i+1, len(questions))
        response, source_nodes = answer_question(index, question)
        print_answer_and_sources(i, question, response, source_nodes)
    sys.exit(SYS_SUCCESS)

if __name__ == "__main__":
    main()
