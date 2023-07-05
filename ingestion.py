import urllib
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone


load_dotenv(r"C:\Users\pavni\Documents\WebD\doc-helper\.vscode\.env")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path=r"langchain-docs\api.python.langchain.com\en\latest\langchain\docstore")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
