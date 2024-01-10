import logging
import os
import click
import torch
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings 

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

# Function to log the ingest process
def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

# function to load a single pdf document
def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
       
       loader_class = UnstructuredFileLoader
       if loader_class:
           file_log(file_path + ' loaded.')
           loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()[0]
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None 

def load_documents(source_dir: str):
    '''
    Loads all docs from source directory
    '''
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
    files = os.listdir(source_dir)
    return([load_single_document(f'{source_dir}/{file_path}') for file_path in files])


# Splitting docs into 
def split_documents(documents: list[Document]) -> tuple[list[Document]]:
    '''
    Splits pdfs for Text Splitter
    '''
    # Splits documents for correct Text Splitter
    text_docs = []
    for doc in documents:
        if doc is not None:
        #    check file extention
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == '.pdf':
              text_docs.append(doc)
    return text_docs

@click.command()
@click.option(
    "--device_type",
    "-d",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "mps",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)


def main(device_type):
    '''
    Load documents and split in chunks
    '''

    # SOURCE_DIRECTORY='/Users/tomc/git/localGPT/SOURCE_DOCUMENTS'
    # EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")

    documents = load_documents(SOURCE_DIRECTORY)
    text_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(text_documents)
  
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
