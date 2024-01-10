# ofset_gpt
I have built a simple chatbot with RAG capabilities to provide the foundation model (Llama 2 7B) with ability to contextualize the input queries. 

Basic components of the system:
- LangChain(https://github.com/hwchase17/langchain)
- HuggingFace LLMs(https://huggingface.co/models)
- InstructorEmbeddings(https://instructor-embedding.github.io/)
- LLAMACPP(https://github.com/abetlen/llama-cpp-python)
- ChromaDB(https://www.trychroma.com/)

Foundation Model: llama-2-7b-chat.Q4_K_M.gguf (7 bln parameters)
Vector embedding model: hkunlp/instructor-large

- Data Privacy: Your data remains on your computer, ensuring 100% security.
- Diverse Embeddings**: Choose from a range of open-source embeddings.
- Reuse Your LLM: Once downloaded, reuse your LLM without the need for repeated downloads.
- Chat History: Remembers your previous conversations (in a session use).
- GPU, CPU & MPS Support: by default, it checks GPU availability if not found it defaults to CPU. Apple silicon support can be enabeld by adding - - device_type “mps” CLI argument.

- The system consists of the following components:

-`ingest_pdf.py` uses `LangChain` tools to parse the document and create embeddings locally 
using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store

-`run_simpleGPT.py `run_localGPT.py` uses a local LLM to understand questions and create answers. The context for the answers
 is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

-`prompt_template_simple.py` -  includes the definition of the system prompt used to query LLM

-`constants.py` - includes variable definitions utilized by both ingest_pdf and run_simpleGPT.py

Steps to run the system:

1. Copy the pdf files to ./SOURCE_DOCUMENTS
2. run `python ingest_pdf.py` (add -- device_type mps for M1/M2 apple silicon)
3. run `python run_simpleGPT.py` (add -- device_type mps for M1/M2 apple silicon)
4. Query the system from the command line. Type `exit` to close the application 
