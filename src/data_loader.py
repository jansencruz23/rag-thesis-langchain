from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported documents from the data directory and convert to LangChain document structure
    Supported formats: PDF, TXT, WORD, CSV, EXCEL, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = [] 

    # PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f'[DEBUG] Loading PDF file: {pdf_file}')
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f'[DEBUG] Loaded {len(loaded)} documents from {pdf_file}')
            documents.extend(loaded)
        except Exception as e:
            print(f'[ERROR] Failed to load PDF file {pdf_file}: {e}')

    return documents