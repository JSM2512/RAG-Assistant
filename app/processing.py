import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

REPORTS_DIR = "D:/aviation_dataset/docs"
FAISS_INDEX_PATH = "aviation_reports_index"
BATCH_SIZE = 10

def process_pdf_batch(pdf_files, existing_db=None):
    docs=[]
    for filename in pdf_files:
        loader = PyMuPDFLoader(os.path.join(REPORTS_DIR, filename))
        docs_from_file = loader.load()

        for doc in docs_from_file:
            doc.metadata["source_file"] = filename
            docs.append(doc)
    
    embeddings = OpenAIEmbeddings()
    if existing_db is None:
        db = FAISS.from_documents(docs, embedding=embeddings)
    else:
        db = existing_db
        db.add_documents(docs)
    db.save_local(FAISS_INDEX_PATH)
    return db

def main():
    all_pdfs = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")]
    existing_db = None
    for i in range(0, len(all_pdfs), BATCH_SIZE):
        batch_files = all_pdfs[i:i+BATCH_SIZE]
        print(f"Processing Batch {i//BATCH_SIZE+1} ({len(batch_files)} files)...")
        existing_db = process_pdf_batch(batch_files, existing_db)
        print(f"Batch {i//BATCH_SIZE + 1} processed and indexed.")
    print("All batches processed. FAISS index saved at:", FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()