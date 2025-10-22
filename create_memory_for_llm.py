import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

pdf_folder = "data"  # ضع ملفات PDF هنا
docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"📘 جاري تحميل الملف: {filename}")
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
texts = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

print("⚙️ جاري إنشاء قاعدة بيانات FAISS...")
db = FAISS.from_documents(texts, embedding_model)
db.save_local(DB_FAISS_PATH)
print("✅ تم إنشاء قاعدة بيانات FAISS بنجاح!")
