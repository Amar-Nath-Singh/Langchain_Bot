from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tqdm
import os
import shutil

persist_directory = "Book_db"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

book_name = "SKLearn_book.pdf"
book_dir = f"Books/{book_name}"
embedding_function = OllamaEmbeddings(model="nomic-embed-text", base_url="http://10.21.139.236:11434")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding_function)
loader = PyPDFLoader(book_dir, extract_images=False)
pages = loader.load_and_split()
docs = text_splitter.create_documents(texts=[page.page_content for page in pages], metadatas=[page.metadata for page in pages])
l = len(docs)

for id in tqdm.tqdm(range(l)):
    documents = [docs[id]]
    vectordb.add_documents(
            documents,
        ids=[f"{book_name}_{id}"]
    )
vectordb.persist()

print(vectordb._collection.count())

