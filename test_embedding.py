from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

persist_directory = "LangChain_Bot/Book_db"
embedding_function = OllamaEmbeddings(model="nomic-embed-text", base_url="http://10.21.139.236:11434")

db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
for x in db.similarity_search_with_score('When does LLMs went Rogue in history', k = 10):
    print(x)