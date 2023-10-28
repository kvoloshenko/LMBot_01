import codecs
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


# Функция создания индексной базы знаний
def create_index_db(database):
  source_chunks = []
  splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)

  for chunk in splitter.split_text(database):
      source_chunks.append(Document(page_content=chunk, metadata={}))

  # Initializing the embedding model
  # embeddings = OpenAIEmbeddings()

  model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  # model_kwargs = {'device': 'cpu'}
  model_kwargs = {'device': 'cuda'}
  embeddings = HuggingFaceEmbeddings(
      model_name=model_id,
      model_kwargs=model_kwargs
  )

  # Create an index db from separated text fragments
  db = FAISS.from_documents(source_chunks, embeddings)
  return db

def load_text(file_path):
    # Открытие файла для чтения
    with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
        # Чтение содержимого файла
        content = input_file.read()
    return content


if __name__ == '__main__':
    database = load_text('OrderDeliciousBot_KnowledgeBase_01.txt')
    index_db = create_index_db(database)

    topic ="У кого есть пица?"
    docs = index_db.similarity_search(topic, k = 5)
    i=1
    for doc in docs:
        print(f"{i} Content: {doc.page_content}")
        i+=1