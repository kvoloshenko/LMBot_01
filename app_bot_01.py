import re
import codecs
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import openai


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

def get_message_content(topic, index_db, k_num):
    # Поиск релевантных отрезков из базы знаний
    docs = index_db.similarity_search(topic, k = k_num)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### Document excerpt №{i+1}####\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    print(f"message_content={message_content}")
    return message_content

def answer_index(system, topic, message_content, temp):
    openai.api_type = "open_ai"
    openai.api_base = "http://localhost:1234/v1"
    openai.api_key = "Whatever"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Here is the document with information to respond to the client: {message_content}\n\n Here is the client's question: \n{topic}"}
    ]


    completion = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages,
        temperature=temp
    )


    answer = completion.choices[0].message.content

    return answer  # возвращает ответ

if __name__ == '__main__':
    database = load_text('OrderDeliciousBot_KnowledgeBase_01.txt')
    index_db = create_index_db(database)
    topic ="В каком ресторане есть Labneh? Опиши это блюдо"
    message_content = get_message_content(topic, index_db, k_num=2)
    # Инструкция для LLM, которая будет подаваться в system
    system = load_text('OrderDeliciousBot_Prompt_01.txt')
    ans = answer_index(system, topic, message_content, temp=0.2) # получите ответ модели
    print(ans)

