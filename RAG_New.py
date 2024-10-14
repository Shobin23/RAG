from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma



from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA  # or other chain implementations
from langchain_community.document_loaders import TextLoader

from openai import OpenAI

import os


client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)



# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))

print(texts[3])


persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)


vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("How much money did Pando raise?")
print(len(docs))
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

#from openai import OpenAI
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain.chains import RetrievalQA # Import RetrievalQA



qa_chain = RetrievalQA.from_chain_type(llm=LangChainOpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
print(process_llm_response(llm_response))
