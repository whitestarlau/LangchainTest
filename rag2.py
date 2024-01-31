from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import ChatGLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.embeddings import SelfHostedHuggingFaceEmbeddings

# 文档切分
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

def spDoc(state_of_the_union):
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 128,  # 分块长度
        chunk_overlap  = 10,  # 重合的文本长度
        length_function = len,
    )
    texts = text_splitter.create_documents([state_of_the_union])
    # print(texts[0])

    # 这里metadatas用于区分不同的文档
    metadatas = [{"document": 1}, {"document": 2}]
    documents = text_splitter.create_documents([state_of_the_union, state_of_the_union], metadatas=metadatas)
    return documents

    # docs = text_splitter.split_documents(state_of_the_union)
    # return docs

# 加载txt文档
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('../books/', 
              glob="**/*.txt",  # 遍历txt文件
              show_progress=True,  # 显示进度
              use_multithreading=False,  # 使用多线程
              loader_cls=TextLoader,  # 使用加载数据的方式
              silent_errors=True,  # 遇到错误继续
              loader_kwargs=text_loader_kwargs)  # 可以使用字典传入参数

docs = loader.load()
# print("\n")
sDocss = []
for doc in docs:
    sDocs = spDoc(doc.page_content)
    sDocss.extend(sDocs)


import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# embedding 向量化
hfEmbedding = HuggingFaceEmbeddings(model_name='./text2vec-base-chinese')

vectorstore = FAISS.from_documents(
    sDocss, embedding=hfEmbedding
)
retriever = vectorstore.as_retriever()

# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# model
endpoint_url = "http://127.0.0.1:8000"

model = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    top_p=0.5,
    model_kwargs={"sample_model_args": False},
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


result = chain.invoke("Where did harrison work?")
print(result)