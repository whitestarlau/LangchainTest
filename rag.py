from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.llms import ChatGLM
from langchain_community.embeddings import HuggingFaceEmbeddings


# embedding 向量化
hfEmbedding = HuggingFaceEmbeddings(model_name='./text2vec-base-chinese')

vectorstore = FAISS.from_texts(
    ["John worked at google\nHarrison worked at kensho"], embedding=hfEmbedding
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