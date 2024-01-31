from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ChatGLM
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
 


prompt = ChatPromptTemplate.from_template("{foo}")

# template = """{question}"""
# prompt = PromptTemplate(template=template, input_variables=["question"])

# default endpoint_url for a local deployed ChatGLM api server
endpoint_url = "http://127.0.0.1:8000"

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[
        ["我从日本来中国旅游。", "欢迎问我任何问题。"]
    ],
    top_p=0.5,
    model_kwargs={"sample_model_args": False},
)



output_parser = StrOutputParser()

llm_chain = LLMChain(prompt=prompt, llm=llm,output_parser=output_parser)

question = "成都这座城市有什么特色？"

result = llm_chain.invoke(question)

print(result)