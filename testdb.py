from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ChatGLM

db = SQLDatabase.from_uri("sqlite:///./test.db")

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000"

model = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    top_p=0.5,
    model_kwargs={"sample_model_args": False},
)

output_parser = StrOutputParser()

llm_chain = LLMChain(prompt=prompt, llm=model,output_parser=output_parser)


sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)


# sql_response.invoke({"question": "How many student are there?"})

full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | model
)


result = full_chain.invoke({"question": "How many api_call_record are there?"})

print(result)