from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)

post_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional LinkedIn content creator specializing in technology."),
    ("human", "Write a LinkedIn post about this topic: \n {topic}")
])
post_chain = post_prompt | llm

def post_node(state):
    response = post_chain.invoke({"topic": state["topic"]})
    return {"post_text": response.content}