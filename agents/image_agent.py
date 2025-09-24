from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import re
from langgraph.prebuilt import create_react_agent


async def setup_image_agent():
    client = MultiServerMCPClient({
        "image": {
            "command": "python",
            "args": ["mcp_server/image_server.py"],
            "transport": "stdio"
        }
    })
    
    tools = await client.get_tools()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5)
    return create_react_agent(llm, tools)

async def image_node(state):
    image_agent = await setup_image_agent()
    response = await image_agent.ainvoke({
        "messages": [
            {"role": "user",
                "content": f"Generate a thumbnail image for this LinkedIn post:\n{state['post_text']}"}
        ]
    })
    # 텍스트에서 파일만 추출하기
    content = response['messages'][-1].content
    match = re.search(r'outputs[\\/][\w\-.]+', content)
    if match:
        return {"image_path": match.group(0)}
    else:
        return {"image_path": ""}