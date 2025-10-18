# ============================================================
# agent_logic.py — 실제 OpenAI API를 사용한 AI LinkedIn 포스트 생성기
# ============================================================
"""
이 모듈은 LangGraph, LangChain, OpenAI API를 사용하여
사용자가 입력한 주제(topic)에 맞는 LinkedIn 포스트 텍스트와 이미지를 생성합니다.

Streamlit Cloud에서 실행 가능하며,
환경 변수 OPENAI_API_KEY 가 반드시 설정되어 있어야 합니다.
"""

import os
import uuid
import json
import csv
import re
import base64
import requests
from datetime import datetime
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph 및 LangChain 관련 모듈
from langgraph.graph import StateGraph
try:
    from langgraph.constants import START, END
except ImportError:
    START, END = "__start__", "__end__"

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from openai import OpenAI

load_dotenv()

# ============================================================
# 1️⃣ 상태(State) 정의
# ============================================================
class State(TypedDict):
    """LangGraph 워크플로우에서 공유되는 상태(State) 구조체"""
    topic: str
    post_text: str
    image_path: str
    error: str


# OpenAI LLM 초기화 (GPT-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)


# ============================================================
# 2️⃣ 유틸리티 함수
# ============================================================
def slugify(text: str, maxlen: int = 40) -> str:
    """파일 이름에 안전한 문자열 형태로 변환"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text[:maxlen]


def ensure_outputs():
    """출력 폴더(outputs 및 하위 폴더)를 자동 생성"""
    for folder in ["outputs", "outputs/images", "outputs/posts"]:
        os.makedirs(folder, exist_ok=True)


def save_post_metadata(topic: str, post_text: str, image_path: str, error: str):
    """생성된 포스트 및 관련 메타데이터를 JSON과 CSV로 저장"""
    ensure_outputs()
    timestamp = datetime.utcnow().isoformat() + "Z"
    uid = uuid.uuid4().hex[:8]
    safe_topic = slugify(topic) or "untitled"
    base_name = f"{safe_topic}_{uid}"
    json_path = os.path.join("outputs", "posts", f"{base_name}.json")
    csv_path = os.path.join("outputs", "posts", "posts_index.csv")

    metadata = {
        "id": uid,
        "topic": topic,
        "post_text": post_text,
        "image_path": image_path,
        "error": error,
        "timestamp": timestamp,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metadata)

    return metadata


# ============================================================
# 3️⃣ 이미지 생성 툴 정의 (실제 OpenAI API 호출)
# ============================================================
@tool("image_generator")
def generate_image_tool(prompt: str) -> str:
    """
    OpenAI DALL·E 3 모델을 사용해 이미지를 생성하고 로컬 파일로 저장.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "❌ OPENAI_API_KEY가 설정되어 있지 않습니다."

    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )

        data = response.data[0]
        image_url = getattr(data, "url", None)
        b64 = getattr(data, "b64_json", None)

        ensure_outputs()
        image_name = f"image_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join("outputs/images", image_name)

        if image_url:
            img_data = requests.get(image_url).content
            with open(image_path, "wb") as f:
                f.write(img_data)
            return image_path
        elif b64:
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(b64))
            return image_path
        else:
            return "❌ 이미지 생성 실패: 응답이 비정상적입니다."

    except Exception as e:
        return f"❌ 이미지 생성 오류: {e}"


# ============================================================
# 4️⃣ LangGraph 노드 정의
# ============================================================
post_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문 LinkedIn 콘텐츠 작가입니다. "
               "글은 전문적이면서 친근한 톤으로 작성하며, 이모지와 해시태그를 포함해야 합니다."),
    ("human", "다음 주제에 대해 LinkedIn 포스트를 작성해 주세요:\n{topic}")
])
post_chain = post_prompt | llm


def post_node(state: State) -> dict:
    """GPT-4o를 사용해 LinkedIn 포스트 텍스트를 생성하는 노드"""
    try:
        response = post_chain.invoke({"topic": state["topic"]})
        return {"post_text": response.content.strip(), "error": ""}
    except Exception as e:
        return {"post_text": "", "error": f"포스트 생성 오류: {e}"}


def image_gen_node(state: State) -> dict:
    """생성된 포스트를 기반으로 이미지를 생성하는 노드"""
    try:
        tools = [generate_image_tool]
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 시각적으로 매력적인 LinkedIn 썸네일 이미지를 만드는 전문가입니다. "
                       "'image_generator' 도구만 사용할 수 있습니다. "
                       "이미지 생성이 완료되면 오직 'Final Answer: [이미지 경로]' 형태로만 응답하세요."),
            ("human", "다음 포스트 내용을 기반으로 이미지를 생성하세요:\n{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=agent_prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        resp = executor.invoke({"input": state["post_text"]})

        output_text = resp.get("output", "") if isinstance(resp, dict) else str(resp)
        match = re.search(r"Final Answer:\s*(.+)", output_text, re.DOTALL)
        final_output = match.group(1).strip() if match else output_text.strip()

        if "outputs/images" in final_output or final_output.endswith(".png"):
            return {"image_path": final_output, "error": ""}
        else:
            return {"image_path": "", "error": final_output or "이미지 생성 실패"}

    except Exception as e:
        return {"image_path": "", "error": f"이미지 노드 오류: {e}"}


# ============================================================
# 5️⃣ 그래프 구성 및 실행
# ============================================================
def build_graph():
    """LangGraph 그래프 빌드"""
    graph = StateGraph(State)
    graph.add_node("post", post_node)
    graph.add_node("image_gen", image_gen_node)
    graph.add_edge(START, "post")
    graph.add_edge("post", "image_gen")
    graph.add_edge("image_gen", END)
    return graph.compile()


def run_workflow(initial_state: dict, progress_callback=None) -> dict:
    """전체 워크플로우를 순차적으로 실행"""
    state: State = State(topic=initial_state.get("topic", ""), post_text="", image_path="", error="")

    try:
        if progress_callback:
            progress_callback(10, "포스트 생성 중...")

        post_out = post_node(state)
        state["post_text"] = post_out.get("post_text", "")
        if post_out.get("error"):
            state["error"] = post_out["error"]
            save_post_metadata(state["topic"], state["post_text"], "", state["error"])
            return state

        if progress_callback:
            progress_callback(50, "이미지 생성 중...")

        img_out = image_gen_node(state)
        state["image_path"] = img_out.get("image_path", "")
        state["error"] = img_out.get("error", "")

        save_post_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])
        if progress_callback:
            progress_callback(100, "완료.")
        return state

    except Exception as e:
        state["error"] = f"워크플로우 오류: {e}"
        save_post_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])
        if progress_callback:
            progress_callback(100, "실패.")
        return state


# ============================================================
# 6️⃣ Mermaid 다이어그램 (시각화용)
# ============================================================
def mermaid_diagram() -> str:
    """LangGraph의 워크플로우 구조를 Mermaid 코드로 반환"""
    return """```mermaid
flowchart LR
    START --> Post[포스트 생성 노드\\n(GPT-4o)]
    Post --> Image[이미지 생성 노드\\n(DALL·E-3)]
    Image --> END
```"""
