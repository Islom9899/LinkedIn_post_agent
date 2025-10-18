import os
import uuid
import json
import csv
import re
from datetime import datetime
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
# LangChain import compatibility across versions
try:
    # Newer LangChain versions (>=0.1) expose prompts via langchain_core
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    # Fallback for older LangChain versions
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # MessagesPlaceholder import qilindi
from langchain_openai import ChatOpenAI
from langchain.tools import tool
# Agent creation compatibility across LangChain versions
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
except Exception:
    # Newer versions renamed to create_openai_tools_agent
    from langchain.agents import AgentExecutor, create_openai_tools_agent as create_openai_functions_agent
from openai import OpenAI
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1️⃣ 상태(State) 정의
# ============================================================
class State(TypedDict):
    """
    LangGraph 워크플로우에서 공유되는 상태 구조체 정의.
    """
    topic: str       # 사용자가 입력한 주제
    post_text: str   # 생성된 LinkedIn 포스트 텍스트
    image_path: str  # 생성된 이미지 파일 경로
    error: str       # 에러 메시지

# GPT-4 모델 초기화 (Siz ilgari buni tanlagansiz)
llm = ChatOpenAI(model="gpt-4", temperature=0.5)

# ============================================================
# 2️⃣ 유틸리티 함수
# ============================================================
def slugify(text: str, maxlen: int = 40) -> str:
    """
    파일 이름 등에 사용할 수 있도록 텍스트를 안전한 슬러그(slug) 형태로 변환.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text[:maxlen] if len(text) > maxlen else text

def ensure_outputs():
    """
    출력 폴더(outputs) 및 하위 폴더(images, posts)를 자동 생성.
    """
    out_dirs = ["outputs", "outputs/images", "outputs/posts"]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

def save_post_metadata(topic: str, post_text: str, image_path: str, error: str):
    """
    생성된 포스트 및 관련 메타데이터를 JSON과 CSV 파일로 저장.
    """
    ensure_outputs()
    timestamp = datetime.utcnow().isoformat() + "Z"
    safe_topic = slugify(topic) or "untitled"
    uid = uuid.uuid4().hex[:8]
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
        "json_path": json_path
    }

    # JSON 파일 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # CSV 파일에 추가
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metadata)

    return metadata

# ============================================================
# 3️⃣ 도구(툴) 정의
# ============================================================
@tool("image_generator")
def generate_image_tool(prompt: str) -> str:
    """
    영어 프롬프트를 입력받아 OpenAI DALL·E API를 통해 이미지를 생성하고
    로컬 파일로 저장한 뒤 그 경로를 반환.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return "Image generation failed: OPENAI_API_KEY is not set."

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )

        # 응답에서 이미지 URL 또는 base64 데이터 추출
        image_url, b64 = None, None
        if hasattr(response, "data") and len(response.data) > 0:
            item = response.data[0]
            # SDK 응답 형태에 따라 추출
            image_url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None)
            b64 = getattr(item, "b64_json", None) or (item.get("b64_json") if isinstance(item, dict) else None)

        ensure_outputs()
        out_dir = "outputs/images"

        if image_url:
            image_name = f"image_{uuid.uuid4().hex[:8]}.png"
            image_path = os.path.join(out_dir, image_name)
            img_data = requests.get(image_url).content
            with open(image_path, "wb") as handler:
                handler.write(img_data)
            return image_path
        elif b64:
            import base64
            image_name = f"image_{uuid.uuid4().hex[:8]}.png"
            image_path = os.path.join(out_dir, image_name)
            with open(image_path, "wb") as handler:
                handler.write(base64.b64decode(b64))
            return image_path
        else:
            return "Image generation failed: unexpected API response."

    except Exception as e:
        return f"Image generation failed: {e}"

# ============================================================
# 4️⃣ LangGraph 노드 정의
# ============================================================
post_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 기술 분야의 전문 LinkedIn 콘텐츠 작가입니다. "
               "포스트는 전문적이면서도 친근한 톤으로 작성하고, "
               "적절한 이모지와 해시태그를 포함하며 가독성이 좋게 구성해야 합니다."),
    ("human", "다음 주제에 대한 LinkedIn 포스트를 작성해 주세요:\n{topic}")
])
post_chain = post_prompt | llm

def post_node(state: State) -> dict:
    """
    GPT를 사용하여 포스트 본문을 생성하는 노드.
    """
    try:
        response = post_chain.invoke({"topic": state["topic"]})
        return {"post_text": response.content, "error": ""}
    except Exception as e:
        return {"post_text": "", "error": f"포스트 생성 오류: {e}"}


def image_gen_node(state: State) -> dict:
    """
    생성된 포스트 내용을 바탕으로 이미지 프롬프트를 만들고,
    'image_generator' 툴을 호출하여 썸네일 이미지를 생성.
    """
    try:
        tools = [generate_image_tool]

        # Agent prompt tuzatildi: Final Answer talabi qo'shildi va MessagesPlaceholder import qilindi
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 시각적으로 매력적인 LinkedIn 썸네일을 만들기 위한 이미지 프롬프트 전문가입니다. "
                       "'image_generator' 도구만 사용할 수 있습니다. "
                       "이미지 생성이 완료되면, 다른 말 없이 오직 'Final Answer: [이미지 파일 경로]' 형태로만 응답하세요."),
            ("human", "아래 포스트 내용을 기반으로 이미지를 생성하세요:\n---\n{input}\n---"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=agent_prompt
        )

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Xato tuzatildi: sinxron .invoke() ishlatildi
        resp = agent_executor.invoke({
            "input": state["post_text"]
        })

        # AgentExecutor javobidan 'output' maydonini olish
        final_output = ""
        if isinstance(resp, dict):
            # Agent izohini o'chirish uchun Final Answer: formatini izlash
            output_content = resp.get("output", "") or resp.get("result", "") or ""
            
            # Final Answer: ni ajratib olish
            match = re.search(r"Final Answer:\s*(.+)", output_content, re.DOTALL)
            final_output = match.group(1).strip() if match else output_content.strip()
        else:
            final_output = str(resp).strip()

        # Agar tool fayl yo'li qaytargan bo'lsa, uni images papkasiga nom bilan ko'chirish
        if isinstance(final_output, str) and ("outputs/images" in final_output or final_output.endswith(".png")):
            ensure_outputs()
            src = final_output
            safe_topic = slugify(state["topic"])
            new_name = f"{safe_topic}_{uuid.uuid4().hex[:6]}.png" if safe_topic else f"image_{uuid.uuid4().hex[:8]}.png"
            dst = os.path.join("outputs", "images", new_name)
            try:
                # Fayl nomini o'zgartirish
                os.replace(src, dst)
                return {"image_path": dst, "error": ""}
            except Exception:
                # Agar os.replace ishlamasa (masalan, turli xil qurilmalarda), asl yo'lni qaytarish
                return {"image_path": src, "error": ""}
        else:
            # agent yoki tool xatolik matnini qaytarishi mumkin
            return {"image_path": "", "error": final_output or "이미지 생성 실패"}

    except Exception as e:
        return {"image_path": "", "error": f"이미지 노드 오류: {e}"}

# ============================================================
# 5️⃣ 그래프 빌더 및 실행 로직
# ============================================================
def build_graph():
    """
    LangGraph 상태 그래프를 구성하고 컴파일.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("post", post_node)
    graph_builder.add_node("image_gen", image_gen_node)
    graph_builder.add_edge(START, "post")
    graph_builder.add_edge("post", "image_gen")
    graph_builder.add_edge("image_gen", END)
    return graph_builder.compile()

def run_workflow(initial_state: dict, progress_callback=None) -> dict:
    """
    전체 워크플로우(post → image)를 순차적으로 실행하고
    단계별 진행 상황을 콜백(progress_callback)으로 전달.
    """
    state: State = State(
        topic=initial_state.get("topic", ""),
        post_text=initial_state.get("post_text", ""),
        image_path="",
        error=""
    )

    try:
        if progress_callback:
            progress_callback(10, "포스트 생성 중...")
        
        # 1. Postni yaratish
        post_out = post_node(state)
        state["post_text"] = post_out.get("post_text", "")
        if post_out.get("error"):
            state["error"] = post_out["error"]
            save_post_metadata(state["topic"], state["post_text"], "", state["error"])
            return state

        if progress_callback:
            progress_callback(50, "포스트 생성 완료. 이미지 생성 중...")

        # 2. Rasmni yaratish
        img_out = image_gen_node(state)
        state["image_path"] = img_out.get("image_path", "")
        state["error"] = img_out.get("error", "")

        if progress_callback:
            progress_callback(90, "이미지 생성 완료.")
        # 3. Metama'lumotlarni saqlash
        save_post_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])

        if progress_callback:
            progress_callback(100, "워크플로우 완료.")
        return state

    except Exception as e:
        state["error"] = f"예상치 못한 오류 발생: {e}"
        save_post_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])
        if progress_callback:
            progress_callback(100, "워크플로우 실패.")
        return state

# ============================================================
# 6️⃣ Mermaid 다이어그램 (시각화용)
# ============================================================
def mermaid_diagram() -> str:
    """
    워크플로우 구조를 시각적으로 보여주는 Mermaid 다이어그램 반환.
    """
    diagram = """```mermaid
flowchart LR
    START --> Post[포스트 생성 노드\\n(GPT-4)]
    Post --> Image[이미지 생성 노드\\n(DALL·E-3)]
    Image --> END
```"""
    return diagram
