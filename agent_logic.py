import os
import uuid
import json
import csv
import re
import base64
import requests
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

# .env 파일에서 OPENAI_API_KEY 등 환경 변수 불러오기
load_dotenv()


# ============================================================
# 1️⃣ 상태(State) 정의
# ============================================================
class State(TypedDict):
    """
    LangGraph 워크플로우에서 노드 간에 전달되는 상태(State)를 정의합니다.
    각 노드는 이 상태를 참조하거나 수정할 수 있습니다.
    """
    topic: str       # 사용자가 입력한 주제
    post_text: str   # GPT가 생성한 LinkedIn 포스트 텍스트
    image_path: str  # 생성된 이미지 파일 경로
    error: str       # 에러 메시지 (있을 경우)


# ============================================================
# 2️⃣ 유틸리티 함수 (보조 기능)
# ============================================================
def ensure_outputs():
    """
    출력 결과를 저장할 폴더(outputs, images, posts)를 자동 생성합니다.
    폴더가 이미 존재하면 아무 동작도 하지 않습니다.
    """
    for folder in ["outputs", "outputs/images", "outputs/posts"]:
        os.makedirs(folder, exist_ok=True)


def slugify(text: str) -> str:
    """
    파일 이름으로 사용 가능한 안전한 문자열(slug)을 만듭니다.
    예: "AI 트렌드 2025" → "ai-트렌드-2025"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text[:40]


def save_metadata(topic: str, post_text: str, image_path: str, error: str):
    """
    생성된 포스트의 메타데이터를 JSON과 CSV로 저장합니다.
    JSON: 개별 포스트별 세부 정보
    CSV : 전체 생성 내역 인덱스
    """
    ensure_outputs()
    timestamp = datetime.utcnow().isoformat() + "Z"
    uid = uuid.uuid4().hex[:8]
    safe_name = slugify(topic) or "untitled"
    base = os.path.join("outputs/posts", f"{safe_name}_{uid}")

    metadata = {
        "id": uid,
        "topic": topic,
        "post_text": post_text,
        "image_path": image_path,
        "error": error,
        "timestamp": timestamp,
    }

    # ✅ JSON 파일 저장
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ✅ CSV 인덱스 추가
    csv_path = os.path.join("outputs/posts", "posts_index.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metadata)

    return metadata


# ============================================================
# 3️⃣ 포스트 생성 노드 (GPT-4o-mini 사용)
# ============================================================
def post_generator(state: State) -> dict:
    """
    📄 GPT-4o-mini 모델을 사용하여 LinkedIn 포스트 텍스트를 생성합니다.
    - 전문적이면서도 자연스러운 톤으로 작성
    - 해시태그와 이모지를 포함
    """
    try:
        # 🧩 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 전문적인 LinkedIn 콘텐츠 작가입니다. "
                       "짧지만 영감을 주는 문체로 작성하며, 적절한 해시태그와 이모지를 포함합니다."),
            ("human", "다음 주제에 대해 LinkedIn 포스트를 작성해 주세요:\n{topic}")
        ])

        # ⚙️ LLM 초기화
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

        # 🪄 프롬프트 → 모델 체인 구성
        chain = prompt | llm

        # 🧠 포스트 생성
        response = chain.invoke({"topic": state["topic"]})
        return {"post_text": response.content.strip(), "error": ""}

    except Exception as e:
        # ❌ 예외 처리
        return {"post_text": "", "error": f"포스트 생성 오류: {e}"}


# ============================================================
# 4️⃣ 이미지 생성 노드 (OpenAI DALL·E 2)
# ============================================================
def image_generator(state: State) -> dict:
    """
    🎨 OpenAI Images API를 사용하여 포스트에 어울리는 이미지를 생성합니다.
    GPT-4o가 작성한 텍스트 내용을 바탕으로 프롬프트를 생성합니다.
    """
    try:
        # 🔑 API 클라이언트 초기화
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # 📷 프롬프트 작성
        prompt = f"Generate a visually appealing image suitable for a LinkedIn post about: {state['topic']}"

        # 🧩 이미지 생성 요청
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024"
        )

        # 📁 이미지 저장 경로 지정
        data = response.data[0]
        image_path = os.path.join("outputs/images", f"image_{uuid.uuid4().hex[:8]}.png")

        # 🔄 base64 또는 URL 기반 데이터 저장
        if getattr(data, "b64_json", None):
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(data.b64_json))
        elif getattr(data, "url", None):
            img_data = requests.get(data.url).content
            with open(image_path, "wb") as f:
                f.write(img_data)
        else:
            return {"image_path": "", "error": "이미지 생성 실패: 응답 데이터 없음"}

        return {"image_path": image_path, "error": ""}

    except Exception as e:
        return {"image_path": "", "error": f"이미지 생성 오류: {e}"}


# ============================================================
# 5️⃣ LangGraph 그래프 구성
# ============================================================
def build_graph():
    """
    LangGraph를 이용해 전체 워크플로우를 구성합니다.
    [START] → [포스트 생성] → [이미지 생성] → [END]
    """
    graph = StateGraph(State)
    graph.add_node("post", post_generator)
    graph.add_node("image", image_generator)
    graph.add_edge(START, "post")
    graph.add_edge("post", "image")
    graph.add_edge("image", END)
    return graph.compile()


# ============================================================
# 6️⃣ 전체 워크플로우 실행 로직
# ============================================================
def run_workflow(initial_state: dict, progress_callback=None) -> dict:
    """
    💡 전체 프로세스를 순차적으로 실행합니다.
    1️⃣ 포스트 작성 → 2️⃣ 이미지 생성 → 3️⃣ 메타데이터 저장
    """
    ensure_outputs()
    state: State = State(topic=initial_state.get("topic", ""),
                         post_text="", image_path="", error="")

    try:
        # 1️⃣ 포스트 생성
        if progress_callback:
            progress_callback(10, "포스트 생성 중...")
        post_out = post_generator(state)
        state["post_text"] = post_out["post_text"]

        if post_out["error"]:
            state["error"] = post_out["error"]
            save_metadata(state["topic"], state["post_text"], "", state["error"])
            return state

        # 2️⃣ 이미지 생성
        if progress_callback:
            progress_callback(50, "이미지 생성 중...")
        img_out = image_generator(state)
        state["image_path"] = img_out["image_path"]
        state["error"] = img_out["error"]

        # 3️⃣ 결과 저장
        save_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])

        if progress_callback:
            progress_callback(100, "워크플로우 완료.")
        return state

    except Exception as e:
        state["error"] = f"워크플로우 오류: {e}"
        save_metadata(state["topic"], state["post_text"], state["image_path"], state["error"])
        return state


# ============================================================
# 7️⃣ Mermaid 다이어그램 (시각화용)
# ============================================================
def mermaid_diagram() -> str:
    """
    전체 워크플로우를 Mermaid 다이어그램 형태로 반환합니다.
    Streamlit에서 시각화할 때 사용됩니다.
    """
    return """```mermaid
flowchart LR
    START --> POST[포스트 생성 ✍️]
    POST --> IMAGE[이미지 생성 🎨]
    IMAGE --> END
```"""



