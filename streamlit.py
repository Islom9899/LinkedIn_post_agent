import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

from graph_builder import build_graph  # 그래프 빌더 모듈

# 페이지 설정
st.set_page_config(page_title="AI LinkedIn 포스트 생성기", layout="wide")
st.title("🚀 AI LinkedIn 포스트 생성기")
st.markdown("주제를 입력하면 AI가 자동으로 글과 이미지를 만들어 드립니다.")

# ===== Streamlit Secrets 불러오기 =====
HF_TOKEN = st.secrets["HF_TOKEN"]  # HuggingFace API 키
OUT_DIR = st.secrets.get("OUT_DIR", "outputs")  # 이미지 저장 폴더
# =====================================

# 사용자 입력 받기
topic = st.text_input("포스트 주제를 입력하세요", placeholder="예: 헬스케어 분야에서의 AI 활용")

# 버튼 클릭 시 실행
if st.button("포스트 생성하기"):
    if not topic.strip():
        st.warning("⚠️ 주제를 입력해 주세요.")  # 입력값이 없으면 경고
    else:
        with st.spinner("⏳ 포스트를 생성 중입니다..."):  # 로딩 스피너 표시
            
            # 비동기 파이프라인 함수 정의
            async def run_pipeline():
                graph = build_graph()  # 그래프 생성
                initial_state = {
                    "topic": topic,  # 입력 주제
                    "post_text": f"LinkedIn 포스트 주제: {topic}",  # 초기 텍스트
                    "image_path": ""  # 이미지 경로 초기화
                }
                # MCP graph 실행
                return await graph.ainvoke(initial_state)

            # 🔹 ThreadPoolExecutor 사용: Streamlit과 AnyIO TaskGroup 충돌 방지
            try:
                # run_pipeline()을 별도 쓰레드에서 실행
                def run_pipeline_threadsafe():
                    return asyncio.run(run_pipeline())

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(run_pipeline_threadsafe)
                    final_state = future.result()  # 결과 가져오기

                # 결과 출력
                st.subheader("✍️ 생성된 LinkedIn 포스트")
                st.write(final_state["post_text"])

                if final_state.get("image_path"):
                    st.subheader("🖼️ 생성된 이미지")
                    st.image(final_state["image_path"], caption="AI가 생성한 이미지", use_column_width=True)

            except Exception as e:
                st.error(f"⚠️ 포스트 생성 중 오류 발생: {e}")  # 오류 발생 시 표시
