import streamlit as st
import asyncio
from graph_builder import build_graph

st.set_page_config(page_title="AI LinkedIn 포스트 생성기", layout="wide")

st.title("🚀 AI LinkedIn 포스트 생성기")
st.markdown("주제를 입력하면 AI가 자동으로 글과 이미지를 만들어 드립니다.")

# 사용자 입력
topic = st.text_input("포스트 주제를 입력하세요", placeholder="예: 헬스케어 분야에서의 AI 활용")

# 버튼 클릭 시 실행
if st.button("포스트 생성하기"):
    if not topic.strip():
        st.warning("⚠️ 주제를 입력해 주세요.")
    else:
        with st.spinner("⏳ 포스트를 생성 중입니다..."):
            async def run_pipeline():
                graph = build_graph()
                initial_state = {
                    "topic": topic,
                    "post_text": f"LinkedIn 포스트 주제: {topic}",
                    "image_path": ""
                }
                return await graph.ainvoke(initial_state)
            
            final_state = asyncio.run(run_pipeline())
            
            # 결과 출력
            st.subheader("✍️ 생성된 LinkedIn 포스트")
            st.write(final_state["post_text"])

            if final_state["image_path"]:
                st.subheader("🖼️ 생성된 이미지")
                st.image(final_state["image_path"], caption="AI가 생성한 이미지", use_column_width=True)
