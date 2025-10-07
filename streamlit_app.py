import os
import asyncio
import streamlit_app as st
from graph_builder import build_graph

st.set_page_config(
    page_title="AI LinkedIn 포스트 생성기",
    page_icon="🚀",
    layout="wide"
)

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.warning("⚠️ OPENAI_API_KEY 못 찾음., `.streamlit/secrets.toml` 파일에서 설정해 주시기 바랍니다.")

st.title("🚀 AI LinkedIn 포스트 생성기")
st.markdown("**주제를 입력하면 AI가 자동으로 글과 이미지를 만들어 드립니다.**")

topic = st.text_input("포스트 주제를 입력하세요", placeholder="예: 헬스케어 분야에서의 AI 활용")

if st.button("포스트 생성하기"):
    if not topic.strip():
        st.warning("⚠️ 주제를 입력해 주세요.")
    else:
        with st.spinner("⏳ 포스트를 생성 중입니다..."):
            try:
                graph = build_graph()

                initial_state = {
                    "topic": topic,
                    "post_text": f"LinkedIn 포스트 주제: {topic}",
                    "image_path": ""
                }

                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                final_state = loop.run_until_complete(graph.ainvoke(initial_state))
                loop.close()

               
                st.success("✅ 포스트 생성 완료!")

                st.subheader("✍️ 생성된 LinkedIn 포스트")
                st.write(final_state.get("post_text", "포스트 텍스트를 생성하지 못했습니다."))

                
                image_path = final_state.get("image_path", "")
                if image_path:
                    st.subheader("🖼️ 생성된 이미지")
                    st.image(image_path, caption="AI가 생성한 이미지", use_column_width=True)
                else:
                    st.info("이미지를 생성하지 않았습니다.")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

