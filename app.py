import os
import streamlit as st
from dotenv import load_dotenv
from agent_logic import run_workflow, mermaid_diagram, ensure_outputs

load_dotenv()
ensure_outputs()

# ============================================================
# 다국어 UI 설정 (한국어 / 영어 / 우즈베크어)
# ============================================================
UI = {
    "ko": {
        "title": "🚀 AI LinkedIn 포스트 생성기",
        "subtitle": "**주제를 입력하면 AI가 자동으로 전문적인 글과 이미지를 만들어 드립니다.**",
        "topic_placeholder": "예: 양자 컴퓨팅의 미래와 기업 적용 방안",
        "generate_btn": "포스트 생성하기",
        "enter_topic_warning": "⚠️ 주제를 입력해 주세요.",
        "success": "✅ 포스트 생성 완료!",
        "image_error": "이미지 생성 오류: ",
        "no_image": "이미지를 생성하지 못했습니다.",
        "diagram": "🔧 워크플로우 다이어그램",
        "saved_info": "🔖 생성된 데이터가 outputs/posts에 저장되었습니다."
    },
    "en": {
        "title": "🚀 AI LinkedIn Post Generator",
        "subtitle": "**Enter a topic and AI will automatically create a post and image.**",
        "topic_placeholder": "e.g. Future of Quantum Computing in Business",
        "generate_btn": "Generate Post",
        "enter_topic_warning": "⚠️ Please enter a topic.",
        "success": "✅ Post generated!",
        "image_error": "Image generation error: ",
        "no_image": "No image generated.",
        "diagram": "🔧 Workflow Diagram",
        "saved_info": "🔖 Data saved to outputs/posts."
    },
    "uz": {
        "title": "🚀 AI LinkedIn post generator",
        "subtitle": "**Mavzuni kiriting — AI post va rasm yaratadi.**",
        "topic_placeholder": "Masalan: Kvant hisoblashning kelajagi",
        "generate_btn": "Post yaratish",
        "enter_topic_warning": "⚠️ Mavzu kiriting.",
        "success": "✅ Post yaratildi!",
        "image_error": "Rasm yaratishda xatolik: ",
        "no_image": "Rasm yaratilmadi.",
        "diagram": "🔧 Ish oqimi diagrammasi",
        "saved_info": "🔖 Ma’lumotlar outputs/posts ichiga saqlandi."
    }
}

# ============================================================
# Streamlit 메인 UI
# ============================================================
st.set_page_config(page_title="AI LinkedIn Generator", page_icon="🚀", layout="wide")

lang = st.sidebar.selectbox("언어 / Language / Til", ["ko", "en", "uz"],
                            format_func=lambda x: {"ko": "🇰🇷 한국어", "en": "🇬🇧 English", "uz": "🇺🇿 O‘zbekcha"}[x])
S = UI[lang]

st.title(S["title"])
st.markdown(S["subtitle"])

# st.text_input ogohlantirishini bartaraf etish uchun label_visibility="collapsed" qo'shildi
topic = st.text_input(
    "Mavzu kiriting", 
    placeholder=S["topic_placeholder"],
    label_visibility="collapsed"
)

st.markdown(f"### {S['diagram']}")
st.markdown(mermaid_diagram())

if st.button(S["generate_btn"]):
    if not topic.strip():
        st.warning(S["enter_topic_warning"])
    else:
        progress = st.progress(0)
        status_text = st.empty()

        def progress_callback(percent: int, message: str):
            try:
                progress.progress(min(max(int(percent), 0), 100))
                status_text.markdown(f"**{message}**")
            except Exception:
                pass

        initial_state = {
            "topic": topic,
            "post_text": f"LinkedIn post topic: {topic}",
            "image_path": "",
            "error": ""
        }

        try:
            final_state = run_workflow(initial_state, progress_callback=progress_callback)
            
            # Qo'shimcha statusni tozalash
            progress.empty()
            status_text.empty()
            
            st.success(S["success"])
            st.subheader("✍️ 생성된 포스트")
            st.write(final_state.get("post_text", "포스트 내용을 불러오지 못했습니다."))

            image_path = final_state.get("image_path", "")
            error_message = final_state.get("error", "")

            if image_path and os.path.exists(image_path):
                st.subheader("🖼️ 생성된 이미지")
                st.image(image_path, caption="AI가 생성한 이미지", width=600) 
            elif error_message:
                st.error(S["image_error"] + str(error_message))
            else:
                st.info(S["no_image"])

            st.info(S["saved_info"])

        except Exception as e:
            st.error(f"❌ 워크플로우 실행 오류: {e}")