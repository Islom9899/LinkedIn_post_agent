# 🚀 AI LinkedIn Post Generator (LangGraph Agent)

## Project Overview
This project demonstrates an intelligent AI agent built with **LangGraph** that automatically creates professional LinkedIn posts and corresponding images based on a given topic.

## Key Features
- **Two-Step Workflow:** Separates post generation (GPT-4) and image generation (DALL·E 3 via an Agent).
- **LangGraph Orchestration:** Utilizes a state machine for robust, step-by-step process management.
- **Streamlit UI:** Provides an easy-to-use web interface.

## Installation
```bash
pip install -r requirements.txt
```

## Execution
```bash
streamlit run app.py
```

## Project Structure
```
AI_LinkedIn_Generator/
├── app.py               # Streamlit web interface and UI (사용자 인터페이스)
├── agent_logic.py       # LangGraph workflow, Agent, and Tools logic (AI 로직)
├── requirements.txt     # Python dependencies (필수 라이브러리)
├── README.md            # Project documentation (프로젝트 설명)
├── .env.example         # Environment variables template (환경 변수 템플릿)
└── outputs/
    └── posts/.gitkeep   # Storage for generated posts and images (결과물 저장소)
```
