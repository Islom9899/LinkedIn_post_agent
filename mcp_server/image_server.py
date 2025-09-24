import os
from mcp.server.fastmcp import FastMCP
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN 환경 변수를 설정해주세요 (Streamlit Secrets 사용 가능).")

# HuggingFace Inference 클라이언트 생성
client = InferenceClient(
    provider="nebius",
    api_key=HF_TOKEN,
)

# FastMCP 서버 생성
mcp = FastMCP("ImageGen")

@mcp.tool()
def generate_image(prompt: str, model: str = "black-forest-labs/FLUX.1-schnell") -> str:
    """
    텍스트 프롬프트를 기반으로 이미지를 생성합니다.
    생성된 이미지를 저장하고 경로를 반환합니다.
    """
    print(f"[ImageGen] 프롬프트로 이미지 생성: {prompt}")

    # 배포 환경에서 쓰기 가능한 임시 폴더 사용
    out_dir = os.environ.get("OUT_DIR", "/tmp/outputs")  
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "image.png")

    # 이미지 생성
    image = client.text_to_image(prompt, model=model)
    image.save(path)

    return path

if __name__ == "__main__":
    mcp.run(transport="stdio")
