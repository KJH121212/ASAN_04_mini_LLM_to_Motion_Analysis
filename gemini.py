import os
from dotenv import load_dotenv
from google import genai

# 1. .env 파일 로드
load_dotenv()

client = genai.Client()

print("--- Gemini 2.5 단발성 응답 테스트 (New SDK) ---")

prompt = input("나: ")

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt
)

print("\nGemini:")
print(response.text)