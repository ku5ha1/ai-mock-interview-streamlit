from model_clients import GeminiLLM

llm = GeminiLLM()
response = llm.invoke("Hello Gemini!")
print(response)