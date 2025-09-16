import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Define a simple LLM wrapper class for consistent interface
class SimpleLLM:
    def __init__(self, model_client):
        self.model_client = model_client

    async def invoke(self, prompt): # Make invoke asynchronous
        if isinstance(prompt, ChatPromptTemplate):
            messages = prompt.format_messages()
            return await self._process_messages(messages) # Await the processing
        # Handle already-formatted message lists
        if isinstance(prompt, list):
            return await self._process_messages(prompt)
        # Fallback: treat as plain text
        return await self._process_text(str(prompt)) # Await the processing

    async def _process_messages(self, messages): # Make abstract methods asynchronous
        raise NotImplementedError("Subclass must implement")

    async def _process_text(self, text): # Make abstract methods asynchronous
        raise NotImplementedError("Subclass must implement")

# Gemini implementation
class GeminiLLM(SimpleLLM):
    def __init__(self):
        try:
            import google.generativeai as genai

            # Configure the Gemini API
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("Warning: GEMINI_API_KEY not set. Using default configuration.")

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

        except ImportError:
            raise ImportError("You need to install google-generativeai package to use Gemini.")

    async def _process_messages(self, messages): # Implement as asynchronous
        gemini_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                gemini_messages.append({"role": "system", "parts": [message.content]})
            elif isinstance(message, HumanMessage):
                gemini_messages.append({"role": "user", "parts": [message.content]})
            else:
                gemini_messages.append({"role": "model", "parts": [message.content]})

        response = self.model.generate_content(gemini_messages)
        return response.text

    async def _process_text(self, text): # Implement as asynchronous
        response = self.model.generate_content(text)
        return response.text

# Factory function to get the appropriate LLM client
def get_llm_client(model_name: str) -> SimpleLLM:
    # The app now only supports Gemini; ignore model_name
    return GeminiLLM()