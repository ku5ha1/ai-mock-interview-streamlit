import os
from typing import Any, List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
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
        else:
            # Treat as string
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

# Gemma implementation
class GemmaLLM(SimpleLLM):
    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Load the model from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-1b-it")
            self.model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-1b-it",
                                                             device_map="auto",
                                                             torch_dtype=torch.float16)

        except ImportError:
            raise ImportError("You need to install transformers and torch to use Gemma.")

    async def _process_messages(self, messages): # Implement as asynchronous
        # Convert messages to a conversation format
        conversation = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                conversation += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                conversation += f"Human: {message.content}\n"
            else:
                conversation += f"AI: {message.content}\n"

        conversation += "AI: "
        return await self._process_text(conversation)

    async def _process_text(self, text): # Implement as asynchronous
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

# DeepSeek implementation
class DeepSeekLLM(SimpleLLM):
    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Load the model from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
            self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",
                                                             torch_dtype=torch.float16,
                                                             device_map="auto")

        except ImportError:
            raise ImportError("You need to install transformers and torch to use DeepSeek.")

    async def _process_messages(self, messages): # Implement as asynchronous
        # Convert messages to a conversation format
        conversation = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                conversation += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                conversation += f"Human: {message.content}\n"
            else:
                conversation += f"AI: {message.content}\n"

        conversation += "AI: "
        return await self._process_text(conversation)

    async def _process_text(self, text): # Implement as asynchronous
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

# Factory function to get the appropriate LLM client
def get_llm_client(model_name: str) -> SimpleLLM:
    model_name = model_name.lower()

    if model_name == "gemini":
        return GeminiLLM()
    elif model_name == "gemma":
        return GemmaLLM()
    elif model_name == "deepseek":
        return DeepSeekLLM()
    else:
        # Default to Gemini
        print(f"Unknown model: {model_name}. Using Gemini as default.")
        return GeminiLLM()