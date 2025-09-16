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
            load_dotenv()

            # Get Hugging Face token from environment
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                print("Warning: HUGGINGFACE_HUB_TOKEN not set. You may need to authenticate to access the model.")
                print("Set HUGGINGFACE_HUB_TOKEN environment variable or run 'huggingface-cli login'")

            # Resolve model id from environment with a safer default
            # Known-good public IDs include: google/gemma-1.1-2b-it, google/gemma-2-2b-it
            model_id = os.environ.get("GEMMA_MODEL_ID", "google/gemma-1.1-2b-it")

            # Load the model from Hugging Face with token
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=hf_token,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            except Exception as e:
                # Provide a clearer message for missing/blocked repos
                raise RuntimeError(
                    f"Failed to load Gemma model '{model_id}'. Ensure the repo exists, you have access, and your HUGGINGFACE_HUB_TOKEN is set. "
                    "Try setting GEMMA_MODEL_ID to a public model like 'google/gemma-1.1-2b-it' or run 'huggingface-cli login'.\n"
                    f"Original error: {e}"
                )

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
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

# DeepSeek implementation
class DeepSeekLLM(SimpleLLM):
    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            load_dotenv()

            # Get Hugging Face token from environment
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                print("Warning: HUGGINGFACE_HUB_TOKEN not set. You may need to authenticate to access the model.")
                print("Set HUGGINGFACE_HUB_TOKEN environment variable or run 'huggingface-cli login'")

            # Load the model from HuggingFace with token
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",
                                                             token=hf_token,
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
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
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