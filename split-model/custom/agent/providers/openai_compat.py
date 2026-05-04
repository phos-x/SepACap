import os
import logging
from typing import Optional
from openai import OpenAI
from .base import LLMProviderInterface

logger = logging.getLogger(__name__)

class OpenAICompatibleProvider(LLMProviderInterface):
    """
    Handles any LLM server that uses the standard OpenAI API spec.
    (e.g., Groq, OpenAI, Local Ollama, vLLM).
    """
    def __init__(self, base_url: str, model_name: str, api_key_env_var: Optional[str] = None):
        self.model_name = model_name
        
        # Safely grab the API key from the environment, defaulting to "dummy" for local servers
        api_key = "dummy_key"
        if api_key_env_var:
            api_key = os.environ.get(api_key_env_var)
            if not api_key:
                logger.warning(f"Environment variable {api_key_env_var} is not set. API calls may fail.")

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Provider API Error: {e}")
            raise RuntimeError(f"Provider failed to generate response: {e}")