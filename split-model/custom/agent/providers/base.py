from abc import ABC, abstractmethod

class LLMProviderInterface(ABC):
    """
    Strict interface for all LLM backend servers.
    The Agent relies on this contract, guaranteeing portability.
    """
    
    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """
        Takes a prompt and strictly returns a JSON-formatted string.
        """
        raise NotImplementedError("Providers must implement the generate_json method.")