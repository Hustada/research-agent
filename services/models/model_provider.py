from typing import Optional, Dict, Any
from langchain_community.llms import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_anthropic import Anthropic
from abc import ABC, abstractmethod

class BaseModelProvider(ABC):
    @abstractmethod
    def get_model(self) -> Any:
        pass

class OpenAIProvider(BaseModelProvider):
    def __init__(self, api_key: str, temperature: float = 0.7, max_tokens: int = 1500):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model(self):
        return OpenAI(
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

class GeminiProvider(BaseModelProvider):
    def __init__(self, api_key: str, temperature: float = 0.7, max_tokens: int = 1500):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model(self):
        return GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )

class AnthropicProvider(BaseModelProvider):
    def __init__(self, api_key: str, temperature: float = 0.7, max_tokens: int = 1500):
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model(self):
        return Anthropic(
            anthropic_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model="claude-2"
        )

class ModelFactory:
    PROVIDERS = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider,
        'anthropic': AnthropicProvider
    }

    @classmethod
    def create_model(cls, provider: str, api_key: str, temperature: float = 0.7, max_tokens: int = 1500) -> Any:
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported model provider: {provider}. Choose from: {', '.join(cls.PROVIDERS.keys())}")
        
        provider_class = cls.PROVIDERS[provider]
        model_provider = provider_class(api_key, temperature, max_tokens)
        return model_provider.get_model()
