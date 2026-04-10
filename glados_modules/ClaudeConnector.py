"""Anthropic Claude API connector for GLaDOS personality responses.

Drop-in replacement for ChatGPTConnector. Same interface: Thread-based,
sets real_response when the API call completes. Configured via [CLAUDE]
section in glog.conf.

Recommended models for GLaDOS:
    claude-haiku-4-5-20251001  — fastest, cheapest, great for quips (~200ms)
    claude-sonnet-4-6          — balanced speed + intelligence (~500ms)
    claude-opus-4-6            — most capable, slower (~1-2s)
"""

# builtin
from threading import Thread

# 3rd party
import requests

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class GladosClaude(Thread):
    """Anthropic Claude API connector with the same interface as GladosGPT.

    Usage matches GladosGPT exactly — instantiate with config + prompt,
    optionally add_prompt() for context, then start() and poll real_response.
    """

    def __init__(self, configp, prompt) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.real_response = None
        self.prompt = prompt
        self.configp = configp["CLAUDE"]
        self.model = self.configp["model"]
        self.api_key = self.configp["apikey"]
        self.api_endpoint = self.configp.get(
            "endpoint", "https://api.anthropic.com/v1/messages")
        self.content = self.configp["user_prompt"]
        self.max_tokens = int(self.configp.get("max_tokens", "1024"))
        self.updated_content = None

    def add_prompt(self, content) -> None:
        """Add extra context to the system prompt (e.g., sight description).

        Args:
            content: Additional context string to append.
        """
        self.updated_content = content

    def generate_text(self) -> None:
        """Call the Anthropic Claude API and store the response."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        if self.updated_content is None:
            system_content = self.content
        else:
            system_content = f"{self.content}. {self.updated_content}"

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_content,
            "messages": [
                {"role": "user", "content": self.prompt}
            ],
        }

        try:
            response = requests.post(self.api_endpoint, headers=headers,
                                      json=data, timeout=30)
            if response.status_code == 200:
                response_json = response.json()
                self.real_response = response_json["content"][0]["text"].strip()
                self.logger.debug(f"Response Done: {self.real_response}")
            else:
                self.logger.error(
                    f"Claude API failed. Status: {response.status_code}")
                self.logger.debug(response.text)
                # Set a fallback so the main loop doesn't hang forever
                self.real_response = "I seem to be having trouble thinking right now."
        except requests.exceptions.Timeout:
            self.logger.error("Claude API request timed out after 30s")
            self.real_response = "I seem to be having trouble thinking right now."
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            self.real_response = "I seem to be having trouble thinking right now."

    def run(self) -> None:
        """Thread entry point — calls generate_text."""
        self.generate_text()
