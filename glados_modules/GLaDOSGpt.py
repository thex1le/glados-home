from threading import Thread

# 3rd party
import requests

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import LoggingEnums


class GladosGPT(Thread):
    def __init__(self, configp, prompt):
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.real_response = None
        self.prompt = prompt
        self.configp = configp["OPENAI"]
        self.model = self.configp["model"]
        self.api_key = self.configp["apikey"]
        self.api_endpoint = self.configp["endpoint"]
        self.content = self.configp["user_prompt"]
        self.updated_content = None

    def add_prompt(self, content):
        # allow extra info to be added to the user_prompt
        self.updated_content = content

    def generate_text(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",}
        if self.updated_content is None:
            local_content = self.content
        else:
            local_content = f"{self.content}. {self.updated_content}"
        print(local_content)
        data = {
                "model": self.model,
                "messages": [{"role": "system", "content": local_content},
                    {"role": "user", "content": self.prompt}],
                "max_tokens": 1500}
        response = requests.post(self.api_endpoint, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            self.real_response = response_json['choices'][0]['message']['content'].strip()
            msg = f"Response Done: {self.real_response}"
            self.logger.debug(msg)
            print(msg)
        else:
            self.logger.error(f"Failed to call the API. Status code: {response.status_code}")
            self.logger.debug(response.text)

    def run(self):
        self.generate_text()

