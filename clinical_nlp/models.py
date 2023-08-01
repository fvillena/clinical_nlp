import requests
import json
from unidecode import unidecode
# logger
import logging
logger = logging.getLogger(__name__)


class IclModel:
    def __init__(self, model_url, max_tokens=256, stop=["\n", "###"], max_retries=3):
        self.model_url = model_url
        self.max_tokens = max_tokens
        self.stop = stop
        self.max_retries = max_retries

    def contextualize(self, system_message, classes, user_template, retry_message):
        classes_string = ""
        for c in classes[:-1]:
            classes_string += c + ", "
        classes_string += f"y {classes[-1]}"
        self.system_message = system_message.replace("<classes>", classes_string)
        self.classes = classes
        self.user_template = user_template
        self.retry_message = retry_message

    def query_model(self, messages):
        payload = json.dumps(
            {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "stop": self.stop,
            }
        )
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        response = requests.request(
            "POST", self.model_url, headers=headers, data=payload
        )
        completion = response.json()["choices"][0]["message"]["content"].strip()
        return completion

    def predict(self, x):
        def preprocess(sentence):
            return unidecode(sentence.strip().lower())
        def label(completion):
            y = None
            for c in self.classes:
                if preprocess(c) in preprocess(completion):
                    y = c.upper()
                    break
            return y
        messages = [
            {"content": f"{self.system_message}", "role": "system"},
            {
                "content": f"{self.user_template}".replace("<x>", x),
                "role": "user",
            },
        ]
        for i in range(self.max_retries):
            completion = self.query_model(messages)
            logger.info(f"Completion: {completion}")
            y = label(completion)
            if y is not None:
                break
            else:
                logger.warning(f"No class was found on: {completion}.")
                messages.append({"content": completion, "role": "assistant"})
                messages.append({"content": self.retry_message, "role": "user"})
        return y
        
