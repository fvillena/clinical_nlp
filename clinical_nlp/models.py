import requests
import json
from unidecode import unidecode
import re
# logger
import logging

logger = logging.getLogger(__name__)


class IclModel:
    def __init__(self, model_url, max_tokens=256, stop=["\n", "###"], max_retries=3):
        self.model_url = model_url
        self.max_tokens = max_tokens
        self.stop = stop
        self.max_retries = max_retries

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


class IclClassifier(IclModel):
    def contextualize(self, system_message, classes, user_template, retry_message):
        classes_string = ""
        for c in classes[:-1]:
            classes_string += c + ", "
        classes_string += f"y {classes[-1]}"
        self.system_message = system_message.replace("<classes>", classes_string)
        self.classes = classes
        self.user_template = user_template
        self.retry_message = retry_message

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


class IclNer(IclModel):
    def contextualize(self, system_message, user_template, entities):
        keys = list(entities.keys())
        values = list(entities.values())
        items = list(entities.items())
        entity_names_string = ""
        for e in values[:-1]:
            entity_names_string += e + ", "
        entity_names_string += f"y {values[-1]}"
        schema_string = ""
        for e,d in items[:-1]:
            schema_string += f'"{e}" para {d}' + ", "
        schema_string += f'y "{items[-1][0]}" para {items[-1][1]}'
        self.system_message = system_message.replace("<entities>", entity_names_string).replace("<schema>", schema_string)
        self.user_template = user_template
    
    def predict(self, x):
        def get_json(completion):
            try:
                json = re.search(r".*({.*}).*", completion, flags=re.DOTALL).group(1).strip()
            except:
                raise Exception(f"No JSON was found on your answer.")
            return re.sub("(\w+):", r'"\1":',  json)
        messages = [
            {"content": f"{self.system_message}", "role": "system"},
            {
                "content": f"{self.user_template}".replace("<x>", x),
                "role": "user",
            },
        ]
        completion = self.query_model(messages)
        logger.info(f"Completion: {completion}")
        try:
            y = json.loads(get_json(completion))
        except Exception as e:
            logger.warning(e)
            messages.append({"content": completion, "role": "assistant"})
            messages.append({"content": f"No puedo decodificar tu json porque tiene este error: {e}", "role": "user"})
            try:
                completion = self.query_model(messages)
                y = json.loads(get_json(completion))
            except:
                logger.warning(f"JSON decoding error: {completion}")
                y = None
        return y
