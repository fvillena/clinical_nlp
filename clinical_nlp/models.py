import requests
import json

class IclModel:
    def __init__(self, model_url, max_tokens=256, stop=["\n", "###"]):
        self.model_url = model_url
        self.max_tokens = max_tokens
        self.stop = stop

    def contextualize(self, system_message, classes, user_template):
        classes_string = ""
        for c in classes[:-1]:
            classes_string += c + ", "
        classes_string += f"y {classes[-1]}"
        self.system_message = system_message.replace("<classes>", classes_string)
        self.classes = classes
        self.user_template = user_template

    def predict(self, x):
        payload = json.dumps(
            {
                "messages": [
                    {"content": f"{self.system_message}", "role": "system"},
                    {
                        "content": f"{self.user_template}".replace("<x>", x),
                        "role": "user",
                    },
                ],
                "max_tokens": self.max_tokens,
                "stop": self.stop,
            }
        )
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        response = requests.request(
            "POST", self.model_url, headers=headers, data=payload
        )
        completion = response.json()["choices"][0]["message"]["content"].strip()
        y = None
        for c in self.classes:
            if c in completion:
                y = c.upper()
                break
        return y