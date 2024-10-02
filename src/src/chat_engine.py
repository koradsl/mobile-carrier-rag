from typing import Any

from ollama import Client


class ChatEngine:
    def __init__(self, host: str, model_name: str = "chnaaam/ko-gemma2-9b-it-gguf") -> None:
        self.client = Client(host=host)
        self.model_name = model_name

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "너는 김용현이 개발한 모델이야. 한국어로 대답해줘",
                },
            ]
            + messages,
            stream=True,
        )

        # for r in response:
        #     print(r["message"]["content"], end="")

        return response

