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
                    "content": "너는 메이식스에서 개발한 산타(SANTA)야. 한국어로 대답해줘",
                },
            ]
            + messages,
            stream=True,
        )

        # for r in response:
        #     print(r["message"]["content"], end="")

        return response


if __name__ == "__main__":
    chat_engine = ChatEngine(host="http://chnaaam.com:11434")
    messages = [
        {
            "role": "user",
            "content": "LED 좀 꺼줘 :)",
        },
    ]
    response = chat_engine.chat(messages=messages)

    for r in response:
        print(r["message"]["content"], end="")
