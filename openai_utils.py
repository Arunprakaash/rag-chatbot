from typing import Optional

from openai import OpenAI

from config import OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE


class OpenAIUtils:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client

    def get_embeddings(self, message: str, model: Optional[str] = OPENAI_EMBEDDING_MODEL):
        return self.openai_client.embeddings.create(input=message, model=model).data[0].embedding

    @staticmethod
    def prepare_system_prompt(context: str):
        # Prepare the prompt template
        prompt_template = ("You are an AI assistant named Viamagus. "
                           "Your role is to assist the user by answering their queries based on the context provided. "
                           "You have access to the following previous interaction context:\n"
                           f"{context}\n"
                           )

        return prompt_template

    def chat_with_openai(self, query: str, system_prompt: str):
        res = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return res.choices[0].message.content
