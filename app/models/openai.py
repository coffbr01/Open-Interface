import os
from typing import Any
from models.model import Model

from openai import OpenAI


class OpenAIModel(Model):
    def __init__(self, model_name, base_url, api_key, context):
        super().__init__(model_name, base_url, api_key, context)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
