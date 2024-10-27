import os
from typing import Any


class Model:
    def __init__(self, model_name, base_url, api_key, context):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.context = context

    def get_instructions_for_objective(self, *args) -> dict[str, Any]:
        pass

    def cleanup(self, *args):
        pass
