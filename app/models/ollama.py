import os
import base64
import json
import ollama
from typing import Any
from models.model import Model
from utils.screen import Screen


class OllamaModel(Model):
    def __init__(self, model_name, base_url, api_key, context):
        super().__init__(model_name, base_url, api_key, context)

        # Running context of user and system messages
        self.messages = self.init_messages()

    def init_messages(self):
        return [
            {
                'role': 'system',
                'content': self.context
            }
        ]

    def get_instructions_for_objective(self, original_user_request: str, step_num: int = 0) -> dict[str, Any]:
        # Take screenshot and base64 encode it
        encoded_screenshot = self.base_64_encode_screenshot()

        # Format user request to send to LLM
        formatted_user_request = self.format_user_request_for_llm(original_user_request, step_num, encoded_screenshot)

        self.messages.append(formatted_user_request)

        # Read response
        llm_response = self.send_message_to_llm()
        json_instructions: dict[str, Any] = self.convert_llm_response_to_json_instructions(llm_response)

        return json_instructions

    def send_message_to_llm(self) -> str:
        response = ollama.chat(model=self.model_name, messages=self.messages, stream=False)
        return response

    def base_64_encode_screenshot(self):
        # Take screenshot
        filepath = Screen().get_screenshot_file()

        # Base64 encode it
        with open(filepath, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())

        # Done reading screenshot. Delete it.
        # os.remove(filepath)

        return encoded_string.decode('utf-8')

    def format_user_request_for_llm(self, original_user_request, step_num, encoded_screenshot) -> dict[str, Any]:
        content = 'Step number: ' + str(step_num) + "\n\nOriginal request: " + original_user_request

        return {
            'role': 'user',
            'content': content, 
            'images': [encoded_screenshot]
        }

    def convert_llm_response_to_json_instructions(self, llm_response: Any) -> dict[str, Any]:
        llm_response_data: str = llm_response['message']['content'].strip()

        # Our current LLM model does not guarantee a JSON response hence we manually parse the JSON part of the response
        # Check for updates here - https://platform.openai.com/docs/guides/text-generation/json-mode
        start_index = llm_response_data.find('{')
        end_index = llm_response_data.rfind('}')

        try:
            json_response = json.loads(llm_response_data[start_index:end_index + 1].strip())
        except Exception as e:
            print(f'Error while parsing JSON response - {e}')
            json_response = {}

        return json_response

    def cleanup(self):
        self.messages = self.init_messages()
