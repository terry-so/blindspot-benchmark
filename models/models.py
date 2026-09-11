from openai import OpenAI
from typing import Any, Optional
from google import genai
from google.genai import types


"""
ALL MODELS MUST FOLLOW THE SAME OUTPUT FORMAT
"""

class OpenAIModel:
    """
    OpenAI model wrapper
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()

    def generate_edit(self, prompt: str, input_image_path: str, output_image_path: str) -> dict[str, Any]:
        """
        Generate edit using OpenAI API and save to output image path.
        Also return dictionary containing status, text output and edited image.
        """

        #https://developers.openai.com/api/docs/guides/image-generation?reference-images-api=image#edit-images

        return {"status":"success/failed/reject", "text_response":None, "image":PIL.Image}


class GeminiModel:
    """
    Gemini model wrapper
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client()

    def generate_edit(self, prompt: str, input_image_path: str, output_image_path: str) -> dict[str, Any]:
        """
        Generate edit using Gemini API and save to output image path.
        Also return dictionary containing status, text output and edited image.
        """
        #https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite

        return {"status":"success/failed/reject", "text_response":None, "image":PIL.Image}



"""
ADD OTHER MODELS HERE
"""


