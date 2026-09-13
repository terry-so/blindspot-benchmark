from openai import OpenAI
from typing import Any, Optional
from google import genai
from google.genai import types
from PIL import Image
import torch
from transformers import Mistral3ForConditionalGeneration
from diffusers import Flux2Pipeline, Flux2Transformer2DModel

"""
ALL MODELS MUST FOLLOW THE SAME OUTPUT FORMAT
"""

class FluxModel:
    """
    Flux model wrapper
    """
    def __init__(self, model_name: str = 'diffusers/FLUX.2-dev-bnb-4bit'):
        self.model_name = model_name

        self.torch_dtype = torch.bfloat16
        self.device = "cuda:0"
        
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_name, subfolder = "transformer", dtype=self.torch_dtype, device_map="cpu"
        )
        
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, subfolder = 'text_encoder', dtype=self.torch_dtype, device_map="cpu"
        )
        
        self.pipe = Flux2Pipeline.from_pretrained(
            model_name, transformer = transformer, text_encoder = text_encoder, torch_dtype = self.torch_dtype
        )
        self.pipe.enable_model_cpu_offload()

    def generate_edit(self, prompt: str, input_image_path: str, output_image_path: str) -> dict[str, Any]:
        """
        Generate edit using FLUX.2 and save to output image path.
        Also return dictionary containing status, text output and edited image.
        """
        output_dict = {"status":"failed", "text_response":None,"image":None}


        try:
            
            img = Image.open(input_image_path).covert("RGB")
            output = self.pipe(
                prompt = prompt,
                generator=torch.Generator(device=self.device).manual_seed(42),
                image = img,
                num_inference_steps = 28,
                guidance_scale = 4.0
            ).images[0]

            output.save(output_image_path)

            output_dict['status'] = 'success'
            output_dict['text_response'] = None
            output_dict['image'] = output

        except Exception as e:
            output_dict['status'] = 'failed'
            output_dict['text_response'] = str(e)
            output_dict['image'] = None

        return output_dict






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

        return {"status":"success/failed/reject", "text_response":None, "image":Image.Image}


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

        return {"status":"success/failed/reject", "text_response":None, "image":Image.Image}



"""
ADD OTHER MODELS HERE
"""


