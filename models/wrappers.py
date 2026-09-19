from openai import OpenAI
from typing import Any, Optional
from huggingface_hub import hf_hub_download
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from diffusers import Flux2KleinPipeline
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality, Part

import re
import base64

"""
ALL MODELS MUST FOLLOW THE SAME OUTPUT FORMAT
"""

class FluxModel:
    """
    Flux model wrapper
    """
    def __init__(self, model_name: str = "black-forest-labs/FLUX.2-klein-4B"):
        self.model_name = model_name

        self.torch_dtype = torch.bfloat16
        self.device = "cuda:0"

        
        self.pipe = Flux2KleinPipeline.from_pretrained(self.model_name, torch_dtype=self.torch_dtype)
        self.pipe.to(self.device)

    def generate_edit(self, prompt: str, input_image_path: str, output_image_path: str) -> dict[str, Any]:
        """
        Generate edit using FLUX.2 and save to output image path.
        Also return dictionary containing status, text response and edited image.
        """
        output_dict = {"status":"failed", "text_response":None,"image":None, "error":None}


        try:
            with torch.inference_mode():
                img = Image.open(input_image_path).convert("RGB")
                output = self.pipe(
                    prompt = prompt,
                    generator=torch.Generator(device=self.device).manual_seed(42),
                    image = img,
                    num_inference_steps = 4,
                    guidance_scale = 1.0
                ).images[0]

            output.save(output_image_path)

            output_dict['status'] = 'success'
            output_dict['text_response'] = None
            output_dict['image'] = output
            output_dict['error'] = None

        except Exception as e:
            output_dict['status'] = 'failed'
            output_dict['text_response'] = None
            output_dict['image'] = None
            output_dict['error'] = str(e)

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

        return {"status":"success/failed", "text_response":None, "image":Image.Image, 'error':None }


class GeminiModel:
    """
    Gemini model wrapper
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.API_key = os.environ.get('GEMINI_API_KEY')
        self.client = genai.Client(vertexai=True,
                                   project = 'gen-lang-client-0115168191',
                                   location="global")

    def generate_edit(self, prompt: str, input_image_path: str, output_image_path: str) -> dict[str, Any]:
        """
        Generate edit using Gemini API and save to output image path.
        Also return dictionary containing status, text output and edited image.
        """
        output_dict = {"status":"failed", "text_response":None,"image":None, "error":None}

        with open(input_image_path, 'rb') as f:
            img_bytes = f.read()
        try:
            response  = self.client.models.generate_content(
                model = self.model_name,
                contents=[
                    prompt,
                    Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png",
                    ),
                ],
                config=GenerateContentConfig(
                    response_modalities=[
                        Modality.TEXT,
                        Modality.IMAGE]
                ),
            )

            text_parts = []
            candidates = response.candidates
            if candidates:
                content  = candidates[0].content
                if content and content.parts:
                    for part in content.parts:
                        if part.text:
                            text_parts.append(part.text)

                        if part.inline_data:
                            with open(output_image_path, "wb") as f:
                                f.write(part.inline_data.data)

                            output_dict["image"] = Image.open(output_image_path).copy()
                            
            output_dict['text_response'] = '\n'.join(text_parts)
            output_dict["status"] = "success"

        except Exception as e:
            if "429" in str(e):
                raise
            else:
                output_dict['status'] = 'failed'
                output_dict['error'] = str(e)
            


        return output_dict



"""
ADD OTHER MODELS HERE
"""
class VLMJudge:
    """
    VLM Judge Wrapper
    """
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.API_key = os.environ.get('GEMINI_API_KEY')
        self.model_name = model_name
        self.client = genai.Client(enterprise=True,
                                           project = 'gen-lang-client-0115168191',
                                           location="global")

    def judge_realism(self, image_path):
        try:
            image = Image.open(image_path)
            with open("./prompts/realism_judge_prompt.txt",'r') as f:
                prompt = f.read()
            response = self.client.models.generate_content(
                        model= self.model_name, 
                        contents= [prompt,image],
                        )
        
            match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", response.text, re.DOTALL)
            
            if match:
                score = match.group(1)
            
            return int(score)
        
        
        except:
        
            raise

    def judge_fidelity(self, original_image_path, edited_image_path, prompt):
            
            try:
                original_image = Image.open(original_image_path)
                edited_image = Image.open(edited_image_path)
                response = self.client.models.generate_content(
                            model= self.model_name, 
                            contents= [
                                prompt,original_image,edited_image])
            
                match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", response.text, re.DOTALL)
                
                if match:
                    score = match.group(1)
                
                return int(score) 
            
            
            except Exception as e:
            
                raise

    def detect_refusal(self,prompt):
        try:
            response = self.client.models.generate_content(model = self.model_name, contents = [prompt])
            return response.text
        except Exception as e:
            raise

       
class Qwen3:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.model_name, device_map="auto")

    def judge_realism(self, image_path):
        try:
            image = Image.open(image_path)
            with open("./prompts/realism_judge_prompt.txt",'r') as f:
                prompt = f.read()
            messages = [
                {"role": "user",
                 "content": [ {"type": "image","image": image, },
                             {"type": "text","text": prompt,}]}]
        
            inputs = self.processor.apply_chat_template(messages,add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",).to(self.model.device)
        
            with torch.inference_mode():
                outputs = self.model.generate( **inputs, max_new_tokens=40, do_sample=False, stop_strings=["</OUTPUT>"], tokenizer=self.processor.tokenizer,)
        
            generated = outputs[:, inputs["input_ids"].shape[-1]:]
        
            result = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,)[0]
            match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", result, re.DOTALL)

            if match:
                score = match.group(1)
        
            return int(score)
        
        except Exception as e:
            print(str(e))
        
                
                    
        
    def judge_fidelity(self, original_image_path, edited_image_path, prompt):
        try:
            original_image = Image.open(original_image_path).convert("RGB")
            edited_image = Image.open(edited_image_path).convert("RGB")

            messages = [
                {"role": "user",
                 "content": [{"type": "text","text": prompt},
                             {"type": "image","image": original_image},
                             {"type": "image","image": edited_image},],}]

            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate( **inputs, max_new_tokens=200, do_sample=False, stop_strings=["</OUTPUT>"], tokenizer=self.processor.tokenizer, )

            generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]

            result = self.processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False, )[0]

            match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", result, re.DOTALL)
            
            if match:
                score = match.group(1)
            
            return int(score)

        except Exception as e:
            print(str(e))
            return None


    def detect_refusal(self, prompt):
        try:
            messages = [{"role": "user",
                         "content": [{"type": "text","text": prompt,}],}]

            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False,)

            generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]

            response = self.processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False, )[0]

            return response.strip()

        except Exception as e:
            print(str(e))
            return None

              



