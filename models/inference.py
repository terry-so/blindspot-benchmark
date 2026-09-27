from models.wrappers import FluxModel
from typing import Any
from utils.exponential_backoff import retry_with_exponential_backoff
import json

def create_edit_prompt(model, prompt_path: str, seed_instance: dict) -> dict:
    """
    Receive seed instance and returns and save json
    """
    with open(prompt_path,'r') as f:
        prompt = f.read()

    scene_spec = seed_instance['scene_spec']
    edit_spec = seed_instance['edit_spec']

    prompt = prompt.replace("{scene_spec}", json.dumps(scene_spec, indent = 2))
    prompt = prompt.replace("{edit_spec}", json.dumps(edit_spec, indent = 2))

    gen_prompt = retry_with_exponential_backoff(model.generate_edit_prompt)
    return json.loads(gen_prompt(prompt))


def execute_edit(model, prompt: str, input_image_path: str, output_image_path: str)-> dict[str,Any]:
    """
    Receive model name, input_image_path and a prompt and save output image to output_image_path.
    Return the output of model's API wrapper: a dictionary containing status, text output, edited image and error message. 
    """ 
    gen_edit = retry_with_exponential_backoff(model.generate_edit)

    try:
        return gen_edit(prompt, input_image_path, output_image_path)
    except Exception as e:
        return {
            "status": "failed",
            "text_response": None,
            "image": None,
            "error": str(e)}
    




