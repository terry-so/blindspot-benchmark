from models.wrappers import FluxModel
from typing import Any
from utils.exponential_backoff import retry_with_exponential_backoff
def create_edit_prompt(subcategory: str) -> dict[str,str]:
    """
    Receive subcategory ID (e.g. "1.1") and returns a dictionary of L0 and L1 prompts (expand to L2 and L3 later)
    """

    return {'L0':'change color of the sky to green', 'L1':'Change speed limit to 100'}

def execute_edit(model, prompt: str, input_image_path: str, output_image_path: str)-> dict[str,Any]:
    """
    Receive model name, input_image_path and a prompt and save output image to output_image_path.
    Return the output of model's API wrapper: a dictionary containing status, text output, edited image and error message. 
    """ 
    gen_edit = retry_with_exponential_backoff(model.generate_edit)
    
    return gen_edit(prompt, input_image_path, output_image_path)




