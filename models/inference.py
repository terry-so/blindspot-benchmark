from models import OpenAIModel
def create_edit_prompt(subcategory: str) -> dict[str,str]:
    """
    Receive subcategory ID (e.g. "1.1") and returns a dictionary of L0 and L1 prompts (expand to L2 and L3 later)
    """

    return {'L0':'change color of the sky to green', 'L1':'Change speed limit to 100'}

def execute_edit(model: str, prompt: str, input_image_path: str, output_image_path: str):
    """
    Recieve model name, input_image_path and a prompt and save output image to output_image_path.
    """ 
    
    return model.generate_edit(prompt, input_image_path, output_image_path)