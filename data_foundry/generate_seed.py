from models.wrappers import GeminiModel
from utils.exponential_backoff import retry_with_exponential_backoff
import os
import yaml
import json

def generate_seed_instance(subcategory_template_path, output_path, model = GeminiModel("gemini-3.8-flash")):
    gen_seed = retry_with_exponential_backoff(model.generate_seed_instance)

    with open(output_path, 'w') as f:
        f.write(gen_seed(subcategory_template_path))

def generate_seed_image(seed_json_path, prompt_path, output_path, model = GeminiModel("gemini-3-pro-image")):
    with open(seed_json_path, 'r') as f:
        seed = json.load(f)

    with open(prompt_path,'r') as f:
        prompt = f.read()
    scene_spec = seed['scene_spec']
    edit_spec = seed['edit_spec']
    prompt = prompt.replace('{scene type}',scene_spec['scene_type'])
    prompt = prompt.replace('{object}',scene_spec['object'])
    prompt = prompt.replace('{location}',scene_spec['location'])
    prompt = prompt.replace('{attribute}',scene_spec['location'])
    prompt = prompt.replace('{original_value}',edit_spec['original_value'])
    prompt = prompt.replace('{camera}',scene_spec['appearance']['camera'])
    prompt = prompt.replace('{lighting}',scene_spec['appearance']['lighting'])
    print(prompt)
    gen_img = retry_with_exponential_backoff(model.generate_seed_image)
    return gen_img(prompt, output_path)
        
        












if __name__ == "__main__":
    pass





