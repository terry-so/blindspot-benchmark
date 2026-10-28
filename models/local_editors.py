"""Local BF16 image editors. Heavy dependencies load only when instantiated."""

import hashlib
import json
from pathlib import Path

from PIL import Image
from utils.benchmark_runner import atomic_json

MODEL_IDS = {
    "qwen-image-2.1": "Qwen/Qwen-Image-2.1",
    "flux-3-klein": "black-forest-labs/flux-3-image-klein",
    "flux-2-dev": "black-forest-labs/FLUX.2-dev",
}
PUP_ID = "black-forest-labs/flux-3-image-klein-pup"


def edit_seed(base_seed, output_path):
    # Independent of machine, output root, execution order, and interrupted runs.
    path = Path(output_path)
    key = f"{path.parent.name}/{path.name}"
    return (base_seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")) % (2**32)


class LocalEditor:
    def __init__(self, model, revision, pup_revision=None, seed=42, cpu_offload=False):
        import torch

        self.torch = torch
        self.model = model
        self.model_name = MODEL_IDS[model]
        self.seed = seed
        self.cpu_offload = cpu_offload
        if model == "qwen-image-2.1":
            from diffusers import QwenImage21Pipeline

            self.pipe = QwenImage21Pipeline.from_pretrained(
                self.model_name, revision=revision, torch_dtype=torch.bfloat16
            )
            if cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")
        elif model == "flux-2-dev":
            from diffusers import Flux2Pipeline

            self.pipe = Flux2Pipeline.from_pretrained(
                self.model_name, revision=revision, torch_dtype=torch.bfloat16
            )
            if cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")
        else:
            from diffusers import ModularPipeline
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            from huggingface_hub import hf_hub_download

            self.pipe = ModularPipeline.from_pretrained(self.model_name, revision=revision)
            self.pipe.load_components(dtype=torch.bfloat16)
            self.pipe.to("cpu" if cpu_offload else "cuda")
            self.processor = AutoProcessor.from_pretrained(PUP_ID, revision=pup_revision)
            self.upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
                PUP_ID, revision=pup_revision, dtype=torch.bfloat16
            ).eval().to("cpu" if cpu_offload else "cuda")
            self.system_prompt = json.loads(Path(hf_hub_download(
                PUP_ID, "system_prompts.json", revision=pup_revision
            )).read_text(encoding="utf-8"))["i2i"]

    def _expand(self, prompt, image):
        if self.cpu_offload:
            self.pipe.to("cpu")
            self.torch.cuda.empty_cache()
            self.upsampler.to("cuda")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Reference images follow in order (the first image, the second image, ...).\n"
                    f"User request: {prompt}\nOutput the caption."
                )},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(self.upsampler.device)
        output = self.upsampler.generate(**inputs, max_new_tokens=3584, do_sample=False)
        caption = self.processor.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not caption:
            raise ValueError("FLUX prompt upsampler returned an empty caption")
        return caption

    def generate_edit(self, prompt, input_image_path, output_image_path):
        seed = edit_seed(self.seed, output_image_path)
        stage = "input"
        metadata = {"instruction": prompt, "seed": seed, "model": self.model_name}
        try:
            with Image.open(input_image_path) as source:
                image = source.convert("RGB")
            with self.torch.inference_mode():
                if self.model == "flux-3-klein":
                    stage = "prompt_expansion"
                    caption = self._expand(prompt, image)
                    metadata["expanded_prompt"] = caption
                    if self.cpu_offload:
                        self.upsampler.to("cpu")
                        self.torch.cuda.empty_cache()
                        self.pipe.to("cuda")
                    stage = "generation"
                    result = self.pipe(
                        prompt=caption, image=image, num_inference_steps=16,
                        generator=self.torch.Generator().manual_seed(seed), output="images",
                    )[0]
                elif self.model == "flux-2-dev":
                    stage = "generation"
                    metadata.update(num_inference_steps=50, guidance_scale=4.0,
                                    caption_upsample_temperature=None)
                    result = self.pipe(
                        prompt=prompt, image=image, num_inference_steps=50,
                        guidance_scale=4.0,
                        generator=self.torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                else:
                    stage = "generation"
                    result = self.pipe(
                        prompt=prompt, image=image, num_inference_steps=40,
                        generator=self.torch.Generator("cuda").manual_seed(seed),
                    ).images[0]
            stage = "save"
            Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_image_path)
            atomic_json(Path(output_image_path).with_suffix(".generation.json"), metadata)
            return {"status": "success", "image": result, "text_response": None, "error": None}
        except Exception as error:
            metadata.update(error=str(error), failed_stage=stage)
            atomic_json(Path(output_image_path).with_suffix(".generation.json"), metadata)
            return {
                "status": "failed", "image": None, "text_response": None,
                "error": f"{stage}: {type(error).__name__}: {error}", "technical_error": True,
            }
