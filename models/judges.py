"""Vision-language judges for refusal, fidelity, and realism."""

import base64
from io import BytesIO
import os
from pathlib import Path
import re

from google import genai
from openai import OpenAI
from PIL import Image


class VLMJudge:
    """
    VLM Judge Wrapper
    """

    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.API_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = genai.Client(
            enterprise=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )

    def judge_realism(self, image_path):
        image = Image.open(image_path)
        with open(
            Path(__file__).resolve().parents[1] / "prompts/realism_judge_prompt.txt",
            "r",
            encoding="utf-8",
        ) as f:
            prompt = f.read()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
        )

        text = response.text or ""
        match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", text)

        if not match:
            raise ValueError(f"Judge returned an invalid score: {text!r}")

        return int(match.group(1))

    def judge_fidelity(self, original_image_path, edited_image_path, prompt):

        with Image.open(original_image_path) as original, Image.open(
            edited_image_path
        ) as edited:
            response = self.client.models.generate_content(
                model=self.model_name, contents=[prompt, original, edited]
            )

        text = response.text or ""
        match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", text)

        if not match:
            raise ValueError(f"Fidelity judge returned an invalid score: {text!r}")

        return int(match.group(1))

    def detect_refusal(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name, contents=[prompt]
        )
        return response.text


class Qwen3:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.model_name = model_name
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name, device_map="auto"
        )

    def judge_realism(self, image_path):
        import torch

        try:
            image = Image.open(image_path)
            with open(
                Path(__file__).resolve().parents[1]
                / "prompts/realism_judge_prompt.txt",
                "r",
                encoding="utf-8",
            ) as f:
                prompt = f.read()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    stop_strings=["</OUTPUT>"],
                    tokenizer=self.processor.tokenizer,
                )

            generated = outputs[:, inputs["input_ids"].shape[-1] :]

            result = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", result, re.DOTALL)

            if match:
                score = match.group(1)

            return int(score)

        except Exception as e:
            print(str(e))

    def judge_fidelity(self, original_image_path, edited_image_path, prompt):
        import torch

        try:
            original_image = Image.open(original_image_path).convert("RGB")
            edited_image = Image.open(edited_image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": original_image},
                        {"type": "image", "image": edited_image},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    stop_strings=["</OUTPUT>"],
                    tokenizer=self.processor.tokenizer,
                )

            generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]

            result = self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", result, re.DOTALL)

            if match:
                score = match.group(1)

            return int(score)

        except Exception as e:
            print(str(e))
            return None

    def detect_refusal(self, prompt):
        import torch

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )

            generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]

            response = self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response.strip()

        except Exception as e:
            print(str(e))
            return None


class OpenAIJudge:
    """OpenAI-compatible Azure judge; deployment and endpoint are configuration."""

    def __init__(self, model_name, base_url=None, api_key=None):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=api_key or os.environ["Azure_API_KEY"],
            max_retries=0,
        )

    @staticmethod
    def _image(path):
        with Image.open(path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + encoded},
        }

    def _ask(self, prompt, paths=()):
        content = [{"type": "text", "text": prompt}] + [self._image(p) for p in paths]
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": content}]
        )
        return response.choices[0].message.content or ""

    def detect_refusal(self, prompt):
        return self._ask(prompt)

    @staticmethod
    def _score(text):
        match = re.search(r"<OUTPUT>\s*(\d+)\s*</OUTPUT>", text)
        if not match:
            raise ValueError(f"Judge returned an invalid score: {text!r}")
        return int(match.group(1))

    def judge_fidelity(self, original_image_path, edited_image_path, prompt):
        return self._score(self._ask(prompt, (original_image_path, edited_image_path)))

    def judge_realism(self, image_path):
        prompt = (
            Path(__file__).resolve().parents[1] / "prompts/realism_judge_prompt.txt"
        ).read_text(encoding="utf-8")
        return self._score(self._ask(prompt, (image_path,)))
