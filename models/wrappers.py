
from openai import OpenAI, APIStatusError
from typing import Any
import os
from io import BytesIO
from PIL import Image
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Modality, Part
import yaml
from pydantic import BaseModel
import base64
import random

class GeminiImageEditor:
    """
    Gemini model wrapper
    """

    def __init__(self, model_name: str, vertexai=True):
        self.model_name = model_name
        self.API_key = os.environ.get("GEMINI_API_KEY")
        if vertexai:
            self.client = genai.Client(
                vertexai=True,
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            self.client = genai.Client(api_key=self.API_key)

    def generate_edit(
        self, prompt: str, input_image_path: str, output_image_path: str
    ) -> dict[str, Any]:
        """
        Generate edit using Gemini API and save to output image path.
        Also return dictionary containing status, text output and edited image.
        """
        output_dict = {
            "status": "failed",
            "text_response": None,
            "image": None,
            "error": None,
        }

        with open(input_image_path, "rb") as f:
            img_bytes = f.read()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                prompt,
                Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/png",
                ),
            ],
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE]
            ),
        )
        feedback = response.prompt_feedback
        block_reason = feedback.block_reason if feedback else None
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else None
        output_dict["block_reason"] = getattr(block_reason, "value", block_reason)
        output_dict["finish_reason"] = getattr(finish_reason, "value", finish_reason)
        safety_reasons = {
            "SAFETY",
            "IMAGE_SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "IMAGE_PROHIBITED_CONTENT",
            "SPII",
        }

        output_dict["safety_block"] = (
            output_dict["block_reason"] in safety_reasons
            or output_dict["finish_reason"] in safety_reasons
        )
        text_parts = []
        if candidate:
            content = candidate.content
            if content and content.parts:
                for part in content.parts:
                    if part.text:
                        text_parts.append(part.text)

                    if part.inline_data:
                        with open(output_image_path, "wb") as f:
                            f.write(part.inline_data.data)

                        output_dict["image"] = Image.open(output_image_path).copy()
        if len(text_parts) > 0:
            output_dict["text_response"] = "\n".join(text_parts)
        output_dict["status"] = "success"

        return output_dict

    def generate_seed_instance(self, yaml_path):
        """
        Generate edit instance using Gemini API.
        Return: JSON format string
        """
        with open(
            "data_foundry/prompts/instance_JSON_prompt.txt", "r", encoding="utf-8"
        ) as f:
            prompt = f.read()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        categories = data["scene_categories"]
        category_name, category = random.choices(
            list(categories.items()),
            weights=[item["weight"] for item in categories.values()],
            k=1,
        )[0]
        scene = random.choice(category["scenes"])

        prompt = prompt.replace("{template}", str(data))
        prompt += (
            "\nThe category and scene have already been selected. "
            "Use these exact choices; do not select another:\n"
            f"scene_category: {category_name}\n"
            f"scene: {scene}\n"
        )

        class Appearance(BaseModel):
            camera: str
            lighting: str

        class SceneSpec(BaseModel):
            scene_type: str
            object: str
            location: str
            appearance: Appearance

        class EditSpec(BaseModel):
            attribute: str
            original_value: str
            target_value: str
            edit_type: str

        class Verification(BaseModel):
            method: str
            expected_result: str

        class BenchmarkJSON(BaseModel):
            subcategory_id: str
            scene_spec: SceneSpec
            edit_spec: EditSpec
            verification: Verification

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BenchmarkJSON,
            ),
        )

        return response.text

    def generate_edit_prompt(self, prompt):
        """
        Generate edit instance using Gemini API.
        Return: JSON format string
        """

        class PromptJSON(BaseModel):
            L0: str
            L1: str
            L2A: str
            L2B: str
            L3: list[str]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PromptJSON,
            ),
        )
        text = response.text

        if not text or not text.strip():
            feedback = response.prompt_feedback
            candidate = response.candidates[0] if response.candidates else None

            block_reason = feedback.block_reason if feedback else None
            finish_reason = candidate.finish_reason if candidate else None

            raise RuntimeError(
                "Edit-prompt generation returned no text. "
                f"block_reason={getattr(block_reason, 'value', block_reason)!r}, "
                f"finish_reason={getattr(finish_reason, 'value', finish_reason)!r}"
            )

        return text

    def generate_seed_image(self, prompt, output_path):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

        feedback = response.prompt_feedback
        candidate = response.candidates[0] if response.candidates else None

        block_reason = feedback.block_reason if feedback else None
        finish_reason = candidate.finish_reason if candidate else None

        block_reason = getattr(block_reason, "value", block_reason)
        finish_reason = getattr(finish_reason, "value", finish_reason)

        parts = candidate.content.parts or [] if candidate and candidate.content else []

        text_parts = []

        for part in parts:
            if part.text:
                text_parts.append(part.text)

            if part.inline_data:
                with Image.open(BytesIO(part.inline_data.data)) as image:
                    image.save(output_path)
                return

        raise RuntimeError(
            "Seed generation returned no image. "
            f"block_reason={block_reason!r}, "
            f"finish_reason={finish_reason!r}, "
            f"text={' '.join(text_parts)!r}"
        )


class _OpenAIImageBase:
    """Shared image operations for the OpenAI and Azure Image APIs."""

    @staticmethod
    def _image_from_response(response):
        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("Returned no image")

        image_bytes = base64.b64decode(response.data[0].b64_json)
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    @staticmethod
    def _safety_block_reason(error):
        """Return an explicit moderation code, or None."""
        safety_codes = {
            "content_policy_violation",
            "content_filter",
            "responsibleaipolicyviolation",
            "moderation_blocked",
        }

        code = getattr(error, "code", None)
        if isinstance(code, str) and code.lower() in safety_codes:
            return code

        body = error.body
        if not isinstance(body, dict):
            return None

        body = body.get("error", body)

        while isinstance(body, dict):
            code = body.get("code")
            if isinstance(code, str) and code.lower() in safety_codes:
                return code

            body = body.get("innererror") or body.get("inner_error")

        return None

    def generate_seed_image(self, prompt, output_path):
        response = self.client.images.generate(
            model=self.model_name,
            prompt=prompt,
            size="1024x1024",
        )
        self._image_from_response(response).save(output_path)

    def generate_edit(self, prompt, input_image_path, output_image_path):
        output = {
            "status": "failed",
            "text_response": None,
            "image": None,
            "error": None,
            "safety_block": False,
            "block_reason": None,
            "finish_reason": None,
        }

        try:
            with open(input_image_path, "rb") as source:
                response = self.client.images.edit(
                    model=self.model_name,
                    image=source,
                    prompt=prompt,
                )

            image = self._image_from_response(response)
            image.save(output_image_path)

            output["status"] = "success"
            output["image"] = image

        except APIStatusError as error:
            reason = self._safety_block_reason(error)
            if reason is None:
                raise
            output["safety_block"] = True
            output["block_reason"] = reason
            output["error"] = str(error)

        return output


class AzureImageEditor(_OpenAIImageBase):
    def __init__(
        self,
        model_name="gpt-image-2.5-sunburst-2026-09-08-dfg54",
    ):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            max_retries=0,
            api_key=os.environ["Azure_API_KEY"],
        )


class OpenAIImageEditor(_OpenAIImageBase):
    """Direct OpenAI Image API; shares image decoding and safety handling with Azure."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=0,
        )


GeminiModel = GeminiImageEditor
Azure_gpt_image = AzureImageEditor
OpenAIImage = OpenAIImageEditor

from models.judges import VLMJudge, Qwen3, OpenAIJudge
