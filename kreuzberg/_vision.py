from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.modeling_utils import SpecificPreTrainedModelType

DEFAULT_DIR: Final[Path] = Path(__file__).parent.parent / "models"
DEFAULT_PROMPT: Final[str] = """Extract all text from this image and convert it to markdown."""

QWEN_2_5_VL_7_B: Literal["Qwen/Qwen2.5-VL-7B-Instruct"] = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_2_5_VL_3_B: Literal["Qwen/Qwen2.5-VL-3B-Instruct"] = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL = Literal["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]

TR_OCR_SMALL_HANDWRITTEN: Literal["microsoft/trocr-small-handwritten"] = "microsoft/trocr-small-handwritten"

SupportedModels = Literal[
    "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "microsoft/trocr-small-handwritten"
]


async def get_qwen_model(
    model_id: QWEN_MODEL = QWEN_2_5_VL_3_B,
    is_cpu: bool = True,
) -> tuple[SpecificPreTrainedModelType, AutoProcessor]:
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if is_cpu else "flash_attention_2",
        device_map="mps",
        low_cpu_mem_usage=is_cpu,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


async def extract_text_and_layout(
    *,
    image_path: Path,
    model_id: str,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """Extract text and layout information from images using transformers.

    Args:
        image_path: Path to the image file or list of paths for multiple images
        model_id: The Hugging Face model ID
        prompt: The prompt to guide text extraction

    Returns:
        Extracted text content from the image(s)
    """
    # Map execution_provider to torch device
    model, processor = await get_qwen_model(model_id)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8096)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    outputs = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return outputs[0].removeprefix("```markdown").removesuffix("```")
