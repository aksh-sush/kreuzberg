from pathlib import Path

import pytest
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from kreuzberg._vision import QWEN_2_5_VL_3_B, extract_text_and_layout


@pytest.mark.anyio
async def test_extract_text_and_layout(ocr_image: Path) -> None:
    result = await extract_text_and_layout(model_id=QWEN_2_5_VL_3_B, image_path=ocr_image)
    assert result == ""


def test_trocr(ocr_image: Path) -> None:
    image = Image.open(ocr_image).convert("RGB")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    assert generated_text
