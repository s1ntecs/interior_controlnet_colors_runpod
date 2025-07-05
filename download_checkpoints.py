# download_checkpoints.py  (offline-build)

import os
import subprocess
import time
import torch

from diffusers.pipelines.controlnet import \
    StableDiffusionControlNetInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline


from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LORA_NAMES = [
    "XSArchi_110plan彩总.safetensors",
    "XSArchi_137.safetensors",
    "XSArchi_141.safetensors",
    "XSArchi_162BIESHU.safetensors",
    "XSarchitectural-38InteriorForBedroom.safetensors",
    "XSarchitectural_33WoodenluxurystyleV2.safetensors",
    "house_architecture_Exterior_SDlife_Chiasedamme.safetensors",
    "xsarchitectural-15Nightatmospherearchitecture.safetensors",
    "xsarchitectural-18Whiteexquisiteinterior.safetensors",
    "xsarchitectural-19Houseplan (1).safetensors",
    "xsarchitectural-19Houseplan.safetensors",
    "xsarchitectural-7.safetensors",
    "Justin_Modern_interior_design_lora.safetensors",
    "LAVA_interior_design_lora.safetensors",
    "Oriental_interior_design_lora.safetensors",
    "Wooden_interior_design_lora.safetensors",
    "YAZI_interior_design_lora.safetensors",
    "archikitty_interior_design_lora.safetensors",
    "astra_modern_interior_design_lora.safetensors",
    "bedroom_interior_design_lora.safetensors",
    "children_interior_design_lora.safetensors",
    "children_room_interior_design_lora.safetensors",
    "cozy_interior_design_lora.safetensors",
    "cream_style_interior_design_lora.safetensors",
    "cyberpunk _interior_design_lora.safetensors",
    "dark_brown_interior_design_lora.safetensors",
    "dark_style_interior_design_lora.safetensors",
    "decoration _interior_design_lora.safetensors",
    "french_classic_interior_design_lora.safetensors",
    "french_interior_design_lora.safetensors",
    "futuristic_interior_design_lora.safetensors",
    "gothic _interior_design_lora.safetensors",
    "grey_interior_design_lora.safetensors",
    "indian_interior_design_lora.safetensors",
    "interior_design_lora.safetensors",
    "interior_design_lora_v2.safetensors",
    "japanes_interior_design_lora.safetensors",
    "justin_dark_interior_design_lora.safetensors",
    "luxury_Light_style_interior_design_lora.safetensors",
    "mid_century_interior_design_lora.safetensors",
    "minimalist_interior_design_lora.safetensors",
    "modern_interior_design_lora.safetensors",
    "neoclassic_interior_design_lora.safetensors",
    "neoclassic_wood_interior_design_lora.safetensors",
    "nordic_lux_interior_design_lora.safetensors",
    "retro_interior_design_lora.safetensors",
    "retrofuturistic_interior_design_lora.safetensors",
    "scandinavian_interior_design_lora.safetensors",
    "sunroom_interior_design_lora.safetensors",
    "tropical_brutalism_interior_design_lora.safetensors",
    "woven_interior_design_lora.safetensors"
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""
    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras",
            local_dir_use_symlinks=False,
        )


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = [
        ControlNetModel.from_pretrained(
            "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        ),
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        # "SG161222/Realistic_Vision_V3.0_VAE",
        "hafsa000/interior-design",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    MLSDdetector.from_pretrained("lllyasviel/Annotators")

    return pipe


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
