import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from termcolor import cprint
from typing import Callable, Optional, Any, List
from pathlib import Path

from Env_Config.Qwen.client import QwenClient
from Env_Config.Qwen.sam2_image import processor_sam2image
import prompt

def solver (
    camera,
    img_path : str,
    instruction : str = "",
    prompt_type : str = "default",
    use_depth_info : bool = False,
    use_sam2 : bool = True,
    exist_masks : bool = False,
    labels : np.ndarray = None,
    masks : np.ndarray = None,
    last_idx : int = -1,
    CoT_type : str = None,
):
    client = QwenClient ()
    if use_sam2 == True:
        sam2_labels, sam2_masks = processor_sam2image (
            camera = camera,
            img_path = img_path,
            use_depth_info = use_depth_info,
        )
        print ("[OK] sam2 processor.")

        if exist_masks:
            masks = np.concatenate ([masks, sam2_masks], axis = 0)
            labels = np.concatenate ([labels, sam2_labels], axis = 0)
        else:
            masks = sam2_masks
            labels = sam2_labels

    img = np.array (Image.open(img_path).convert ('RGB'))
    h, w = img.shape[:2]

    text = prompt.GenPrompt (
        [h, w], labels, last_idx, instruction,
        prompt_type = prompt_type,
        CoT_type = CoT_type
    )

    img_paths = [
        # img_path,
        Path (img_path).with_name (f"{Path (img_path).stem}.label{Path (img_path).suffix}"),
    ]
    if prompt_type == "check_if_masks_need_regen":
        img_paths.append (
            Path (img_path).with_name (f"{Path (img_path).stem}.mask&label{Path (img_path).suffix}")
        )
    if prompt_type == "check_if_need_rightarm":
        img_paths.append (
            os.path.dirname (img_path) + '/current_scene.label.jpg'
        )

    if prompt_type.startswith ("b1_"):
        img_paths= [img_path]
        if prompt_type == "b1_check_if_need_rightarm":
            img_paths.append (
                os.path.dirname (img_path) + '/current_scene.jpg'
            )

    full_ans, x, y = client.ask (text, img_paths)
    with open(os.path.dirname(img_path) + '/.full_ans.txt', "w", encoding="utf-8") as f:
        f.write (full_ans)
    
    if prompt_type == "check_if_multiple_picked_garments":
        return bool (x > 0 or y > 0)

    idx = -1
    if labels is not None:
        for i, label in enumerate(labels):
            if label[0] == x and label[1] == y:
                idx = i
                cprint (f"qwen selected idx = {i + 1}", "cyan")

    # assert idx != -1, "Selection point is not a numbered marker"
    return x, y, idx, masks