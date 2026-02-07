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

def solver_4stir (
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

    full_ans = client.ask_4stir (text, img_paths)
    with open(os.path.dirname(img_path) + '/.full_ans.txt', "w", encoding="utf-8") as f:
        f.write (full_ans)

    def read_last_line_and_convert_to_list(file_path):
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            while f.tell() > 0 and f.read(1) != b'\n':
                f.seek(f.tell() - 2, os.SEEK_SET)
            last_line = f.readline().decode('utf-8').strip()
        import re
        last_line = re.sub(r'^Final Answer:\s*', '', last_line, flags=re.IGNORECASE)
        parts = [part.strip() for part in last_line.split(',')]
        number_list = [int(part) for part in parts]
        return number_list

    idx_list = read_last_line_and_convert_to_list(os.path.dirname(img_path) + '/.full_ans.txt')
    idx_choosed = []
    for i in idx_list:
        if i >= 1 and i <= len (masks):
            idx_choosed.append (i - 1)
            print (f"qwen: mask_{i} need to regen!")
    idx_choosed = np.array (idx_choosed)
    return idx_choosed, masks
