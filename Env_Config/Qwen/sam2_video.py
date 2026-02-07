import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from termcolor import cprint
from typing import Callable, Optional, Any, List
from scipy.ndimage import label as cc_label

from pathlib import Path
import shutil
import subprocess

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, borders = True):
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = np.concatenate([np.random.random(3), [0.5]])
    # color_mask = np.array([0.0, 0.0, 0.0, 0.8])
    img[mask] = color_mask
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        cv2.drawContours(img, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(img)

def show_border(mask, ax):
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = np.array([0.0, 0.0, 0.0, 0.0])
    img[mask] = color_mask
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
    cv2.drawContours(img, contours, -1, (1, 1, 1, 0.5), thickness=3)
    ax.imshow(img)

from matplotlib.colors import to_rgb
def show_point(coord, ax, rgb, marker, marker_size=500):
    ax.scatter(coord[0], coord[1], color=to_rgb((0, 0, 0)), marker = 's', s = marker_size * 2, edgecolor = None, linewidth = 0.0)
    ax.scatter(coord[0], coord[1], color=to_rgb(tuple(c/255 for c in rgb)),
        marker=marker, s=marker_size, edgecolor='white', linewidth=3)

from Env_Config.Qwen.sam2_image import processor_sam2image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = os.path.join(os.path.dirname(__file__), "sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def processor_sam2video (
    camera,
    video_path : str,
    retained_masks : np.ndarray,
    use_depth_info : bool = False,
    pick_point : np.ndarray = np.array ([0, 0]),
    only_mask : bool = False,
    use_auto_predictor : bool = False,
    save_type : int = 0,
):
    video_path = Path (video_path)
    video_dir = video_path.parent / "result"

    shutil.rmtree (video_dir, ignore_errors = True)
    video_dir.mkdir (parents = True, exist_ok = True)

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", str (video_path),
        "-q:v", "2",
        "-start_number", "0",
        "-vf", "fps=2",
        str (video_dir / "%05d.jpg"),
    ]
    subprocess.run(
        cmd,
        check = True,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path = str (video_dir))
    predictor.reset_state (inference_state)
    
    img = np.array (Image.open(video_dir/frame_names[0]).convert ('RGB'))
    h, w = img.shape[:2]
    dpi = 100
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / dpi, h / dpi)
    fig.set_dpi(dpi)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.imshow(img)
    green = np.array([0, 255, 0])

    out_masks = []
    if retained_masks.shape[0] > 0: # with mask prompt
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i, mask in enumerate (retained_masks):
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state = inference_state,
                    frame_idx = 0,
                    obj_id = i,
                    mask = mask
                )
                show_mask (mask, ax)
                
    if pick_point[0] + pick_point[1] > 0:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state = inference_state,
                frame_idx = 0,
                obj_id = len (retained_masks),
                points = np.array ([pick_point], dtype = np.float32),
                labels = np.array ([1], np.int32)
            )
            show_point (pick_point, ax, green, "*")

    if save_type == 2:
        img_file = "picked_scene.jpg"
    else:
        img_file = "current_scene.jpg"
    
    shutil.copy (
        video_dir / f"{frame_names[len (frame_names) - 1]}",
        video_path.parent / img_file
    )
    
    if len (retained_masks) + int (pick_point[0] + pick_point[1] > 0) > 0: # need propagate
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            for _, out_mask in video_segments[len (frame_names) - 1].items():
                out_masks.append (out_mask if out_mask.ndim == 2 else out_mask[0])

    return processor_sam2image (
        camera = camera,
        img_path = video_path.parent / img_file,
        only_mask = only_mask,
        use_depth_info = use_depth_info,
        use_auto_predictor = use_auto_predictor,
        exist_retained_mask = bool (len (out_masks) > 0),
        retained_mask = np.array (out_masks),
        save_type = save_type,
    )