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


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = os.path.join(os.path.dirname(__file__), "sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

single_predictor = SAM2ImagePredictor (
    sam_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
)

auto_predictor = SAM2AutomaticMaskGenerator (
    model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False),
    points_per_side = 32,
    points_per_batch = 64,
    pred_iou_thresh = 0.85,
    box_nms_thresh = 0.35,
    stability_score_thresh = 0.65,
    crop_n_layers = 2,
    crop_n_points_downscale = 2,
    crop_nms_thresh = 0.45,
    crop_overlap_ratio = 0.25,
)


# ----- #

def mask_nms (anns, iou_thresh = 0.5):
    anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    kept = []
    for ann in anns:
        mask = ann["segmentation"]
        if all (
            (np.logical_and (mask, km["segmentation"]).sum () /
                np.logical_or(mask, km["segmentation"]).sum ()
            ) < iou_thresh
            for km in kept
        ):
            kept.append (ann)
    return kept

def sorted_by_manhattan(h, w):
    ys = np.arange(0, h)
    xs = np.arange(0, w)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    cy, cx = h/2.0, w/2.0
    D = np.abs(Y - cy) + np.abs(X - cx)
    idx = np.argsort(D.ravel(), kind='stable')
    coords = np.stack([Y.ravel()[idx], X.ravel()[idx]], axis=1)
    return coords

def processor_sam2image (
    camera,
    img_path : str,
    only_mask : bool = False,
    use_depth_info : bool = False,
    use_auto_predictor : bool = True,
    exist_retained_mask : bool = False,
    retained_mask : np.ndarray = None,
    if_pass : bool = False,
    save_type : int = 0,
):
    img = np.array (Image.open(img_path).convert ('RGB'))
    h, w = img.shape[:2]
    cprint (f"processor: h, w = ({h}, {w})", "cyan")

    FRAGMENT = 0.005 * h * w

    masks = []
    label = np.zeros ((h, w))
    covered = np.zeros ((h, w))
    idx = 1
    
    if use_auto_predictor == True:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = auto_predictor.generate (img)
        result = mask_nms (result)

        for i in range (len (result)):
            mask = result[i]['segmentation']
            covered += mask
            mask[label > 0] = 0
            if np.sum (mask) > FRAGMENT:
                label[mask == 1] = idx
                idx += 1
                masks.append (mask)

    if exist_retained_mask == True:
        for i, mask in enumerate (retained_mask):
            covered += mask
            mask[label > 0] = 0
            if np.sum (mask) > FRAGMENT:
                label[mask == 1] = idx
                idx += 1
                masks.append (mask)
    
    manual_point = []
    incomplete_mask = masks

    if save_type == 2:
        if_pass = True
        
    if if_pass == False:
        single_predictor.set_image(img)

        coords = sorted_by_manhattan(h, w)
        def check(x, y):
            y0 = max(y - 5, 0)
            y1 = min(y + 6, h)
            x0 = max(x - 5, 0)
            x1 = min(x + 6, w)
            return np.sum (covered[y0 : y1, x0 : x1]) == 0

        vcheck = np.vectorize(lambda y, x: check(x, y))
        coords = coords[vcheck(coords[:, 0], coords[:, 1])]

        while coords.shape[0] > 0:
            y, x = coords[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mask, _, _ = single_predictor.predict(
                    point_coords=np.array ([[x + 1, y + 1]]),
                    point_labels=np.array ([1]),
                    multimask_output=False,
                )
            mask = mask[0]
            mask[y, x] = 1
            mask[label > 0] = 0
            covered += mask
            if np.sum (mask) > FRAGMENT:
                manual_point.append (np.array ([x + 1, y + 1]))
                label[mask == 1] = idx
                idx += 1
                masks.append (mask)

            vcheck = np.vectorize(lambda y, x: check(x, y))
            coords = coords[vcheck(coords[:, 0], coords[:, 1])]

    span = []
    span_d = [np.array ([])]
    for d in range (1, h + w):
        border = []
        for dx in range(-d, d+1):
            dy = d - abs(dx)
            for sgn in (-1, 1) if dy>0 else (0,):
                border.append (np.array ([dx, dy * sgn]))
        span += border
        span_d.append (np.array (border, dtype = int))

    if if_pass == False:
        fix_board = np.zeros ((h, w))
        for y in range (h):
            for x in range (w):
                if label[y, x] > 0:
                    continue
                for dx, dy in span:
                    nx, ny = x + dx, y + dy
                    if 0 < nx < w and 0 < ny < h and label[ny, nx] > 0:
                        fix_board[y, x] = label[ny, nx]
                        break
                masks[int(fix_board[y, x]) - 1][y, x] = 1
        label += fix_board


    GROUND = 0.35 * h * w
    TIDY_RATIO = 0.4

    def check_span (mask : np.ndarray):
        n, m = mask.shape
        mi_y = np.where (mask, np.arange (m), m).min (axis = 1)
        mx_y = np.where (mask, np.arange (m), -1).max (axis = 1)
        mi_x = np.where (mask, np.arange (n)[ : , None], n).min (axis = 0)
        mx_x = np.where (mask, np.arange (n)[ : , None], -1).max (axis = 0)
        spans = min (
            np.maximum (0, (mx_y - mi_y + 1)).sum (),
            np.maximum (0, (mx_x - mi_x + 1)).sum ()
        )
        # cprint (f"spans ratio: {1.0 * spans / mask.sum ()}", "cyan")
        return spans > (1 + TIDY_RATIO) * mask.sum ()

    def is_background (mask : np.ndarray):
        area = mask.sum ()
        assert mask.ndim == 2, "mask wrong!"
        assert area > 0, "WTF the empty mask?"
        if area > GROUND:
            return 1

        mask = np.asarray (mask, dtype = bool)
        labeled, num = cc_label (mask, structure = np.ones ((3, 3), dtype = bool))

        max_area = 0
        for i in range (1, num + 1):
            current_area = np.sum (labeled == i)
            max_area = max (max_area, current_area)
            if current_area <= area * 0.3:
                continue
            if check_span (labeled == i) == True:
                return 1

        if max_area <= area * 0.3:
            return 1

        if use_depth_info == True:
            mask_pc, mask_pc_colors = camera.get_pointcloud_from_depth (
                sample_flag = False,
                object_mask_or_not = True,
                object_mask = mask
            )
            print (f"sam2 check mask: {mask.shape}, {area}, {mask_pc.shape}")
            if mask_pc.shape[0] == 0: return 1
            
            z = np.sort (mask_pc[ : , 2])
            n = z.shape[0]

            mi, mx = z[int (n * 0.15)], z[min (int (n * 0.85), n - 1)]
            dec = mx - mi
            cprint (f"  sam2 => dec = {dec}, mi = {mi}, mx = {mx}", "cyan")

            if dec < 0.01: return 1
            
            camera.get_point_cloud_data_from_segment (
                save_or_not = False,
                sample_flag = False,
                real_time_watch = False
            )
            lookup = set (map (tuple, np.round (camera.point_cloud.astype (np.float32), 1)))
            in_mask = np.array ([i for i, b in enumerate (np.round (mask_pc.astype (np.float32), 1)) if tuple (b) in lookup], dtype = int)
            cprint (f"  sam2 => in_mask = {in_mask.shape[0]} / {np.sum (mask)} = {1.0 * in_mask.shape[0] / np.sum (mask)}", "cyan")
            # camera.show_pointcloud (camera.point_cloud, camera.colors)
            # camera.show_pointcloud (mask_pc, mask_pc_colors)
            
            # if in_mask.shape[0] < np.sum (mask) * 0.9: # empty
            #     return 1

        return 0


    N_ROUNDS = 30
    y_pad = int (0.005 * h)
    x_pad = int (0.005 * w)

    labels = []
    final_masks = []

    for i in range (len (masks)):
        check_code = is_background (masks[i])
        if check_code == 1:
            continue

        if only_mask == False:
            coords = np.argwhere(masks[i][y_pad : h - y_pad, x_pad : w - x_pad] == 1)
            sel_idx = np.random.choice(len(coords), size=N_ROUNDS)
            ys = y_pad + coords[sel_idx, 0]
            xs = x_pad + coords[sel_idx, 1]
            done = np.zeros (N_ROUNDS, dtype = bool)
            dists = np.zeros (N_ROUNDS, dtype = int)

            for d in range (1, h + w):
                dxs, dys = span_d[d][ : , 0], span_d[d][ : , 1]
                Xn = xs[:, None] + dxs[None, :]
                Yn = ys[:, None] + dys[None, :]
                valid = (0 <= Xn) & (Xn < w) & (0 <= Yn) & (Yn < h)
                Xn = np.clip(Xn, 0, w-1)
                Yn = np.clip(Yn, 0, h-1)
                newly = np.any((label[Yn, Xn] != (i + 1)) & valid, axis=1) & (~ done)
                if not newly.any(): continue
                dists[newly] = d
                done[newly] = True
                if done.all(): break

            best_round = np.argmax(dists)
            labels.append ([xs[best_round] + 1, ys[best_round] + 1] + list (
                (np.sum(img.transpose(2, 0, 1) * masks[i], axis = (1, 2)) / np.sum (masks[i])).astype (int)
            ))

        final_masks.append (masks[i])

    labels = np.array (labels)
    final_masks = np.array (final_masks, dtype = bool)

    dpi = 100
    black = np.array([0, 0, 0])

    if only_mask == False:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w / dpi, h / dpi)
        fig.set_dpi(dpi)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        ax.imshow(img)
        for i, coord in enumerate(labels):
            show_point(coord, ax, black, f"${i+1}$")
        for i, mask in enumerate (final_masks):
            show_border(mask, ax)

        out_path = Path(img_path).with_name(f"{Path(img_path).stem}.label{Path(img_path).suffix}")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(w / dpi, h / dpi)
        fig.set_dpi(dpi)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        ax.imshow(img)
        for i, coord in enumerate(labels):
            show_point(coord, ax, black, f"${i+1}$")
        for i, mask in enumerate (final_masks):
            show_mask(mask, ax)

        out_path = Path(img_path).with_name(f"{Path(img_path).stem}.mask&label{Path(img_path).suffix}")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / dpi, h / dpi)
    fig.set_dpi(dpi)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.imshow(img)
    for i, mask in enumerate (final_masks):
        show_mask(mask, ax)

    out_path = Path(img_path).with_name(f"{Path(img_path).stem}.mask{Path(img_path).suffix}")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / dpi, h / dpi)
    fig.set_dpi(dpi)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.imshow(img)
    for i, mask in enumerate (final_masks):
        show_border(mask, ax)

    out_path = Path(img_path).with_name(f"{Path(img_path).stem}{Path(img_path).suffix}")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    if only_mask == False:
        return labels, final_masks
    else:
        return final_masks