import numpy as np
from termcolor import cprint


def is_yellow(pixel):
    r,g,b=pixel
    if 150<= r <= 255 and 80 <= g <= 255 and b <= 100:
        return True
    return False


def yellow_pixel_ratio(image: np.ndarray,boundary:list) -> float:
    """
    Calculate the ratio of yellow pixels within a specified range in the image.

    Parameters:
    - image (np.ndarray): Input image data with shape (height, width, channels).
    - boundary (list):[x_start,x_end,y_start,y_end]
        x_start (int): Starting x-coordinate of the region.
        x_end (int): Ending x-coordinate of the region.
        y_start (int): Starting y-coordinate of the region.
        y_end (int): Ending y-coordinate of the region.

    Returns:
    - float: Ratio of yellow pixels within the specified range.
    """
    if len(boundary)!=4:
        raise ValueError("must be x_start x_end y_start y_end!")
    
    inside = 0
    outside = 0
    # Ensure the specified range is within the image boundaries
    y_start = max(0, boundary[2])
    y_end = min(image.shape[0], boundary[3])
    x_start = max(0, boundary[0])
    x_end = min(image.shape[1], boundary[1])
    
    # Iterate through each pixel in the specified range
    # Iterate through each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]
            if is_yellow(pixel):    
                if y_start <= y < y_end and x_start <= x < x_end:
                    inside += 1
                else:
                    outside += 1
    total=inside+outside
    if total== 0:
        return 0.0,total
    
    return inside / total,total

def judge_store(image: np.ndarray, boundary:list=[260,380,302,392],threshold=0.12):
    proportion,_=yellow_pixel_ratio(image,boundary)
    return proportion >=(1-threshold)

def judge_pcd(pcd:np.ndarray,boundary:list,threshold=0.1):
    x0, x1, y0, y1 = boundary
    x_coords = pcd[:, 0]
    y_coords = pcd[:, 1]
    
    within_x = (x_coords >= x0) & (x_coords <= x1)
    within_y = (y_coords >= y0) & (y_coords <= y1)
    within_boundary = within_x & within_y
    
    proportion = np.sum(within_boundary) / pcd.shape[0]
    
    cprint("----------- Judge Begin -----------", color="blue", attrs=["bold"])
    cprint(f"in_domain points: {np.sum(within_boundary)}, all_points: {pcd.shape[0]}", color="blue")
    cprint(f"proportion (executed / demo): {proportion}", color="blue")
    cprint("----------- Judge End -----------", color="blue", attrs=["bold"])
    
    return proportion>=(1-threshold)
    
