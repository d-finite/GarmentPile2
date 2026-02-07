


import open3d as o3d
import numpy as np
from termcolor import cprint


def is_yellow(pixel):
    r,g,b=pixel
    if 150<= r <= 255 and 80 <= g <= 255 and b <= 100:
        return True
    return False

def count_pixels(image_array):
    height, width, _ = image_array.shape
    
    _count = 0
    
    for x in range(height):
        for y in range(width):
            if is_yellow(image_array[x, y]):
                _count += 1
    
    return _count

def calculate_garment_proportion(image:np.ndarray)->float:
    """calculate proportion of yellow garment in image

    Args:
        image (np.ndarray): image

    Returns:
        float: proportion
    """
    if len(image.shape)!=3 or image.shape[2]!=3:
        raise ValueError("must be RGB image!")

    yellow_pixels=count_pixels(image)
    
    total_pixels = image.shape[0] * image.shape[1]
    
    percentage = (yellow_pixels / total_pixels) * 100

    return percentage,yellow_pixels


def compute_plane_model(pc:o3d.geometry.PointCloud, distance_threshold=0.005, ransac_n=3, num_iterations=1000)->list:
    """
    Computes the plane model of a point cloud using the RANSAC algorithm.

    Parameters:
    - pc: open3d.geometry.PointCloud object
    - distance_threshold: Distance threshold to determine inliers in RANSAC
    - ransac_n: Number of points to sample for generating a plane in each RANSAC iteration
    - num_iterations: Number of iterations for the RANSAC algorithm

    Returns:
    - plane_model: Plane parameters [a, b, c, d] of the fitted plane
    """
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                           ransac_n=ransac_n,
                                           num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    # print(f"Fitted plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    return plane_model

def compute_fit_error(pc:o3d.geometry.PointCloud, plane_model:list)->float:
    """
    Computes the standard deviation of the fit error, which is the distance of each point to the plane.

    Parameters:
    - pc: open3d.geometry.PointCloud object
    - plane_model: Plane parameters [a, b, c, d]

    Returns:
    - error_std: Standard deviation of the fit error
    """
    [a, b, c, d] = plane_model
    points = np.asarray(pc.points)
    # Calculate the perpendicular distance from each point to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    # Compute the standard deviation of the distances
    error_std = np.std(distances)
    return error_std
    
def judge_fling(image_judge,image_end,threshold=0.2)->bool:
    image_judge=calculate_garment_proportion(image_judge)
    image_end=calculate_garment_proportion(image_end)
    cprint("----------- Judge Begin -----------", color="blue", attrs=["bold"])
    cprint(f"demo_garment_cover: {image_judge[0]:.2f}, executed_garment_cover: {image_end[0]:.2f}", color="blue")
    cprint(f"proportion (executed / demo): {image_end[0]/image_judge[0]}", color="blue")
    cprint("----------- Judge End -----------", color="blue", attrs=["bold"])
    if image_judge[0]*(1-threshold)<image_end[0]:
        return True
    return False

def judge_fling_PFE(pc_end:np.ndarray,pc_judge:np.ndarray,tolerance = 0.15)->bool:
    pc_0=o3d.geometry.PointCloud()
    pc_1=o3d.geometry.PointCloud()
    pc_0.points = o3d.utility.Vector3dVector(pc_judge)
    pc_1.points = o3d.utility.Vector3dVector(pc_end)
    
    # Compute the reference plane model from pc_judge
    plane_model=compute_plane_model(pc_0)
    
    # Compute fit error for pc_judge,pc_end
    error_0=compute_fit_error(pc_0,plane_model)
    error_1=compute_fit_error(pc_1,plane_model)
    
    # Compare the fit errors
    if error_1<=error_0*(1+tolerance):
        return True
    return False

def judge_fling_full(image_judge,image_begin,image_end,threshold=0.15,tolerance=0.15)->bool:
    """
    Judges the flatness of pc_end compared to pc_judge by combining garment proportion and plane fitting error.
    

    Parameters:
    - image_judge (tuple): Tuple containing (image, pc_judge as NumPy array of shape (N, 3)).
    - image_begin (tuple): Tuple containing (image, pc_begin as NumPy array of shape (N, 3)).
    - image_end (tuple): Tuple containing (image, pc_end as NumPy array of shape (N, 3)).
    - threshold (float): Threshold for the combined score to decide flatness.
    - tolerance (float): Tolerance factor for individual metric comparisons.

    Returns:
    - bool: True if garment_end's flatness is comparable to or better than garment_judge's, False otherwise.
    """
    
    image_judge,pc_judge=image_judge
    image_begin,_=image_begin
    image_end,pc_end=image_end
    
    
    flag0=judge_fling(image_judge,image_begin,image_end,threshold)
    flag1=judge_fling_PFE(pc_end,pc_judge,tolerance)
    
    return flag0 and flag1

