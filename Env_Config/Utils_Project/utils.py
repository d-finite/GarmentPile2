import numpy as np
import os

import torch
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from isaacsim.core.utils.prims import delete_prim, set_prim_visibility
from termcolor import cprint
from plyfile import PlyData, PlyElement
from PIL import Image
import math


def get_unique_filename(base_filename, extension=".png", counter_return=False):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"

    if extension == ".ply":
        return filename, counter
    if counter_return:
        return filename, counter
    else:
        return filename


def record_success_failure(flag: bool, file_path, str=""):

    with open(file_path, "rb") as file:
        file.seek(0, 2)
        file_empty = file.tell() == 0
        if not file_empty:
            file.seek(-1, 2)
            last_char = file.read(1)
    if file_empty or last_char != b"\n":
        if flag:
            print("write success")
            with open(file_path, "a") as file:
                file.write("1 " + "success" + "\n")
        else:
            print("write failure")
            with open(file_path, "a") as file:
                file.write("0 " + str + "\n")
    else:
        print("No writing")
        return


def read_ply(filename):
    """read XYZ point cloud from filename PLY file"""
    plydata = PlyData.read(filename)
    pc = plydata["vertex"].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def read_ply_with_colors(filename):
    plydata = PlyData.read(filename)
    pc = plydata["vertex"].data
    pc_array = np.array([[x, y, z] for x, y, z, r, g, b in pc])
    colors = np.array([[r, g, b] for x, y, z, r, g, b in pc])
    return pc_array, colors


def write_ply(points, filename):
    """
    save 3D-points and colors into ply file.
    points: [N, 3] (X, Y, Z)
    colors: [N, 3] (R, G, B)
    filename: output filename
    """
    # combine vertices and colors
    vertices = np.array(
        [tuple(point) for point in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    el = PlyElement.describe(vertices, "vertex")

    # save PLY file
    PlyData([el], text=True).write(filename)


def write_ply_with_colors(points, colors, filename):
    """
    save 3D-points and colors into ply file.
    points: [N, 3] (X, Y, Z)
    colors: [N, 3] (R, G, B)
    filename: output filename
    """
    # combine vertices and colors
    colors = colors[:, :3]
    vertices = np.array(
        [tuple(point) + tuple(color) for point, color in zip(points, colors)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    # create PlyElement
    el = PlyElement.describe(vertices, "vertex")

    # save PLY file
    PlyData([el], text=True).write(filename)


def compare_position_before_and_after(pre_poses, cur_poses, index):
    nums = 0
    for i in range(len(pre_poses)):
        if i == index:
            continue
        dis = torch.norm(cur_poses[i] - pre_poses[i]).item()

        if dis > 0.2:
            nums += 1
    print(f"{nums} garments changed a lot")
    return nums


def judge_once_per_time(cur_poses, index):
    nums = 1
    for i in range(len(cur_poses)):
        if i == index:
            continue
        dis = torch.norm(cur_poses[i] - cur_poses[index]).item()
        if dis < 0.4:
            nums += 1
    print(f"pick {nums} of garments once")
    return nums


def table_judge_final_poses(
    stage, position, index, garment_index
):
    flg = True
    for i in range(len(garment_index)):
        if i == index:
            cprint (f"garment_{i} position: {position[i]}", "cyan")
            ub = np.array([0.29895124, 1.87372363, 1.85835081])
            lb = np.array([-0.32654363, 1.46116352, 0.0])
            points = get_all_mesh_points(stage, f"/World/Garment/garment_{i}/mesh")
            assert points.shape[0] > 0, "empty points"
            mask = np.logical_and(points>lb, points<ub).all(axis=-1)
            cprint (f"[IN-BASKTER RATIO]: {mask.sum() / points.shape[0]}", "cyan")
            if(mask.sum() / points.shape[0] < 0.8):
                flg = False
                # record_success_failure(False, save_path, "final pose not correct")

            delete_prim(f"/World/Garment/garment_{index}")
            cprint (f"delete_prim garment_{i} [index]", "cyan")
            garment_index[i] = False
        elif garment_index[i]:
            if (
                position[i][2] < 0.66
                or position[i][2] > 1.0
                or position[i][1] > 2.8
                or position[i][1] < 1.85
                or position[i][0] > 0.6
                or position[i][0] < -0.6
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                cprint (f"delete_prim garment_{i}", "cyan")
                garment_index[i] = False

    # record_success_failure(True, save_path, " success")

    return garment_index, flg


def open_scene_judge_final_all_poses(
    stage, position, config, garment_index,
):
    suc = 0
    check_code = np.zeros ((len (garment_index)))
    for i in range(len(garment_index)):
        if garment_index[i] == False:
            check_code[i] = -2
            continue 
        
        if (position[i][0] > - 0.6 and position[i][0] < 0.6) and \
            (position[i][1] > 1.85 and position[i][1] < 2.8) and \
            (position[i][2] > 0.66 and position[i][2] < 1.0): # in initial basket
                continue
        
        cprint (f"garment_{i} position: {position[i]}", "cyan")
        ub = np.array([0.29895124, 1.87372363, 1.05835081])
        lb = np.array([-0.32654363, 1.46116352, 0.0])
        points = get_all_mesh_points(stage, f"/World/Garment/garment_{i}/mesh")
        assert points.shape[0] > 0, "empty points"
        mask = np.logical_and(points>lb, points<ub).all(axis=-1)
        cprint (f"[IN-BASKTER RATIO]: {mask.sum() / points.shape[0]}", "cyan")
        
        if(mask.sum() / points.shape[0] >= 0.2):
            suc += bool (mask.sum() / points.shape[0] >= 0.8)
            check_code[i] = bool (mask.sum() / points.shape[0] >= 0.8)
            delete_prim(f"/World/Garment/garment_{i}")
            cprint (f"delete_prim garment_{i} [index]", "cyan")
            garment_index[i] = False
        
        else:
            if (
                position[i][2] < 0.66
                or position[i][2] > 1.0
                or position[i][1] > 2.8
                or position[i][1] < 1.85
                or position[i][0] > 0.6
                or position[i][0] < -0.6
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                cprint (f"delete_prim garment_{i}", "cyan")
                check_code[i] = -1
                garment_index[i] = False

    return garment_index, suc, check_code


def closed_scene_judge_final_all_poses(
    stage, position, config, garment_index,
):
    suc = 0
    check_code = np.zeros ((len (garment_index)))
    for i in range(len(garment_index)):
        if garment_index[i] == False:
            check_code[i] = -2
            continue 
        
        if (position[i][0] > - 0.4 and position[i][0] < 0.4) and \
            (position[i][1] > 1.7 and position[i][1] < 2.5) and \
            (position[i][2] > 0.5 and position[i][2] < 0.9): # in initial basket
                continue
        
        cprint (f"garment_{i} position: {position[i]}", "cyan")
        ub = np.array([0.29895124, 1.87372363, 1.05835081])
        lb = np.array([-0.32654363, 1.46116352, 0.0])
        points = get_all_mesh_points(stage, f"/World/Garment/garment_{i}/mesh")
        assert points.shape[0] > 0, "empty points"
        mask = np.logical_and(points>lb, points<ub).all(axis=-1)
        cprint (f"[IN-BASKTER RATIO]: {mask.sum() / points.shape[0]}", "cyan")
        
        if(mask.sum() / points.shape[0] >= 0.2):
            suc += bool (mask.sum() / points.shape[0] >= 0.8)
            check_code[i] = bool (mask.sum() / points.shape[0] >= 0.8)
            delete_prim(f"/World/Garment/garment_{i}")
            cprint (f"delete_prim garment_{i} [index]", "cyan")
            garment_index[i] = False
        
        else:
            if (
                position[i][2] < 0.45
                or position[i][2] > 1.2
                or position[i][1] > 2.5
                or position[i][1] < 1.7
                or position[i][0] > 0.43
                or position[i][0] < -0.43
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                cprint (f"delete_prim garment_{i}", "cyan")
                check_code[i] = -1
                garment_index[i] = False

    return garment_index, suc, check_code


def basket_judge_final_poses(
    stage, position, index, garment_index, save_path: str = "basket_record.txt"
):
    flg = True
    for i in range(len(garment_index)):
        if i == index:
            cprint (f"garment_{i} position: {position[i]}", "cyan")
            ub = np.array([0.29895124, 1.87372363, 1.05835081])
            lb = np.array([-0.32654363, 1.46116352, 0.0])
            points = get_all_mesh_points(stage, f"/World/Garment/garment_{i}/mesh")
            assert points.shape[0] > 0, "empty points"
            mask = np.logical_and(points>lb, points<ub).all(axis=-1)
            cprint (f"[IN-BASKTER RATIO]: {mask.sum() / points.shape[0]}", "cyan")
            if(mask.sum() / points.shape[0] < 0.8):
                flg = False
                # record_success_failure(False, save_path, "final pose not correct")

            delete_prim(f"/World/Garment/garment_{index}")
            cprint (f"delete_prim garment_{i} [index]", "cyan")
            garment_index[i] = False
        elif garment_index[i]:
            if (
                position[i][2] < 0.45
                or position[i][2] > 1.2
                or position[i][1] > 2.5
                or position[i][1] < 1.7
                or position[i][0] > 0.43
                or position[i][0] < -0.43
            ):
                delete_prim(f"/World/Garment/garment_{i}")
                cprint (f"delete_prim garment_{i}", "cyan")
                garment_index[i] = False

    # record_success_failure(True, save_path, " success")

    return garment_index, flg


def furthest_point_sampling(points, colors=None, semantics=None, n_samples=4096):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    # print(colors)
    points = torch.Tensor(points).cuda(0)  # [N, 3]
    if colors is not None:
        colors = torch.Tensor(colors).cuda(0)
    if semantics is not None:
        semantics = semantics.astype(np.int32)
        semantics = torch.Tensor(semantics).cuda(0)

    # Number of points
    num_points = points.size(0)  # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda(0)  # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points).cuda(0) * float("inf")  # [N]

    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda(0)  # [1]
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()


def write_rgb_image(rgb_data, filename):
    from PIL import Image

    image = Image.fromarray(rgb_data)
    image.save(filename)
    cprint(f"write to .png file successful : {filename}", "magenta")

def get_points(mesh):
    points = mesh.GetPointsAttr().Get()
    world_xform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    world_points = [world_xform.Transform(p) for p in points]
    return world_points

def get_all_mesh_points(stage, root_path):
    points = []

    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_points = get_points(mesh)
        for x,y,z in mesh_points:
            points.append([x,y,z])

    return np.array(points)

def align_image_border (img_path : str, px : float, py : float, save_path : str = None):
    if save_path == None:
        save_path = img_path
    img = Image.open (img_path)
    w, h = img.size
    new_w = math.floor (w * px)
    new_h = math.floor (h * py)
    left = (w - new_w + 1) // 2
    top = (h - new_h + 1) // 2
    right = w - left
    bottom = h - top
    cropped = img.crop ((left, top, right, bottom))
    cropped.save (save_path)
    return left, top, w, h
