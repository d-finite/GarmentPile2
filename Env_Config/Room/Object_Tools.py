import numpy as np
import math
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility, delete_prim



def hanger_load(scene, pos_dx=0.0, pos_dy=0.0, pos_dz=0.0, theta=0.0):
        cos = math.cos(math.radians(theta))
        sin = math.sin(math.radians(theta))
        scene.add(
            FixedCuboid(
                name="cylinder",
                position=np.array([0.0+pos_dx, 0.85+pos_dy, 0.8+pos_dz]),
                prim_path="/World/hanger1",
                scale=np.array([0.02,0.02,0.75]),
                orientation=euler_angles_to_quat([theta,90.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=True,
            )
        )
        scene.add(
            VisualCuboid(
                name="collumn_left",
                position=np.array([pos_dx-0.375*cos, 0.85+pos_dy+0.375*sin, 0.75+pos_dz]),
                prim_path="/World/hanger2",
                scale=np.array([0.1,0.1,0.25]),
                orientation=euler_angles_to_quat([0.0,0.0,90.0-theta],degrees=True),
                color=np.array([0,0,1]),
                visible=True,
            )
        )
        scene.add(
            VisualCuboid(
                name="collumn_right",
                position=np.array([pos_dx+0.375*cos, 0.85+pos_dy-0.375*sin, 0.75+pos_dz]),
                prim_path="/World/hanger3",
                scale=np.array([0.1,0.1,0.25]),
                orientation=euler_angles_to_quat([0.0,0.0,90.0-theta],degrees=True),
                color=np.array([0,0,1]),
                visible=True,
            )
        )
        
        return np.array([0.0+pos_dx, 0.85+pos_dy, 0.8+pos_dz])
        
def pothook_load(scene, pos_dx=0.0, pos_dy=0.0):
        scene.add(
            VisualCuboid(
                prim_path = "/World/pothook1",
                color=np.array([0.545, 0.411, 0.078]),
                name = "hang_cube_1", 
                position = [0.0+pos_dx, 1.2+pos_dy, 0.7],
                scale=[0.05, 0.05, 0.8],
                size = 1.0,
                visible = True,
            )
        )
        scene.add(
            FixedCuboid(
                prim_path = "/World/pothook2",
                color=np.array([1.0, 0.756, 0.145]),
                name = "hang_cube_2", 
                position = [0.0+pos_dx, 1.075+pos_dy, 1.0],
                scale=[0.025, 0.2, 0.025],
                size = 1.0,
                visible = True,
            )
        )
        scene.add(
            FixedCuboid(
                prim_path = "/World/pothook3",
                color=np.array([1.0, 0.756, 0.145]),
                name = "hang_cube_3", 
                position = [0.0+pos_dx, 0.985+pos_dy, 1.0],
                scale=[0.12, 0.025, 0.025],
                size = 1.0,
                visible = True,
            )
        )
        
        return np.array([0.0+pos_dx, 0.985+pos_dy, 1.0])
    
def hat_helper_load(scene, pos_dx=0.0, pos_dy=0.0, env_dx=0.0, env_dy=0.0):
        # load hanger
        scene.add(
            FixedCuboid(
                name="hanger",
                position=[pos_dx, pos_dy, 0.4],
                prim_path="/World/hanger",
                scale=np.array([0.07,0.07,0.4]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=True,
            )
        )
        scene.add(
            FixedCuboid(
                name="hanger_helper",
                position=[pos_dx, pos_dy, 0.6],
                prim_path="/World/hanger_helper",
                scale=np.array([0.5,0.5,0.001]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        
        # load helper
        scene.add(
            FixedCuboid(
                name="head_helper",
                position=[0.0+env_dx,1.13+env_dy,1.03],
                prim_path="/World/head_helper",
                scale=np.array([0.08, 0.10, 0.10]),
                orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
                color=np.array([180,180,180]),
                visible=False,
            )
        )
        
        return np.array([0.0+env_dx,1.03+env_dy,1.25])
    
def pusher_loader(scene):
    pusher = FixedCuboid(
        name="pusher",
        position=[0.0,10.0,0.0],
        prim_path="/World/pusher",
        scale=np.array([0.5, 0.5, 0.1]),
        orientation=euler_angles_to_quat([0.0,0.0,0.0],degrees=True),
        color=np.array([180,180,180]),
        visible=True,
    )
    scene.add(pusher)
    return pusher
    
        
def set_prim_visible_group(prim_path_list:list, visible:bool):
    """
    Set the visibility of a group of prims.
    
    Args:
        prim_path_list (list): List of prim paths to set visibility for.
        visible (bool): Visibility state to set.
    """
    for prim_path in prim_path_list:
        prim = prims_utils.get_prim_at_path(prim_path)
        set_prim_visibility(prim, visible)
        
def delete_prim_group(prim_path_list:list):
    """
    Delete a group of prims.
    
    Args:
        prim_path_list (list): List of prim paths to delete.
    """
    for prim_path in prim_path_list:
        delete_prim(prim_path)