# Closed Scene
import os
import sys
import open3d as o3d

from isaacsim import SimulationApp
sys.path.append (os.getcwd ())
simulation_app = SimulationApp ({"headless": False})

# ---------------------coding begin---------------------#
import shutil
import numpy as np
import threading
import time
import random
import sys
import inspect
from pathlib import Path
from termcolor import cprint

import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema, Tf
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.prims import delete_prim, set_prim_visibility
from isaacsim.core.utils.viewports import set_camera_view

from Env_Config.Config.closed_config import Config
from Env_Config.Room.Room import Wrap_basket
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Garment.Garment import WrapGarment
from Env_Config.Utils_Project.utils import (
    get_unique_filename,
    basket_judge_final_poses,
)
from Env_Config.Utils_Project.Closed_Scene_Collision_Group import Collision_Group
from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Camera.PointCloud_Camera import PointCloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera
import Env_Config.Utils_Project.utils as util
import copy
from Env_Eval.base import BaseEnv

from Env_Config.Qwen.sam2_image import processor_sam2image


class ClosedSceneEnv (BaseEnv):
    def __init__ (self):
        # define the world
        super ().__init__ (backend = "numpy", light_position = [0, 0, 0])
        set_camera_view (
            eye = [-2.0, 1.4, 1.8],
            target = [0.0, 2.2, 0.2],
            camera_prim_path = "/OmniverseKit_Persp",
        )

        self.scene = self.world.get_physics_context ()._physics_scene

        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)

        # get environment config
        self.config = Config ()
        self.config.x_scale_factor = random.uniform (0.6, 1.0)
        self.config.y_scale_factor = random.uniform (0.6, 1.0)
        cprint (f"config.x_scale_factor = {self.config.x_scale_factor}", "cyan")
        cprint (f"config.y_scale_factor = {self.config.y_scale_factor}", "cyan")

        # load cameras
        self.recording_camera = Recording_Camera (
            camera_position = self.config.recording_camera_position,
            camera_orientation = self.config.recording_camera_orientation,
            prim_path = "/World/recording_camera",
            resolution = (1080, 960)
        )
        self.top_camera = PointCloud_Camera (
            camera_position = self.config.top_camera_position,
            camera_orientation = self.config.top_camera_orientation,
            prim_path = "/World/top_camera",
            resolution = tuple ([int (x) for x in self.config.top_camera_resolution])
        )

        # load ground
        self.ground = Real_Ground (
            self.scene,
            visual_material_usd = os.getcwd () + "/Assets/Material/Floor/WoodFloor003.usd"
        )

        # load basket
        delete_prim ("/World/Basket")
        self.config.basket_sx *= self.config.x_scale_factor
        self.config.basket_sy *= self.config.y_scale_factor

        self.world.scene.add (
            FixedCuboid (
                prim_path = f"{self.config.basket_prim_path}/comp1",
                name = "basket_comp1",
                position = [self.config.basket_position[0],
                            self.config.basket_position[1],
                            self.config.basket_position[2] - self.config.basket_sz,
                            ],
                scale = [self.config.basket_sx * 2,
                         self.config.basket_sy * 2,
                         self.config.basket_thick,
                        ],
                visible = True,
                color = self.config.basket_color,
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = f"{self.config.basket_prim_path}/comp2",
                name = "basket_comp2",
                position = [self.config.basket_position[0] + self.config.basket_sx,
                            self.config.basket_position[1],
                            self.config.basket_position[2],
                            ],
                scale = [self.config.basket_thick,
                         self.config.basket_sy * 2,
                         self.config.basket_sz * 2,
                        ],
                visible = True,
                color = self.config.basket_color,
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = f"{self.config.basket_prim_path}/comp3",
                name = "basket_comp3",
                position = [self.config.basket_position[0] - self.config.basket_sx,
                            self.config.basket_position[1],
                            self.config.basket_position[2],
                            ],
                scale = [self.config.basket_thick,
                         self.config.basket_sy * 2,
                         self.config.basket_sz * 2,
                        ],
                visible = True,
                color = self.config.basket_color,
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = f"{self.config.basket_prim_path}/comp4",
                name = "basket_comp4",
                position = [self.config.basket_position[0],
                            self.config.basket_position[1] + self.config.basket_sy,
                            self.config.basket_position[2],
                            ],
                scale = [self.config.basket_sx * 2,
                         self.config.basket_thick,
                         self.config.basket_sz * 2,
                        ],
                visible = True,
                color = self.config.basket_color,
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = f"{self.config.basket_prim_path}/comp5",
                name = "basket_comp5",
                position = [self.config.basket_position[0],
                            self.config.basket_position[1] - self.config.basket_sy,
                            self.config.basket_position[2],
                            ],
                scale = [self.config.basket_sx * 2,
                         self.config.basket_thick,
                         self.config.basket_sz * 2,
                        ],
                visible = True,
                color = self.config.basket_color,
            )
        )

        # load collection_baskeet
        delete_prim ("/World/Collection_Basket")
        self.collection_basket = Wrap_basket (
            self.config.collection_basket_position,
            self.config.collection_basket_orientation,
            self.config.collection_basket_scale,
            self.config.collection_basket_usd_path,
            self.config.collection_basket_prim_path,
        )

        for i in range (self.config.max_garment_num):
            delete_prim (f"/World/Garment/garment_{i}")
        self.garment_num = 3 + random.choices (range (5)) [0]
        cprint (f"garment_num: {self.garment_num}", "red")

        # load garment
        self.garments = WrapGarment (
            self.world,
            self.garment_num,
            self.config.clothpath,
            self.config.garment_position,
            self.config.garment_orientation,
            self.config.garment_scale,
        )
        self.garment_index = [True] * self.garment_num

        self.create_attach_block ()
        for i in range (self.garment_num):
            self.garments.add_garment (self.config.garment_position[i], self.config.garment_orientation[i], render = False)

        # -------------------initialize world------------------- #
        self.world.reset ()
        # create collision group
        self.collision = Collision_Group (self.world.stage)

        self.attach.block_list[0]._rigid_prim_view.disable_gravities ()
        self.attach.block_list[1]._rigid_prim_view.disable_gravities ()
        cprint ("world load successfully", "green")

        # initialize camera
        self.recording_camera.initialize (
            depth_enable = True,
            segment_pc_enable = True,
            segment_prim_path_list = [ f"/World/Garment/garment_{i}" for i in range (self.garment_num) ]
        )
        self.top_camera.initialize (
            depth_enable = True,
            segment_pc_enable = True,
            segment_prim_path_list = [ f"/World/Garment/garment_{i}" for i in range (self.garment_num) ]
        )
        cprint ("camera initialize successfully", "green")

        # begin to record mp4
        mp4_generation_thread = threading.Thread (target = self.recording_camera.collect_rgb_graph_for_video)
        mp4_generation_thread.start ()
        self.recording_camera.capture = True

        # transport garment
        self.garment_transportation ()
        cprint ("garment transportation finish!", "green")
        self.recording_camera.capture = False        

        # erase illegal garment
        self.garment_index, _ = basket_judge_final_poses (
            self.world.stage, self.garments.get_cur_poses (),
            -1, self.garment_index
        )
        cprint (f"legal garment: {sum (self.garment_index)} / {self.garment_num}")

        for i in range (15):
            self.world.step ()

        self.recording_camera.create_mp4 ("Results/closed_scene_tmp/initial.mp4")
        cprint ("world ready!", "green")


    def move_block (self,
        target_pos, attach, index,
        dt = 0.01
    ):
        start_pos, _ = attach.block_list[index].get_world_pose ()
        dist = np.linalg.norm (target_pos - start_pos)
        traj = np.linspace (start_pos, target_pos, num = int (dist / dt) + 1, axis = 0)
        for i in range (traj.shape[0]):
            attach.block_list[index].set_world_pose (traj[i], [1.0, 0.0, 0.0, 0.0])
            self.world.step (render = True)

    def collection_data_one_step (self):
        self.top_camera.get_rgb_graph (save_or_not = True, save_path = f"Results/closed_scene_tmp/current_scene.jpg")
        self.garments.set_invisible_to_secondary_rays (False)

        # a implementation of geting mask from sam2
        _, masks = processor_sam2image (
            camera = self.top_camera,
            img_path = 'Results/closed_scene_tmp/current_scene.jpg',
            use_depth_info = True,
        )
        
        if len (masks) == 0:
            cprint ("sam2 failed to segment garment => return", "red")
            return -1, None, None

        try_rounds = 30
        while try_rounds > 0:
            mask = masks[np.random.choice (len (masks), size = 1) [0]]
            # get only-garment pointcloud
            self.top_camera.get_point_cloud_data_from_segment (
                save_or_not = False,
                sample_flag = False,
                real_time_watch = False
            )
            # get mask pointcloud
            mask_pc, _ = self.top_camera.get_pointcloud_from_depth (
                sample_flag = False,
                object_mask_or_not = True,
                object_mask = mask
            )
            # get pick_point & final pointcloud
            pick_point, pointcloud, colors = self.top_camera.get_random_point_from_mask (
                mask_pc,
                sample_flag = True
            )

            if pick_point[0] != -1e3: # success
                try_rounds = -1
                break
            try_rounds -= 1

        if try_rounds != -1: # failed
            return -1, None, None

        self.cloth_picking = self.top_camera.get_cloth_picking ()
        self.set_attach_to_garment (attach_position = [pick_point, None], indices = [0])

        # -- whole procedure -- #
        self.garments.set_invisible_to_secondary_rays (True)
        mp4_generation_thread = threading.Thread (target = self.recording_camera.collect_rgb_graph_for_video)
        mp4_generation_thread.start ()
        self.recording_camera.capture = True

        pick_point[-1] += 0.75
        self.move_block (pick_point, attach = self.attach, index = 0)
        self.move_block (np.array (self.config.target_positions[0]), attach = self.attach, index = 0)

        for i in range (100):
            self.world.step ()

        tar_position = np.array (self.collection_basket._basket_position.copy ())
        tar_position[-1] += 0.85
        self.move_block (tar_position - np.array ([0, 0.01, 0]), attach = self.attach, index = 0)
        for i in range (10):
            self.world.step ()
        tar_position[-1] -= 0.3
        self.move_block (tar_position - np.array ([0, 0.01, 0]), attach = self.attach, index = 0)
        for i in range (75):
            self.world.step ()

        self.attach.detach (0)
        for i in range (75):
            self.world.step ()

        self.recording_camera.create_mp4 ("Results/closed_scene_tmp/putinto.mp4")
        cprint ("ok pick", "green")

        self.garment_index, success = basket_judge_final_poses (
            self.world.stage, self.garments.get_cur_poses (),
            int (self.cloth_picking[8 : ]), self.garment_index
        )
        cprint (f"success = {success}", "red")
        for i in range (25):
            self.world.step ()
        return int (success), pointcloud, colors


    def garment_transportation (self):
        self.garments.particle_material.set_friction (0.0)
        for i in range (15):
            self.world.step ()

        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_ceiling1",
                name = "temp_ceiling1",
                position = self.config.temp_ceiling_position,
                scale = [1.2, 1.2, 0.01],
                visible = False 
            )
        )
        self.config.temp_ceiling_position[2] -= 9.0

        self.config.temp_wall1_position[0] -= \
            (self.config.temp_wall1_position[0] - self.config.basket_position[0]) * \
            (1 - self.config.x_scale_factor)
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall1",
                name = "temp_wall1",
                position = self.config.temp_wall1_position,
                scale = [0.01, 0.6, 30.0],
                visible = False
            )
        )
        self.config.temp_wall2_position[1] -= \
            (self.config.temp_wall2_position[1] - self.config.basket_position[1]) * \
            (1 - self.config.y_scale_factor)
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall2",
                name = "temp_wall2",
                position = self.config.temp_wall2_position,
                scale = [1.2, 0.01, 30.0],
                visible = False
            )
        )

        # change gravity direction
        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.5, 0.5, -0.0002))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)
        for i in range (150):
            self.world.step ()

        self.config.temp_wall1_position[0] = \
            2 * self.config.basket_position[0] - \
            self.config.temp_wall1_position[0]
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall3",
                name = "temp_wall3",
                position = self.config.temp_wall1_position,
                scale = [0.01, 0.6, 30.0],
                visible = False
            )
        )
        self.config.temp_wall2_position[1] = \
            2 * self.config.basket_position[1] - \
            self.config.temp_wall2_position[1]
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall4",
                name = "temp_wall4",
                position = self.config.temp_wall2_position,
                scale = [1.2, 0.01, 30.0],
                visible = False
            )
        )

        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.0, 0.0, -0.2))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)

        for i in range (350):
            self.world.step ()

        cprint ("ready to change", "green")
        # return to the normal gravity direction
        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)

        for i in range (250):
            self.world.step ()
        cprint ("ok garments into basket", "green")

        delete_prim ("/World/temp_wall1")
        delete_prim ("/World/temp_wall2")
        delete_prim ("/World/temp_wall3")
        delete_prim ("/World/temp_wall4")
        delete_prim ("/World/temp_ceiling1")

        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_ceiling2",
                name = "temp_ceiling2",
                position = self.config.temp_ceiling_position,
                scale = [1.2, 1.2, 0.01],
                visible = False 
            )
        )
        self.garments.particle_material.set_friction (0.35)

        for _ in range (20):
            vec = np.random.normal (size = 3)
            vec /= np.linalg.norm (vec)
            vec *= 180
            self.garments.particle_system.set_wind (vec)
            for i in range (8):
                self.world.step ()

        self.garments.particle_system.set_wind (np.zeros (3))
        for i in range (20):
            self.world.step ()
        self.garments.particle_material.set_friction (2.0)

        delete_prim ("/World/temp_ceiling2")
        for i in range (100):
            self.world.step ()


    def create_attach_block (self, init_position = np.array ([0.0, 1.7, 0.9]), scale = [0.001, 0.001, 0.001]):
        """
        Create attachment block and update the collision group at the same time.
        """
        # create attach block and finish attach
        self.attach = AttachmentBlock (
            self.world,
            "/World/AttachmentBlock",
            self.garments,
        )
        self.attach.create_block (
            block_name = "attach0",
            block_position = init_position,
            block_visible = False,
            scale = scale,
        )
        self.attach.create_block (
            block_name = "attach1",
            block_position = init_position,
            block_visible = False,
            scale = scale,
        )
        cprint ("attach block create successfully", "green")

    def set_attach_to_garment (self, attach_position, indices = [0, 1]):
        """
        push attach_block to new grasp point and attach to the garment
        """
        # set the position of block
        for i in indices:
            self.attach.set_block_position (i, attach_position[i])
            self.attach.set_block_velocity_0 (i)
            cprint (f"attach set pose {self.attach.block_list[i].get_world_pose ()}", "cyan")
        self.world.step ()

        # create attach
        self.attach.attach (indices)
        self.world.step (render = True)

        cprint ("attach block set successfully", "green")


if __name__ == "__main__":
    for i in range (3):
        env = ClosedSceneEnv ()
        cprint (f"[ROUND]: {i}", "red")

        while sum (env.garment_index) > 0:
            remain = sum (env.garment_index)
            cprint (f"remain = {remain}", "cyan")
            result, pointcloud, colors = env.collection_data_one_step ()
            if result == -1:
                cprint ("motion_gen failed => break", "red")
                break
            else:
                save_path = f"Model_Train/Data/closed_scene/{result}-{remain}"
                os.makedirs (save_path, exist_ok = True)
                pcd = o3d.geometry.PointCloud ()
                pcd.points = o3d.utility.Vector3dVector (pointcloud)
                pcd.colors = o3d.utility.Vector3dVector (colors)
                idx = sum (1 for e in os.scandir (save_path) if e.is_file ())
                cprint (f"save_path = {save_path}/{idx:06d}.pcd", "green")
                o3d.io.write_point_cloud (f"{save_path}/{idx:06d}.pcd", pcd)

        env.world.clear_instance ()


simulation_app.close ()