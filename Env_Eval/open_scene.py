# Open Scene
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

from Env_Config.Config.open_config import Config
from Env_Config.Room.Room import Wrap_basket, Wrap_table
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Garment.Garment import WrapGarment
from Env_Config.Utils_Project.utils import (
    get_unique_filename,
    open_scene_judge_final_all_poses,
)
from Env_Config.Utils_Project.Open_Scene_Collision_Group import Collision_Group
from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Camera.PointCloud_Camera import PointCloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera
import Env_Config.Utils_Project.utils as util
import copy
from Env_Eval.base import BaseEnv

from Env_Config.Qwen.solver import solver
from Env_Config.Qwen.solver_4stir import solver_4stir
from Env_Config.Qwen.sam2_image import processor_sam2image
from Env_Config.Qwen.sam2_video import processor_sam2video


class OpenSceneEnv (BaseEnv):
    def __init__ (self):
        # define the world
        super ().__init__ (backend = "numpy", light_position = [0, 0, 10])
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
        self.basket = Wrap_basket (
            np.array (self.config.robot_position) - [0, 0.1, 0],
            self.config.basket_orientation,
            self.config.basket_scale,
            self.config.basket_usd_path,
            self.config.basket_prim_path,
        )

        # load table
        delete_prim ("/World/Table")
        self.table = Wrap_table (
            self.config.table_position,
            self.config.table_orientation,
            self.config.table_scale,
            self.config.table_usd_path,
            self.config.table_prim_path,
        )

        for i in range (self.config.max_garment_num):
            delete_prim (f"/World/Garment/garment_{i}")
        self.garment_num = 5 + random.choices (range (5)) [0]
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
        self.recording_thread = threading.Thread (target = self.recording_camera.collect_rgb_graph_for_video)
        self.recording_thread.start ()

        # transport garment
        top_thread = threading.Thread (target = self.top_camera.collect_rgb_graph_for_video)
        top_thread.start ()

        self.garment_transportation ()
        cprint ("garment transportation finish!", "green")
        self.recording_camera.capture = False        

        # erase illegal garment
        self.garment_index, _, _ = open_scene_judge_final_all_poses (
            self.world.stage, self.garments.get_cur_poses (),
            -1, self.garment_index
        )
        cprint (f"legal garment: {sum (self.garment_index)} / {self.garment_num}")

        self.recording_camera.capture = True        
        for i in range (15):
            self.world.step ()
        self.recording_camera.capture = False

        self.top_camera.create_mp4 ("Results/open_scene_tmp/initial.mp4")
        cprint ("world ready!", "green")
        
        from isaacsim.core.prims import SingleXFormPrim as XFormP 
        cube_pos, _ = XFormP ('/World/Table/Cube/Cube_F').get_local_pose ()
        cprint (f"check cube pos: {cube_pos[2]}", "red")

    def capture_rgb (self, save_path):
        Path (save_path).mkdir (parents = True, exist_ok = True)
        self.recording_camera.get_rgb_graph (save_or_not = True, save_path = f"{save_path}/recording.jpg")
        self.top_camera.get_rgb_graph (save_or_not = True, save_path = f"{save_path}/top.jpg")

    def move_block (self,
        target_pos, attach, indices,
        dt = 0.01
    ):
        target_pos = np.array (target_pos)
        indices = np.array (indices)

        start_pos = np.zeros_like (target_pos)
        for j, idx in enumerate (indices):
            start_pos[j], _ = attach.block_list[idx].get_world_pose ()

        dist = np.linalg.norm (target_pos - start_pos)
        traj = np.linspace (start_pos, target_pos, num = int (np.max (dist / dt)) + 1, axis = 1)

        for i in range (traj.shape[1]):
            for j, idx in enumerate (indices):
                attach.block_list[idx].set_world_pose (traj[j][i], [1.0, 0.0, 0.0, 0.0])
            self.world.step (render = True)

    def model_pick_one_step (self):
        self.top_camera.get_rgb_graph (save_or_not = True, save_path = "Results/open_scene_tmp/current_scene.jpg")
        self.garments.set_invisible_to_secondary_rays (False)
        labels, masks = processor_sam2image (
            camera = self.top_camera,
            img_path = 'Results/open_scene_tmp/current_scene.jpg',
            use_depth_info = True,
        )

        # checking stir
        idx_choosed, masks = solver_4stir (
            camera = self.top_camera,
            img_path = "./Results/open_scene_tmp/current_scene.jpg",
            prompt_type = "check_if_masks_need_regen",
            use_depth_info = True,
            use_sam2 = False,
            exist_masks = True,
            labels = labels,
            masks = masks
        )
        
        if len (idx_choosed) > 2:
            idx_choosed = idx_choosed[ : 2]
            
        if len (idx_choosed) > 0:
            labels, masks = self.adjust_mask (
                idx_choosed, masks,
                self.top_camera.get_mask_pc,
            )
        else:
            cprint ("pass adjust_mask!", "yellow")
        
        # pick garment
        self.garments.set_invisible_to_secondary_rays (True)
        self.top_camera.get_rgb_graph (
            save_or_not = True,
            save_path = "./Results/open_scene_tmp/current_scene.jpg",
        )
        
        self.garments.set_invisible_to_secondary_rays (False)
        _, _, idx_choosed, masks = solver (
            camera = self.top_camera,
            img_path = "./Results/open_scene_tmp/current_scene.jpg",
            prompt_type = "open_scene_all_garments",
            use_depth_info = True,
            use_sam2 = False,
            exist_masks = True,
            labels = labels,
            masks = masks
        )
        if idx_choosed == -1:
            cprint ("[wrong] qwen choosed a illegal point\n", "red")
            return -1
        
        self.top_camera.get_point_cloud_data_from_segment (
            save_or_not = False,
            sample_flag = False,
            real_time_watch = False
        )
        pick_point, aff_value, pick_point_idx = self.top_camera.get_model_point_from_masks (
            idx_choosed = int (idx_choosed),
            masks = np.array (masks),
            sample_flag = True,
            show_select = False,
            need_pick_point_idx = True
        )
        if aff_value == -1:
            cprint ("[wrong] unexpectedly, cant find pick point in mask", "red")
            return -1

        point_in_img = np.argwhere (masks[int (idx_choosed)] > 0) [pick_point_idx]
        point_in_img[0], point_in_img[1] = point_in_img[1], point_in_img[0]
        
        if point_in_img[0] < 1280 / 2:
            master = 0
        else:
            master = 1
        print (f"master = {master}")
        
        self.garments.set_invisible_to_secondary_rays (True)
        print (f"pick_point = {pick_point}")
        if aff_value == -1:
            cprint ("[wrong] not pick garment\n", "red")
            return -1

        top_thread = threading.Thread (target = self.top_camera.collect_rgb_graph_for_video)
        top_thread.start ()
        
        self.recording_camera.capture = True 
        for i in range (100):
            self.world.step (render = True)

        # -- whole procedure -- #
        if master == 0:
            self.set_attach_to_garment (attach_position = [pick_point, None], indices = [0])
        else:
            self.set_attach_to_garment (attach_position = [None, pick_point], indices = [1])

        pick_point[-1] += 0.55
        if master == 0:
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            self.move_block ([np.array (self.config.target_positions[0])], attach = self.attach, indices = [0])
        else:
            self.move_block ([pick_point], attach = self.attach, indices = [1])
            self.move_block ([np.array (self.config.target_positions[1])], attach = self.attach, indices = [1])
        print ("ok master reach target lifted-position\n")

        for i in range (150):
            self.world.step ()
        self.top_camera.create_mp4 (f"Results/open_scene_tmp/pick.mp4")
        self.recording_camera.capture = False

        # considering the right arm
        cprint ("considering the right arm", "green")
        self.top_camera.get_rgb_graph (
            save_or_not = True,
            save_path = f"Results/open_scene_tmp/picked_scene.jpg"
        )

        self.garments.set_invisible_to_secondary_rays (False)
        labels, masks = processor_sam2video (
            camera = self.top_camera,
            video_path = 'Results/open_scene_tmp/pick.mp4',
            use_depth_info = True,
            retained_masks = np.array ([]),
            pick_point = point_in_img,
            save_type = 2,
        )
        multiple_picked_garments = solver (
            camera = self.top_camera,
            img_path = "Results/open_scene_tmp/picked_scene.jpg",
            prompt_type = 'check_if_multiple_picked_garments',
            use_depth_info = True,
            use_sam2 = False,
            exist_masks = True,
            labels = labels,
            masks = masks
        )
        
        if multiple_picked_garments:
            cprint ("Multiple picked garments detected", "yellow")
            self.garments.set_invisible_to_secondary_rays (True)
            self.recording_camera.capture = True 
            for i in range (100):
                self.world.step (render = True)
                
            if master == 0:
                self.attach.detach (0)
            else:
                self.attach.detach (1)

            for i in range (25):
                self.world.step () 
                
            self.recording_camera.capture = False 
            return 0
        else:
            cprint ("No multiple picked garments detected", "yellow")

        _, _, idx_choosed, masks = solver (
            camera = self.top_camera,
            img_path = "Results/open_scene_tmp/picked_scene.jpg",
            prompt_type = 'check_if_need_rightarm',
            use_depth_info = True,
            use_sam2 = False,
            exist_masks = True,
            labels = labels,
            masks = masks
        )
        
        if idx_choosed == -1:
            need_right_arm = False
            cprint (f"dont need right arm", "green")
        else:
            need_right_arm = True
            self.top_camera.get_point_cloud_data_from_segment (
                save_or_not = False,
                sample_flag = False,
                real_time_watch = False
            )
            pick_point_right = self.top_camera.get_random_right_pick_point (masks[idx_choosed])
            cprint (f"need right arm, point = {pick_point_right}", "green")
        
        self.garments.set_invisible_to_secondary_rays (True)
        self.recording_camera.capture = True
        for i in range (100):
            self.world.step (render = True)

        if need_right_arm == True:
            if master == 0:
                self.set_attach_to_garment (attach_position = [None, pick_point_right], indices = [1])
            else:
                self.set_attach_to_garment (attach_position = [pick_point_right, None], indices = [0])
            print ("ok assistant reach pick_point\n")
            
            if master == 0:
                self.move_block ([np.array (self.config.target_positions[1])], attach = self.attach, indices = [1])
            else:
                self.move_block ([np.array (self.config.target_positions[0])], attach = self.attach, indices = [0])
            print ("ok assistant reach target lifted-position\n")
        
            tar_position = np.array (self.basket._basket_position.copy ())
            tar_position[-1] += 0.75
            self.move_block ([tar_position - np.array ([0.16, 0.01, 0]),
                              tar_position - np.array ([ -0.16, 0.01, 0])],
                                attach = self.attach, indices = [0, 1])
            
            print ("ok franka up\n")
            tar_position[-1] -= 0.3
            self.move_block ([tar_position - np.array ([0.16, 0.01, 0]),
                              tar_position - np.array ([ -0.16, 0.01, 0])],
                                attach = self.attach, indices = [0, 1])
            print ("ok franka down\n")

        else:
            tar_position = np.array (self.basket._basket_position.copy ())
            tar_position[-1] += 0.75
            if master == 0:
                self.move_block ([tar_position - np.array ([0.16, 0.01, 0])], attach = self.attach, indices = [0])
            else:
                self.move_block ([tar_position - np.array ([ -0.16, 0.01, 0])], attach = self.attach, indices = [1])
            print ("ok franka up\n")
            
            tar_position[-1] -= 0.3
            if master == 0:
                self.move_block ([tar_position - np.array ([0.16, 0.01, 0])], attach = self.attach, indices = [0])
            else:
                self.move_block ([tar_position - np.array ([ -0.16, 0.01, 0])], attach = self.attach, indices = [1])
            print ("ok franka down\n")

        for i in range (75):
            self.world.step ()
        
        if master == 0:
            self.attach.detach (0)
            if need_right_arm == True:
                self.attach.detach (1)
        else:
            self.attach.detach (1)
            if need_right_arm == True:
                self.attach.detach (0)
            
        for i in range (250):
            self.world.step () 


        for i in range (50):
            self.world.step ()
        cprint ("ok pick", "green")
        
        # erase illegal garment
        self.garment_index, _, _ = open_scene_judge_final_all_poses (
            self.world.stage, self.garments.get_cur_poses (),
            -1, self.garment_index
        )
        cprint (f"legal garment: {sum (self.garment_index)} / {self.garment_num}")
        
        self.recording_camera.capture = False 
        return 1


    def adjust_mask (self,
        idxs_choosed, masks, pc_gen,
    ):
        cprint ("adjust_mask!", "yellow")
        
        other_masks = []
        for i, mask in enumerate (masks):
            flag = False
            for j in idxs_choosed:
                if i == j:
                    flag = True
            if flag == False:
                other_masks.append (mask)
        other_masks = np.array (other_masks, dtype = bool)
        
        pick_points = []
        for i, idx_choosed in enumerate (idxs_choosed):
            mask_choosed = masks[idx_choosed]
            mask_pc = pc_gen (mask_choosed)
            
            p_idx = np.random.choice (np.arange (mask_pc.shape[0]))
            pick_point = mask_pc[p_idx].copy ()
            
            print (f"z_mean of mask_pc: ", mask_pc[ : , 2].mean ())
            pick_points.append (pick_point)

        self.garments.set_invisible_to_secondary_rays (True)
        # pick whole procedure
        cprint ("ready to log!", "green")
        top_thread = threading.Thread (target = self.top_camera.collect_rgb_graph_for_video)
        top_thread.start () 
        
        self.recording_camera.capture = True 
        for i in range (100):
            self.world.step (render = True)
        
        for i, pick_point in enumerate (pick_points):
            for i in range (15):
                self.world.step ()

            self.set_attach_to_garment (attach_position = [pick_point, None], indices = [0])
            
            pick_point[-1] += 0.65
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[0] += 0.1
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[0] -= 0.1 * 2
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[0] += 0.1
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[1] += 0.1
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[1] -= 0.1 * 2
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[1] += 0.1
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            pick_point[-1] -= 0.35
            self.move_block ([pick_point], attach = self.attach, indices = [0])
            print ("ok franka whole-adjust-procedure\n")

            for i in range (50):
                self.world.step ()

            self.attach.detach (0)
            for i in range (100):
                self.world.step ()

        for i in range (150):
            self.world.step ()
        self.recording_camera.capture = False

        self.top_camera.create_mp4 ("Results/open_scene_tmp/stir.mp4")

        # sam2 => regen mask
        self.garments.set_invisible_to_secondary_rays (False)
        labels, masks = processor_sam2video (
            camera = self.top_camera,
            video_path = 'Results/open_scene_tmp/stir.mp4',
            retained_masks = other_masks,
            use_depth_info = True,
            save_type = 1,
        )
        return labels, masks


    def garment_transportation (self):
        self.garments.particle_material.set_friction (0.0)
        self.recording_camera.capture = True        
        for i in range (15):
            self.world.step ()
        self.recording_camera.capture = False        

        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_ceiling",
                name = "temp_ceiling",
                position = self.config.temp_ceiling_position,
                scale = [2.0, 2.0, 0.01],
                visible = False 
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall1",
                name = "temp_wall1",
                position = self.config.temp_wall1_position,
                scale = [0.01, 2.0, 30.0],
                visible = False
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall2",
                name = "temp_wall2",
                position = self.config.temp_wall2_position,
                scale = [2.0, 0.01, 30.0],
                visible = False
            )
        )
        
        # change gravity direction
        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.2, 0.2, -0.2))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)
        self.recording_camera.capture = True
        for i in range (50):
            self.world.step ()
        self.recording_camera.capture = False
        
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall3",
                name = "temp_wall3",
                position = self.config.temp_wall3_position,
                scale = [0.01, 2.0, 30.0],
                visible = False
            )
        )
        self.world.scene.add (
            FixedCuboid (
                prim_path = "/World/temp_wall4",
                name = "temp_wall4",
                position = self.config.temp_wall4_position,
                scale = [2.0, 0.01, 30.0],
                visible = False
            )
        )

        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.0, 0.0, -0.2))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)

        self.recording_camera.capture = True
        for i in range (350):
            self.world.step ()
        self.recording_camera.capture = False
                
        cprint ("ready to change", "green")
        # return to the normal gravity direction
        self.scene.CreateGravityDirectionAttr ().Set (Gf.Vec3f (0.0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr ().Set (9.8)
        
        self.recording_camera.capture = True
        for i in range (250):
            self.world.step ()
        self.recording_camera.capture = False
        cprint ("ok garments onto table", "green")

        self.garments.particle_material.set_friction (0.35)

        self.recording_camera.capture = True
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
        self.recording_camera.capture = False

        delete_prim ("/World/temp_wall1")
        delete_prim ("/World/temp_wall2")
        delete_prim ("/World/temp_wall3")
        delete_prim ("/World/temp_wall4")
        delete_prim ("/World/temp_ceiling")
        
        from isaacsim.core.prims import SingleXFormPrim as XFormP 
        cube_pos, _ = XFormP ('/World/Table/Cube/Cube_F').get_local_pose ()
        cube_pos[1] -= 5
        XFormP ('/World/Table/Cube/Cube_F', translation = cube_pos)
        cube_pos, _ = XFormP ('/World/Table/Cube/Cube_B').get_local_pose ()
        cube_pos[1] += 5
        XFormP ('/World/Table/Cube/Cube_B', translation = cube_pos, visible = True)
        # this "visible=True" is for debugging, you can remove it after confirming the cube is moved successfully
        cube_pos, _ = XFormP ('/World/Table/Cube/Cube_R').get_local_pose ()
        cube_pos[0] -= 5
        XFormP ('/World/Table/Cube/Cube_R', translation = cube_pos)
        cube_pos, _ = XFormP ('/World/Table/Cube/Cube_L').get_local_pose ()
        cube_pos[0] += 5
        XFormP ('/World/Table/Cube/Cube_L', translation = cube_pos)
        
        joint_pris = UsdPhysics.Joint.Get (self.world.stage, "/World/Table/Cube/PrismaticJoint")
        drive = UsdPhysics.DriveAPI.Get (joint_pris.GetPrim (), 'linear')
        drive.CreateTargetPositionAttr ().Set ( -10.0)
        
        for i in range (25):
            self.world.step ()

        self.recording_camera.capture = True
        for i in range (100):
            self.world.step ()
        self.recording_camera.capture = False


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
    Path ("Results/open_scene_tmp").mkdir (parents = True, exist_ok = True)

    video_path = "Results/whole_procedure/open_scene"
    Path (video_path).mkdir (parents = True, exist_ok = True)
    for i in range (1):
        env = OpenSceneEnv ()
        total = sum (env.garment_index)
        cprint (f"[ROUND]: {i}", "red")
        mp4_id = sum (1 for e in os.scandir (video_path) if e.is_file ())

        flag = True
        while sum (env.garment_index) > 0:
            result = env.model_pick_one_step ()
            if result == -1:
                flag = False
                break

        # flag dedicates whether the pick procedure is all right
        # if flag == False, it means there is some wrong during the pick procedure
        # like picking a illegal point or cant find pick point in mask, etc.
        env.recording_camera.create_mp4 (f"{video_path}/{mp4_id}_{flag}.mp4")
        env.world.clear_instance ()


simulation_app.close ()