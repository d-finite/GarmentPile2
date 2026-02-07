import os
import sys
import numpy as np
import torch
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.prims import SingleXFormPrim

sys.path.append(os.getcwd())
from Env_Config.Robot.DexLeft_Franka import DexLeft_Franka
from Env_Config.Robot.DexRight_Franka import DexRight_Franka
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation


class Bimanual_Franka:
    def __init__(self, world:World, dexleft_pos, dexleft_ori, dexright_pos, dexright_ori):
        self.world = world
        self.dexleft = DexLeft_Franka(world, dexleft_pos, dexleft_ori)
        self.dexright = DexRight_Franka(world, dexright_pos, dexright_ori)
        self.left = self.dexleft
        self.right = self.dexright
        
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********              Hand Pose Control             ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
    
    def set_both_hand_state(self, left_hand_state:str="None", right_hand_state:str="None"):
        # set hand pose according to given hand state ('open' / 'close')

        if left_hand_state == "close":
            self.dexleft.close ()
        elif left_hand_state == "open":
            self.dexleft.open ()
        
        if right_hand_state == "close":
            self.dexright.close ()
        elif right_hand_state == "open":
            self.dexright.open ()

        # wait action to be done
        for i in range(20):
            self.world.step(render=True)
       
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********         Inverse Kinematics Control         ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////      
            
    def return_to_initial_position(self, pos, ori=[0,0,90], offset=[0.5, 0, 0], angular_type="euler"):
        pos = np.array(pos)
        ori = np.array(ori)
        self.dense_move_both_ik(pos-offset, ori, pos+offset, ori, angular_type=angular_type)
        
    def dense_move_both_ik(self, left_pos, left_ori, right_pos, right_ori, angular_type="quat",degree=True, dense_sample_scale:int=0.05):
        '''
        Move DexLeft and DexRight simultaneously once and use dense trajectory to guaranteer smoothness.
        '''
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori,degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori,degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
        
        ee_left_pos = left_pos
        ee_right_pos = right_pos

        # self.dexleft.move (ee_left_pos, left_ori)
        # self.dexright.move (ee_right_pos, right_ori)
        # self.world.step (render = True)

        current_ee_left_pos, current_ee_left_ori = self.dexleft.get_cur_ee_pos()
        current_ee_right_pos, current_ee_right_ori = self.dexright.get_cur_ee_pos()
        
        """
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)
        
        left_interp_pos, left_interp_ori = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            start_quat=current_ee_left_ori,
            end_quat=left_ori,
            num_points=dense_sample_num,
        )
        right_interp_pos, right_interp_ori = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            start_quat=current_ee_right_ori,
            end_quat=right_ori,
            num_points=dense_sample_num,
        )

        # print (f"right_start = {current_ee_right_pos} {current_ee_right_ori}")
        # print (f"right_end = {ee_right_pos} {right_ori}")

        for i in range(dense_sample_num): 
            iter = 0
            while self.dexleft.reached (left_interp_pos[i], left_interp_ori[i]) == False or \
                self.dexright.reached (right_interp_pos[i], right_interp_ori[i]) == False:
                    # print (f"right_now = {self.dexright.get_cur_ee_pos ()[0]} {self.dexright.get_cur_ee_pos ()[1]}")
                    self.dexleft.move (left_interp_pos[i], left_interp_ori[i])
                    self.dexright.move (right_interp_pos[i], right_interp_ori[i])
                    self.world.step (render = True)
                    iter += 1
                    if iter > 500:
                        print (f"FAILED")
                        return False

            self.world.step (render = True)
        """

        self.world.step (render = True)
        iter = 0
        while self.dexleft.reached (ee_left_pos, left_ori) == False or \
            self.dexright.reached (ee_right_pos, right_ori) == False:
                self.dexleft.move (ee_left_pos, left_ori)
                self.dexright.move (ee_right_pos, right_ori)
                self.world.step (render = True)
                iter += 1
                if iter > 500:
                    print (f"FAILED")
                    return False

        print (f"DONE")
        return True
    
    def move_both_with_blocks(self, left_pos, left_ori, right_pos, right_ori,
                            angular_type="quat",degree=None, dense_sample_scale:int=0.1,
                            attach=None, indices=None):
        '''
        Move DexLeft and DexRight simultaneously once and use dense trajectory to guaranteer smoothness.
        '''

        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori,degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori,degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
            
        left_base_pose, _ = self.dexleft.get_world_pose()
        right_base_pose, _ = self.dexright.get_world_pose()
        
        ee_left_pos, ee_left_ori = left_pos, left_ori
        ee_right_pos, ee_right_ori = right_pos, right_ori
        
        current_ee_left_pos, current_ee_left_ori = self.dexleft.get_cur_ee_pos()
        current_ee_right_pos, current_ee_right_ori = self.dexright.get_cur_ee_pos()

        """ 
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        left_interp_pos, left_interp_ori = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            start_quat=current_ee_left_ori,
            end_quat=left_ori,
            num_points=dense_sample_num,
        )
        right_interp_pos, right_interp_ori = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            start_quat=current_ee_right_ori,
            end_quat=right_ori,
            num_points=dense_sample_num,
        )        
        
        for i in range(dense_sample_num):
            iter = 0
            while self.dexleft.reached (left_interp_pos[i], left_interp_ori[i]) == False or \
                self.dexright.reached (right_interp_pos[i], right_interp_ori[i]) == False:
                    self.dexleft.move (left_interp_pos[i], left_interp_ori[i])
                    self.dexright.move (right_interp_pos[i], right_interp_ori[i])

                    if attach is not None and indices is not None:
                        block1_pos = self.dexleft.get_cur_ee_pos () [0]
                        block0_pos = self.dexright.get_cur_ee_pos () [0]
                        block_pos = []
                        block_pos.append (block0_pos)
                        block_pos.append (block1_pos)
                        gripper_ori = []
                        gripper_ori.append ([1.0, 0.0, 0.0, 0.0])
                        gripper_ori.append ([1.0, 0.0, 0.0, 0.0])
                        for j in indices:
                            attach.block_list[j].set_world_pose (block_pos[j], gripper_ori[j]) 
                        
                    self.world.step (render = True)
                    iter += 1
                    if iter > 500:
                        print (f"FAILED")
                        return False

            self.world.step (render = True)
        """ 

        self.world.step (render = True)
        iter = 0
        while self.dexleft.reached (ee_left_pos, left_ori) == False or \
            self.dexright.reached (ee_right_pos, right_ori) == False:
                self.dexleft.move (ee_left_pos, left_ori)
                self.dexright.move (ee_right_pos, right_ori)

                if attach is not None and indices is not None:
                    block_pos = []
                    block_pos.append (self.dexleft.get_cur_ee_pos () [0])
                    block_pos.append (self.dexright.get_cur_ee_pos () [0])
                    gripper_ori = []
                    gripper_ori.append ([1.0, 0.0, 0.0, 0.0])
                    gripper_ori.append ([1.0, 0.0, 0.0, 0.0])
                    for j in indices:
                        attach.block_list[j].set_world_pose (block_pos[j], gripper_ori[j]) 

                self.world.step (render = True)
                iter += 1
                if iter > 500:
                    print (f"FAILED")
                    return False

        print (f"DONE")
        return True 