import os 
import sys
import torch
import numpy as np

import omni
from pxr import UsdPhysics, PhysxSchema

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim, XFormPrim
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.universal_robots import KinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Set_Drive import set_drive
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation

class DexLeft_Franka(Robot):
    def __init__(self, world:World, translation:np.ndarray, orientation:np.ndarray):
        # define world
        self.world = world
        # define DexLeft name
        self._name = "DexLeft"
        # define DexLeft prim
        self._prim_path = "/World/DexLeft"
        # get DexLeft usd file path
        # self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/dexleft.usd")
        # define DexLeft positon
        self.translation = translation
        # define DexLeft orientation
        self.orientation = euler_angles_to_quat(orientation, degrees=True)
        
        # add DexLeft USD to stage
        # add_reference_to_stage(self.asset_file, self._prim_path)
        # initialize DexLeft Robot according to USD file loaded in stage
        # super().__init__(
            # prim_path=self._prim_path,
            # name=self._name,
            # translation=self.translation,
            # orientation=self.orientation,
            # articulation_controller = None
        # )
        # add DexLeft to the scene
        # self.world.scene.add(self)
        self.world.scene.add (Franka (
            prim_path = self._prim_path,
            name = self._name,
            position = self.translation,
            orientation = self.orientation
        ))

        self._robot : Franka = self.world.scene.get_object (self._name)

        self._controller = RMPFlowController (
            name = "rmpflow_controller",
            robot_articulation = self._robot
        )
        self._controller.reset ()
        self._articulation_controller = self._robot.get_articulation_controller ()

        # check whether pick point is reachable or not
        self.pre_distance = 0
        self.distance_nochange_epoch = 0
  
   
    def set_kinematic(self, flag: bool = True):
        stage = omni.usd.get_context().get_stage()

        root_prim_path = f"{self._prim_path}/panda_link0"
        prim = stage.GetPrimAtPath(root_prim_path)

        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.GetRigidBodyEnabledAttr().Set(True)
        rb.GetKinematicEnabledAttr().Set(flag)
    
    
    def initialize(self, physics_sim_view):
        # initialize robot
        super().initialize(physics_sim_view)
        
        """
        # reset default status
        self.set_default_state(position=self.translation, orientation=self.orientation)
        self.set_joints_default_state(
            np.array([
                -1.57, -1.84, -2.5, -1.89, -1.57, 0.0,
                0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                ])
        )
        self.post_reset()

        # get arm_dof names and arm_dof indices
        self.arm_dof_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        self.arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.arm_dof_names]
        # get hand_dof names and hand_dof indices
        self.hand_dof_names = [
            "WRJ2", "WRJ1", 
            "FFJ4", "FFJ3", "FFJ2", "FFJ1", 
            "MFJ4", "MFJ3", "MFJ2", "MFJ1", 
            "RFJ4", "RFJ3", "RFJ2", "RFJ1", 
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", 
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"
        ]
        self.hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.hand_dof_names]
        
        # inverse kinematics control
        self.ki_solver = KinematicsSolver(self, end_effector_frame_name="ee_link")
        self.end_effector = SingleXFormPrim(self._prim_path + "/ee_link", "end_effector")
        self.end_effector.initialize(physics_sim_view)
        """
        
        self.disable_gravity()

        # RMPFlow control
        # self.rmp_config = load_supported_motion_policy_config("UR10e", "RMPflow")
        # self.rmpflow = RmpFlow(**self.rmp_config)
        # self.rmpflow.set_robot_base_pose(self.translation, self.orientation)
        # self.articulation_rmpflow = ArticulationMotionPolicy(self, self.rmpflow, default_physics_dt = 1 / 60.0)

    def close (self):
        for _ in range (15):
            self._robot.gripper.close ()
            self.world.step (render = True)

    def open (self):
        for _ in range (15):
            self._robot.gripper.open ()
            self.world.step (render = True)

    def get_cur_ee_pos (self):
        ee_pos, R = self._controller.get_motion_policy ().get_end_effector_as_prim ().get_world_pose ()
        return ee_pos, R
    def get_cur_grip_pos(self):
        """
        get current gripper position and orientation
        """
        position, orientation = self._robot.gripper.get_world_pose()
        return position, orientation
    
    def get_world_pose (self):
        return self._robot.get_world_pose ()


    def position_reached (self, target,thres = 0.03):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy ().get_end_effector_as_prim ().get_world_pose ()
        pos_diff = np.linalg.norm (ee_pos - target)
        if pos_diff < thres:
            return True
        else:
            return False 
        
    def rotation_reached (self, target):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy ().get_end_effector_as_prim ().get_world_pose ()
        angle_diff = quat_diff_rad (R, target)[0]
        if angle_diff < 0.1:
            return True
        return False
        
    def reached (self, end_loc, env_ori = None, thres = 0.03):
        pose = self.get_cur_ee_pos()
        if (end_loc is not None) and np.linalg.norm(pose[0] - end_loc) > thres:
            return False
        if (env_ori is not None) and quat_diff_rad(pose[1], env_ori)[0] > thres:
            return False
        return True
        if env_ori is None:
            if self.position_reached (end_loc):
                return True
        else:
            if self.position_reached (end_loc) and self.rotation_reached (env_ori):
                return True
        return False
            
    
    def move (self, end_loc, env_ori = None):
        if end_loc is None:
            return

        target_joint_positions = self._controller.forward (
            target_end_effector_position=end_loc,
            target_end_effector_orientation=env_ori
        )
        self._articulation_controller.apply_action (target_joint_positions)
    
    def move_to (self, target_pos, target_ori = None):
        iter = 0
        while self.reached (target_pos, target_ori) == False:
            self.move (target_pos, target_ori)
            self.world.step ()
            iter += 1
            if iter > 300:
                print (f"FAILED")
                return False

        print (f"DONE")
        return True