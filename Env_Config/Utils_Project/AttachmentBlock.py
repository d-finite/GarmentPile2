'''
AttachmentBlock
used to create a cube and attach the cube to the garment
in order to make Franka catch the garment smoothly
'''

import numpy as np
import torch
from termcolor import cprint

from pxr import PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.objects.sphere import DynamicSphere
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path


class AttachmentBlock:
    def __init__(self, world:World, prim_path="/World/AttachmentBlock", garments=None):
        '''
        Args:
        - prim_path: The prim position of 'AttachmentBlock' directory
        - garment_path: the prims of all the garment in the stage
        '''
        self.world = world
        self.stage = world.stage
        self.root_prim_path = prim_path
        self.garments = garments
        # attachment path list of all blocks(which means this list is a 2-dimension list)
        self.attachment_path_list = []
        # block path list
        self.block_path_list = []
        # block list
        self.block_list = []
        # block controller list
        self.move_block_controller_list = []
        # control the gravity of each block
        self.block_gravity_list = []
        
            
        
    def create_block(self, block_name, block_position=np.array([0.0, 0.0, 1.0]), scale=np.array([0.001, 0.001, 0.001]), block_visible=True):
        self.block_path = self.root_prim_path + "/" + block_name
        self.block_path_list.append(self.block_path)
        self.block = DynamicSphere(
            prim_path = self.block_path,
            color=np.array([1.0, 0.0, 0.0]),
            name = block_name, 
            position = block_position,
            scale=scale, 
            mass = 1e8,
            visible = block_visible,
            )
        self.block_list.append(self.block)
        self.world.scene.add(self.block)
        self.move_block_controller = self.block._rigid_prim_view
        self.move_block_controller_list.append(self.move_block_controller)
        
        # block can't be moved by external forces such as gravity and collisions
        # self.block.disable_rigid_body_physics()
        # self.move_block_controller.disable_rigid_body_physics()

        # at the beginning, block will be affected by gravity.
        self.block_gravity_list.append(False)
        # here we choose the disable the influence of gravity.
        self.move_block_controller.disable_gravities() # here means disable, function 'enable_gravities' will turn flag 'from True to False' or 'from False to True'
        # self.move_block_controller.enable_gravities()

        # one block reprensents one line of attachment path.
        self.attachment_path_list.append([])
        self.set_block_velocity_0 (len (self.block_list) - 1)
        
    
    def attach(self, block_index_indices:list):
        # In this function we will try to attach the cube to all the garment
        # Actually attachment will be generated successfully only when the cube is close to the particle of clothes
        # So attach the cube to all the garment will be fine
        # It will achieve the goal that different garment may get tangled up.
        garment_path = self.garments.garment_path
        self.garment_num = len(garment_path)
        for block_index in block_index_indices:
            for i in range(self.garment_num):
                self.attachment_path_list[block_index].append(garment_path[i] + f"/mesh/attachment_{block_index}")
                attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, self.attachment_path_list[block_index][i])
                attachment.GetActor0Rel().SetTargets([garment_path[i]+"/mesh"])
                attachment.GetActor1Rel().SetTargets([self.block_path_list[block_index]])
                att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
                att.Apply(attachment.GetPrim())
                _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.01)
                
                cprint("attachment finish", "green")
        
    def detach(self, block_index:int):
        # delete all the attachment related to the cube
        for i in range(self.garment_num):
            delete_prim(self.attachment_path_list[block_index][i])
        self.attachment_path_list[block_index] = []

    def set_block_position(self, block_index, grasp_point, grasp_orientations=torch.Tensor([1.0, 0.0, 0.0, 0.0])):
        '''
        set block position
        '''
        # grasp_point = torch.Tensor(grasp_point)
        self.block_list[block_index].set_world_pose(grasp_point, grasp_orientations)
    
    def set_block_velocity_0(self, block_index):
        self.move_block_controller_list[block_index].set_linear_velocities (np.array ([[0.0, 0.0, 0.0]], dtype = np.float32))
        self.move_block_controller_list[block_index].set_angular_velocities (np.array ([[0.0, 0.0, 0.0]], dtype = np.float32))

    def get_block_position(self, block_index):
        '''
        get block position
        '''
        pos, rot = self.move_block_controller_list[block_index].get_world_poses()
        return pos
    
    def set_block_velocity(self, cmd, block_index):
        '''
        set block velocity
        '''
        self.move_block_controller_list[block_index].set_velocities(cmd)
        
    def disable_rigid_body_physics(self, block_index):
        self.block_list[block_index].disable_rigid_body_physics()
        
    def enable_rigid_body_physics(self, block_index):
        self.block_list[block_index].enable_rigid_body_physics()


def attach_fixedblock(stage, attachment_path, garment_path, block_path):
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        
        attachment.GetActor0Rel().SetTargets([garment_path+"/mesh"])
        attachment.GetActor1Rel().SetTargets([block_path])
        att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
        att.Apply(attachment.GetPrim())
        _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.01)
        cprint("attachment finish", "green")