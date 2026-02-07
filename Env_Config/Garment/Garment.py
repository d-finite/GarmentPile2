import torch
import numpy as np
import random
from termcolor import cprint

from isaacsim.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.prims import (
    SingleXFormPrim as XFormPrim,
    SingleClothPrim as ClothPrim,
    SingleRigidPrim as RigidPrim,
    SingleGeometryPrim as GeometryPrim,
    SingleParticleSystem as ParticleSystem,
)

from isaacsim.core.api.materials import ParticleMaterial
import Env_Config.Utils_Project.utils as util
from .Particle_Garment import Particle_Garment


class WrapGarment:
    def __init__(
        self,
        world,
        garment_num,
        garment_usd_path_dict,
        garment_position,
        garment_orientation,
        garment_scale,
    ) -> None:
        self.world = world
        self.stage = world.stage
        self.garment_path = []
        self.garment_num = 0
        self.garment_group = []
        self.garment_usd_path_dict = garment_usd_path_dict
        garment_usd_num = len (garment_usd_path_dict)
        self.random_numbers = random.sample (range (0, garment_usd_num), garment_num)
        # self.random_numbers = [0] + random.sample (range (1, garment_usd_num), garment_num - 1)
        cprint (f"[garment] random_numbers = {self.random_numbers}", "cyan")

    def get_cur_poses(self):
        cur_pose = []
        for i in range(self.garment_num):
            pose = np.mean(util.get_all_mesh_points(self.stage, self.garment_path[i]), axis=0)
            pose0 = self.garment_group[i].garment_mesh.get_world_pose()[0]
            cprint (f"[GARMENT]: i = {i}, pose = {pose}, pose0 = {pose0}", "cyan")
            cur_pose.append(pose)
        return cur_pose
    
    def add_garment(self, pos, ori, scale=[0.0045, 0.0045, 0.0045], render = True):
        key = f"cloth{self.random_numbers[self.garment_num]}"
        
        garment = Particle_Garment(
            self.world,
            self.garment_usd_path_dict[key],
            pos, ori, scale,
        )
        self.garment_path.append(garment.garment_prim_path)
        self.garment_group.append(garment)
        self.garment_num += 1
        
        if render:
            for i in range(15):
                self.world.step()
            garment.particle_material.set_friction(0.0)
            for i in range(15):
                self.world.step()
                
        garment.particle_material.set_friction(2.0)
        self.particle_system = garment.particle_system
        self.particle_material = garment.particle_material

    def set_invisible_to_secondary_rays(self, flag:bool=False):
        for garment in self.garment_group:
            garment.set_secondary_rays(flag)
        for i in range(8):
            self.world.step()
       
    def set_garment_damping(self):
        for garment in self.garment_group:
            garment.particle_material.set_damping(5.0) 
            
        for i in range(6):
            self.world.step()




"""
Class Garment
used to grnerate one piece of garment
It will be encapsulated into Class WrapGarment to generate many garments(gatment_nums can be changed)
you can also use Class Garment seperately
"""


class Garment_:
    def __init__(
        self,
        stage,
        scene,
        usd_path,
        position=torch.tensor,
        orientation=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
        garment_index=int,
    ):
        self.stage = stage
        self.scene = scene
        self.usd_path = usd_path
        self.garment_view = UsdGeom.Xform.Define(self.stage, "/World/Garment")
        self.garment_name = f"garment_{garment_index}"
        self.garment_prim_path = f"/World/Garment/garment_{garment_index}"
        self.garment_mesh_prim_path = self.garment_prim_path + "/mesh"
        self.particle_system_path = f"/World/Garment/particleSystem_{garment_index}"
        self.particle_material_path = self.garment_prim_path + "/particleMaterial"
        self.garment_position = position
        self.garment_orientation = euler_angles_to_quat(orientation, degrees=True)
        self.garment_scale = scale

        # when define the particle cloth initially
        # we need to define global particle system & material to control the attribute of particle

        # particle system
        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            simulation_owner=self.scene.GetPath(),  # /physicsScene
            particle_contact_offset=0.008,
            enable_ccd=True,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
            solver_position_iteration_count=16,
            # ----optional parameter---- #
            # contact_offset=0.01,
            # rest_offset=0.008,
            # solid_rest_offset=0.01,
            # fluid_rest_offset=0.01,
        )

        # particle material
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            friction=2.0,
            drag=0.0,
            lift=0.03,
            particle_friction_scale=0.5,
            particle_adhesion_scale=1.0,
            adhesion_offset_scale=0.05,
            cohesion=50.0,
            # damping=0.0,
        )

        # bind particle material to particle system
        physicsUtils.add_physics_material_to_prim(
            self.stage,
            self.stage.GetPrimAtPath(self.particle_system_path),
            self.particle_material_path,
        )

        # add garment to stage
        add_reference_to_stage(self.usd_path, self.garment_prim_path)

        # garment configuration
        # define Xform garment in stage
        self.garment = XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            orientation=self.garment_orientation,
            position=self.garment_position,
            scale=self.garment_scale,
        )
        # add particle cloth prim attribute
        print (f"garment_usd : {usd_path}")
        self.garment_mesh = ClothPrim(
            name=self.garment_name + "_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=0.05,
            stretch_stiffness=1e15,
            bend_stiffness=5.0,
            shear_stiffness=5.0,
            spring_damping=10.0,
        )
        # get particle controller
        self.particle_controller = self.garment_mesh._cloth_prim_view

    def get_garment_prim_path(self):
        return self.garment_prim_path