import numpy as np
import torch
import os

from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path, get_prim_children
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial

class Real_Ground:
    def __init__(
        self,
        scene,
        z_position: float = 0,
        name="default_ground_plane",
        prim_path: str = "/World/defaultGroundPlane",
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.8,
        visual_material_usd=None,
    ):
        usd_path = os.getcwd() + "/Assets/Scene/Collected_default_environment/default_environment.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        physics_material_path = find_unique_string_name(
            initial_name="/World/Physics_Materials/physics_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        physics_material = PhysicsMaterial(
            prim_path=physics_material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        if visual_material_usd is not None:
            self.ground_visual_material_path=find_unique_string_name("/World/Looks/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
            add_reference_to_stage(usd_path=visual_material_usd,prim_path=self.ground_visual_material_path)
            self.visual_material_prim=get_prim_at_path(self.ground_visual_material_path)
            self.material_prim=get_prim_children(self.visual_material_prim)[0]
            self.material_prim_path=self.material_prim.GetPath()
            self.visual_material=PreviewSurface(self.material_prim_path)
        else:
            self.visual_material=None
            
        
        plane = GroundPlane(prim_path=prim_path, name=name, z_position=z_position, physics_material=physics_material, visual_material=self.visual_material)