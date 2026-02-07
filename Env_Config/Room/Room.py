from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
import torch


class Wrap_room:
    def __init__(
        self,
        position=torch.tensor,
        orientation=[0.0, 0.0, 0.0],
        scale=[1, 1, 1],
        usd_path=str,
        prim_path: str = "/World/room",
    ):
        self._room_position = position
        self._room_orientation = orientation
        self._room_scale = scale
        self._room_prim_path = find_unique_string_name(
            prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._room_usd_path = usd_path

        add_reference_to_stage(self._room_usd_path, self._room_prim_path)

        self._room_prim = XFormPrim(
            self._room_prim_path,
            name="Room",
            scale=self._room_scale,
            position=self._room_position,
            orientation=euler_angles_to_quat(self._room_orientation, degrees=True),
        )

class Wrap_table:
    def __init__(
        self,
        position=np.ndarray,
        orientation=[0.0, 0.0, 0.0],
        scale=[1, 1, 1],
        usd_path=str,
        prim_path: str = "/World/table",
    ):
        self._table_position = position
        self._table_orientation = orientation
        self._table_scale = scale
        self._table_prim_path = find_unique_string_name(
            prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._table_usd_path = usd_path

        add_reference_to_stage(self._table_usd_path, self._table_prim_path)

        self._table_prim = XFormPrim(
            self._table_prim_path,
            name="Table",
            scale=self._table_scale,
            position=self._table_position,
            orientation=euler_angles_to_quat(self._table_orientation, degrees=True),
        )

        # print(RigidPrim(self._table_prim_path).get_mass())
        
class Wrap_basket:
    def __init__(
        self,
        position=torch.tensor,
        orientation=[0.0, 0.0, 0.0],
        scale=[1, 1, 1],
        usd_path=str,
        prim_path: str = "/World/basket",
    ):
        self._basket_position = position
        self._basket_orientation = orientation
        self._basket_scale = scale
        self._basket_prim_path = find_unique_string_name(
            prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._basket_usd_path = usd_path

        add_reference_to_stage(self._basket_usd_path, self._basket_prim_path)

        self._basket_prim = XFormPrim(
            self._basket_prim_path,
            name="Basket",
            scale=self._basket_scale,
            position=self._basket_position,
            orientation=euler_angles_to_quat(self._basket_orientation, degrees=True),
        )


class Wrap_base:
    def __init__(
        self,
        position=torch.tensor,
        orientation=[0.0, 0.0, 0.0],
        scale=[1, 1, 1],
        usd_path=str,
        prim_path: str = "/World/base",
    ):
        self._base_position = position
        self._base_orientation = orientation
        self._base_scale = scale
        self._base_prim_path = find_unique_string_name(
            prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._base_usd_path = usd_path

        add_reference_to_stage(self._base_usd_path, self._base_prim_path)

        self._base_prim = XFormPrim(
            self._base_prim_path,
            name="base",
            scale=self._base_scale,
            position=self._base_position,
            orientation=euler_angles_to_quat(self._base_orientation, degrees=True),
        )        