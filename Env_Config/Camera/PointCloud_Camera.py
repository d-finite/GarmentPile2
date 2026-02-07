import os
import sys
import numpy as np
import open3d as o3d
import imageio
import av
import time
import torch
from termcolor import cprint

import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles, quat_to_rot_matrix

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Code_Tools import get_unique_filename
from Env_Config.Utils_Project.Point_Cloud_Manip import furthest_point_sampling
from Env_Config.Model.pn2_aff import Aff_Model

from pxr import UsdGeom, Gf
import omni.usd
import numpy as np

class PointCloud_Camera:
    def __init__(self,
        camera_position:np.ndarray=np.array([0.0, 6.0, 2.6]),
        camera_orientation:np.ndarray=np.array([0, 20.0, -90.0]),
        frequency=20, resolution=(1280, 720),
        prim_path="/World/top_camera", ori_type="angle",
        aff_model_path = "./Env_Config/Model/closed_scene.pth",
    ):
        # define camera parameters
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.frequency = frequency
        self.resolution = resolution
        self.camera_prim_path = prim_path
        # define capture photo flag
        self.capture = True

        # define camera
        if ori_type == "angle":
            self.camera = Camera(
                prim_path=self.camera_prim_path,
                position=self.camera_position,
                orientation=euler_angles_to_quat(self.camera_orientation, degrees=True),
                frequency=self.frequency,
                resolution=self.resolution,
            )
            self.camera.set_world_pose(
                self.camera_position,
                euler_angles_to_quat(self.camera_orientation, degrees=True),
                camera_axes="usd"
            )
        elif ori_type == "quat":
            self.camera = Camera(
                prim_path=self.camera_prim_path,
                position=self.camera_position,
                orientation=self.camera_orientation,
                frequency=self.frequency,
                resolution=self.resolution,
            )
            self.camera.set_world_pose(
                self.camera_position,
                self.camera_orientation,
                camera_axes="usd"
            )

        self.aff_model_path = aff_model_path

        # Attention: Remember to initialize camera before use in your main code. And Remember to initialize camera after reset the world!!

    def initialize(self, depth_enable:bool=False, segment_pc_enable:bool=False, segment_prim_path_list=None):

        self.video_frame = []
        self.camera.initialize()


        # choose whether add depth attribute or not
        if depth_enable:
            self.camera.add_distance_to_image_plane_to_frame()

        # choose whether add pointcloud attribute or not
        if segment_pc_enable:
            for path in segment_prim_path_list:
                semantic_type = "class"
                semantic_label = path.split("/")[-1]
                print(semantic_label)
                prim_path = path
                print(prim_path)
                rep.modify.semantics([(semantic_type, semantic_label)], prim_path)

            self.render_product = rep.create.render_product(self.camera_prim_path, [640, 480])
            self.annotator = rep.AnnotatorRegistry.get_annotator("pointcloud")
            self.annotator.attach(self.render_product)
            self.annotator_semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            self.annotator_semantic.attach(self.render_product)

        self.aff_model = Aff_Model ().cuda (0)
        self.aff_model.load_state_dict (torch.load (self.aff_model_path))
        self.aff_model.eval ()


    def get_rgb_graph(self, save_or_not:bool=False, save_path:str=get_unique_filename(base_filename=f"./image",extension=".png")):
        '''
        get RGB graph data from recording_camera, save it to be image file(optional).
        Args:
            save_or_not(bool): save or not
            save_path(str): The path you wanna save, remember to add file name and file type(suffix).
        '''
        data = self.camera.get_rgb()
        if save_or_not:
            imageio.imwrite(save_path, data)
            cprint (f"RGB image has been save into {save_path}", "green")
        return data

    def get_depth_graph(self):
        return self.camera.get_depth()

    def get_intrinsic_matrix(self):
        K = self.camera.get_intrinsics_matrix()
        print("Camera intrinsics K:\n", K)
        return K

    def get_extrinsic_matrix(self) -> np.ndarray:
        """
        返回相机从相机坐标系到世界坐标系的 4×4 齐次变换矩阵 T_cw。
        T_cw = [ R  t ]
               [ 0  1 ]
        其中 R 由 self.camera_orientation（Euler 角）转换而来，t 是 self.camera_position。
        """
        # 1. 将 Euler 角换成四元数
        quat = euler_angles_to_quat(self.camera_orientation, degrees=True, extrinsic=False)
        # 2. 四元数转旋转矩阵 R (3×3)
        R = quat_to_rot_matrix(quat)
        # 3. 平移向量 t (3,)
        t = self.camera_position

        # 4. 拼成 4×4 齐次矩阵
        T_cw = np.eye(4, dtype=float)
        T_cw[:3, :3] = R
        T_cw[:3,  3] = t

        return T_cw

    def get_world_points_from_image_coords(self, points_2d: np.ndarray, depth: np.ndarray):
        return self.camera.get_world_points_from_image_coords(points_2d, depth)

    def show_pointcloud (self, pointcloud, colors = np.zeros ((0, 3))):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    
    def save_pointcloud (self, save_path, pointcloud, colors = np.zeros ((0, 3))):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(save_path, pcd)

    def get_point_cloud_data_from_segment(
        self,
        save_or_not:bool=True,
        save_path:str=get_unique_filename(base_filename=f"./pc",extension=".pcd"),
        sample_flag:bool=True,
        sampled_point_num:int=2048,
        real_time_watch:bool=False
        ):
        '''
        get point_cloud's data and color(between[0, 1]) of each point, down_sample the number of points to be 2048, save it to be ply file(optional).
        '''
        self.data=self.annotator.get_data()
        self.point_cloud=np.array(self.data["data"])
        pointRgb=np.array(self.data["info"]['pointRgb'].reshape((-1, 4)))
        self.colors = np.array(pointRgb[:, :3] / 255.0)
        if sample_flag:
            self.point_cloud, self.colors = furthest_point_sampling(self.point_cloud, self.colors, sampled_point_num)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        if real_time_watch:
            o3d.visualization.draw_geometries([pcd])
        if save_or_not:
            o3d.io.write_point_cloud(save_path, pcd)

        return self.point_cloud, self.colors

    def get_pointcloud_from_depth(
        self,
        show_original_pc_online:bool=False,
        sample_flag:bool=True,
        sampled_point_num:int=2048,
        show_downsample_pc_online:bool=False,
        workspace_x_limit:list=[None, None],
        workspace_y_limit:list=[None, None],
        workspace_z_limit:list=[0.0001, None],
        object_mask_or_not : bool = False,
        object_mask : np.ndarray = None,
        show_mask_pc_online : bool = False
        ):
        '''
        get environment pointcloud data (remove the ground) from recording_camera, down_sample the number of points to be 2048.
        '''
        point_cloud = self.camera.get_pointcloud()
        color = self.camera.get_rgb().reshape(-1, 3).astype(np.float32) / 255.0  # (N, 3)

        if show_original_pc_online:
            self.show_pointcloud (point_cloud, color)

        if object_mask_or_not == True:
            # cprint (f"object_mask.shape = {object_mask.shape}", "cyan")
            object_mask = object_mask.reshape (-1)
            assert point_cloud.shape[0] == object_mask.shape[0], \
                    f"[WRONG] point_cloud & mask don't match! [{point_cloud.shape} & {object_mask.shape}]"
            point_cloud = point_cloud[np.array(object_mask, dtype = bool)]
            color = color[np.array(object_mask, dtype = bool)]
        
        print (f"test pc_shape = {point_cloud.shape}")

        # set the workspace limit
        mask = np.ones(point_cloud.shape[0], dtype=bool)
        # x limit
        if workspace_x_limit[0] is not None:
            mask &= point_cloud[:, 0] >= workspace_x_limit[0]
        if workspace_x_limit[1] is not None:
            mask &= point_cloud[:, 0] <= workspace_x_limit[1]
        # y limit
        if workspace_y_limit[0] is not None:
            mask &= point_cloud[:, 1] >= workspace_y_limit[0]
        if workspace_y_limit[1] is not None:
            mask &= point_cloud[:, 1] <= workspace_y_limit[1]
        # z limit
        if workspace_z_limit[0] is not None:
            mask &= point_cloud[:, 2] >= workspace_z_limit[0]
        if workspace_z_limit[1] is not None:
            mask &= point_cloud[:, 2] <= workspace_z_limit[1]
        # mask the point cloud
        point_cloud = point_cloud[mask]
        color = color[mask]
        
        print (f"test pc_shape = {point_cloud.shape}")

        if show_mask_pc_online:
            self.show_pointcloud (point_cloud, color)

        if sample_flag:
            down_sampled_point_cloud, down_sampled_color = furthest_point_sampling(point_cloud, colors=color, n_samples=sampled_point_num)
            if show_downsample_pc_online:
                self.show_pointcloud (down_sampled_point_cloud, down_sampled_color)
            # down_sampled_point_cloud = np.hstack((down_sampled_point_cloud, down_sampled_color))
            return down_sampled_point_cloud, down_sampled_color
        else:
            # point_cloud = np.hstack((point_cloud, color))
            return point_cloud, color

    def get_mask_pc (
        self,
        mask: np.ndarray
    ):
        mask_pc, _ = self.get_pointcloud_from_depth (
            sample_flag = False,
            object_mask_or_not = True,
            object_mask = mask
        )
        return mask_pc

    def get_pc_ratio(self):
        """
        get ratio of point cloud that is greater than 0.95
        """
        input = self.point_cloud.reshape(1, -1, 3)
        input = torch.from_numpy(input).float().cuda(0)
        output = self.model(input)
        count_greater_than_0_95 = (output > 0.95).sum().item()
        ratio_greater_than_0_95 = count_greater_than_0_95 / output.numel()
        return ratio_greater_than_0_95

    def get_cloth_picking (self):
        self.seg = self.annotator_semantic.get_data ()["info"]["idToLabels"]
        return self.seg.get (str (int (self.semantic_id)))["class"]

    def get_random_right_pick_point (self,
        mask: np.ndarray
    ):
        cprint ("Get right point (from mask)", "green")
        self.semantic_data = self.data["info"]['pointSemantic']
        
        def remove_outer_k (mask: np.ndarray, k: int) -> np.ndarray:
            from scipy.ndimage import distance_transform_cdt
            m = mask.astype(bool)
            dist = distance_transform_cdt(m, metric='taxicab')
            inner = dist > k
            return inner.astype(mask.dtype)
        
        inner = remove_outer_k (mask, 10)
        if inner.sum () > 0:
            mask = inner

        mask_pc, _ = self.get_pointcloud_from_depth (
            sample_flag = False,
            object_mask_or_not = True,
            object_mask = mask
        )
        assert mask_pc.sum () > 0, "empty mask"

        lookup = set (map (tuple, np.round (mask_pc.astype (np.float32), 1)))
        in_mask = np.array ([i for i, b in enumerate (np.round (self.point_cloud, 1)) if tuple (b) in lookup], dtype = int)
        if in_mask.shape[0] == 0: # empty
            return np.array ([-1e3, -1e3, -1e3])

        vals, counts = np.unique (self.semantic_data[in_mask], return_counts = True)
        self.semantic_id = vals[np.argmax (counts)]

        z = mask_pc[ : , 2]
        n = z.shape[0]

        order = np.argsort (z, kind = "mergesort")
        lo = max (0, int (np.floor (0.15 * n)))
        hi = min (n - 1, max (lo + 1, int (np.ceil (0.30 * n))))
        pool = order[lo : hi]

        pick_idx = np.random.choice (pool)
        pick_point = mask_pc[pick_idx].copy ()
        return pick_point

    def get_model_point_from_masks (self,
        idx_choosed: int,
        masks: np.ndarray,
        sample_flag: bool = True,
        show_select: bool = False,
        sampled_point_num: int = 4096,
        need_pick_point_idx : bool = False,
    ):
        """
        get model points from point_cloud (masked) graph to pick
        return pick_point
        """
        cprint ("Get model point (from mask)", "green")
        self.semantic_data = self.data["info"]['pointSemantic']
        all_mask = np.sum (masks, axis = 0)

        mask_pc, mask_pc_color = self.get_pointcloud_from_depth (
            sample_flag = False,
            object_mask_or_not = True,
            object_mask = masks[idx_choosed]
        )

        lookup = set (map (tuple, np.round (mask_pc.astype (np.float32), 1)))
        in_mask = np.array ([i for i, b in enumerate (np.round (self.point_cloud, 1)) if tuple (b) in lookup], dtype = int)
        if in_mask.shape[0] == 0: # empty
            if need_pick_point_idx == True:
                return np.array ([-1e3, -1e3, -1e3]), -1, -1
            else:
                return np.array ([-1e3, -1e3, -1e3]), -1

        vals, counts = np.unique (self.semantic_data[in_mask], return_counts = True)
        self.semantic_id = vals[np.argmax (counts)]

        rem_pc, rem_pc_color = self.get_pointcloud_from_depth (
            sample_flag = False,
            object_mask_or_not = True,
            object_mask = all_mask - masks[idx_choosed]
        )
        
        pointcloud = np.concatenate ([mask_pc, rem_pc], axis = 0)
        rgb_colors = np.concatenate ([mask_pc_color, rem_pc_color], axis = 0)
        
        colors = np.array ([[1.0, 1.0, 1.0]] * mask_pc.shape[0] + [[0.0, 0.0, 0.0]] * rem_pc.shape[0])
        
        if sample_flag == True:
            pointcloud, colors, fps_idx = furthest_point_sampling (pointcloud, colors=colors, n_samples=sampled_point_num, need_idx=True)

        input_pc = np.concatenate ([pointcloud, colors[ : , 0].reshape (-1, 1)], axis=1)
        # normalization
        xy = input_pc[:, :2]
        mi, ma = xy.min(0), xy.max(0)
        d = ma - mi
        cprint (f"normalization => d = {d}", "cyan")
        d[d == 0] = 1.0
        input_pc[:, :2] = 2 * (xy - mi) / d - 1

        input_pc = torch.from_numpy (input_pc).float ().cuda (0).unsqueeze (0)
        output = self.aff_model (input_pc.transpose (2, 1))
        value = output.detach ().to (torch.float32).cpu ().numpy ().reshape (-1)
        index = np.argmax (np.where (colors[ : , 0], value, - np.inf))
        cprint (f"index = {index}, value = {value[index], value[colors[ : , 0].astype (bool)].mean ()}", "cyan")
        
        pick_point = pointcloud[index].copy ()
        if show_select == True:
            colors[index] += [ - 1.0, 1.0, 0.0]
            self.show_pointcloud (pointcloud, colors) # red: mask, green: pick_point
            colors[index] -= [ - 1.0, 1.0, 0.0]
            aff_colors = np.stack([value, np.zeros_like(value), 1.0 - value], axis=1).astype(np.float64)
            aff_colors[colors[ : , 0] == 0.0] = np.zeros (3)
            self.show_pointcloud (pointcloud, aff_colors)
        
        if need_pick_point_idx == True:
            return pick_point, value[index], fps_idx[index]
        else:
            return pick_point, value[index]

    def get_random_point_from_mask (self,
        mask_pc : np.ndarray,
        show_select : bool = False,
        sample_flag : bool = False
    ):
        """
        get random points from point_cloud (masked) graph to pick
        return pick_point & point_cloud
        """
        cprint ("Get random point (from mask)", "green")
        self.semantic_data = self.data["info"]['pointSemantic']

        lookup = set (map (tuple, np.round (mask_pc.astype (np.float32), 1)))
        in_mask = np.array ([i for i, b in enumerate (np.round (self.point_cloud, 1)) if tuple (b) in lookup], dtype = int)
        if in_mask.shape[0] == 0: # empty
            return np.array ([-1e3, -1e3, -1e3]), None, None

        vals, counts = np.unique (self.semantic_data[in_mask], return_counts = True)
        self.semantic_id = vals[np.argmax (counts)]
        in_mask = in_mask[np.where (self.semantic_data[in_mask] == self.semantic_id)]

        if show_select:
            self.show_pointcloud (mask_pc)
            self.show_pointcloud (self.point_cloud, self.colors)
            self.show_pointcloud (self.point_cloud[in_mask], self.colors[in_mask])

        mask_info = np.zeros ((self.point_cloud.shape[0], 3), dtype = float)
        mask_info[in_mask, 0] = 0.3

        if sample_flag:
            pointcloud, colors = furthest_point_sampling (
                points = self.point_cloud,
                colors = mask_info,
                n_samples = 4096
            )
        else:
            pointcloud = self.point_cloud
            colors = mask_info

        pick_num = np.random.choice (np.where (colors[ : , 0] == 0.3) [0])
        colors[pick_num, 0] = 1

        if show_select:
            self.show_pointcloud (pointcloud, colors)

        pick_point = pointcloud[pick_num].copy ()
        return pick_point, pointcloud, colors

    def get_random_point_without_mask (self,
        show_select : bool = False,
        sample_flag : bool = False
    ):
        cprint ("Get random point (without mask)", "green")
        self.semantic_data = self.data["info"]['pointSemantic']
        self.semantic_id = np.random.choice(self.semantic_data)

        mask_info = np.zeros ((self.point_cloud.shape[0], 3), dtype = float)
        mask_info[self.semantic_data==self.semantic_id, 0] = 0.3

        if sample_flag:
            pointcloud, colors = furthest_point_sampling (
                points = self.point_cloud,
                colors = mask_info,
                n_samples = 4096
            )
        else:
            pointcloud = self.point_cloud
            colors = mask_info

        pick_num = np.random.choice (np.where (colors[ : , 0] == 0.3) [0])
        colors[pick_num, 0] = 1

        if show_select:
            self.show_pointcloud (pointcloud, colors)

        pick_point = pointcloud[pick_num].copy ()
        return pick_point, pointcloud, colors

    def get_point_from_xy(self, 
        xy: tuple,
        xy_convention: str = 'xy',
        show_select : bool = False
    ):
        """
        根据给定的2D像素坐标(x, y)，从完整的高清点云中精确找到对应的3D点。

        Args:
            xy (tuple): 一个包含(x, y)像素坐标的元组。
            xy_convention (str, optional): 指定输入'xy'元组的约定。
                                        - 'xy': 表示 (x=列, y=行)，这是标准约定。
                                        - 'yx': 表示 (y=行, x=列)。
                                        默认为 'xy'。
            show_select (bool, optional): 是否可视化显示选中的点。默认为 False。

        Returns:
            tuple: (pick_point, point_cloud, colors)
                - pick_point (np.ndarray): 找到的三维点的坐标 (3,)。
                - point_cloud (np.ndarray): 用于可视化的完整场景点云。
                - colors (np.ndarray): 用于可视化的颜色数组，选中的点会被高亮显示。
        """
        cprint(f"Getting point from pixel coordinate {xy} with convention '{xy_convention}'", "green")
        
        # 1. 获取相机分辨率 (宽度和高度)
        width,height = self.resolution
        
        # 2. 根据 flag 解析 x 和 y 的值
        if xy_convention == 'xy':
            # 标准约定：xy[0]是x(列), xy[1]是y(行)
            x, y = xy
        elif xy_convention == 'yx':
            # 备用约定：xy[0]是y(行), xy[1]是x(列)
            y, x = xy
        else:
            raise ValueError(f"Invalid xy_convention: '{xy_convention}'. Must be 'xy' or 'yx'.")

        # 3. 检查坐标是否在图像范围内
        if not (0 <= x < width and 0 <= y < height):
            cprint(f"Coordinate (x={x}, y={y}) is out of image bounds ({width}, {height})!", "red")
            return np.array([-1e3, -1e3, -1e3]), self.point_cloud, np.zeros_like(self.point_cloud)

        # 4. 计算一维索引
        index = int(y * width + x)
        if len(self.point_cloud) != width * height:
            raise ValueError(f"Point cloud size {len(self.point_cloud)} does not match image size {width*height}!")
        # 5. 从完整点云中直接选取该点
        pick_point = self.point_cloud[index].copy()
        
        # 6. (可选) 可视化
        if show_select:
            colors = np.full_like(self.point_cloud, 0.5)  # 灰色背景
            colors[index] = [1.0, 0.0, 0.0]  # 红色高亮选中的点
            self.show_pointcloud(self.point_cloud, colors)
        else:
            colors = np.zeros_like(self.point_cloud)

        return pick_point, self.point_cloud, colors
        

    def convert_translation_world_to_camera(self, points_world):
        '''
        convert world coordinate to camera coordinate
        '''
        view_matrix = self.camera.get_view_matrix_ros()
        # print(view_matrix.shape)
        N = points_world.shape[0]
        # Add homogeneous coordinate
        points_homog = np.hstack([points_world, np.ones((N, 1))])  # (N, 4)
        # Apply transformation
        points_cam_homog = (view_matrix @ points_homog.T).T  # (N, 4)
        # Remove homogeneous coordinate
        points_camera = points_cam_homog[:, :3]
        return points_camera

    def convert_translation_camera_to_world(self, points_camera):
        '''
        Convert camera coordinate to world coordinate
        '''
        view_matrix = self.camera.get_view_matrix_ros()  # T_wc⁻¹
        world_matrix = np.linalg.inv(view_matrix)        # T_wc

        N = points_camera.shape[0]
        points_homog = np.hstack([points_camera, np.ones((N, 1))])  # (N, 4)
        points_world_homog = (world_matrix @ points_homog.T).T      # (N, 4)
        return points_world_homog[:, :3]

    def convert_rotation_camera_to_world(self, R_cam):
        '''
        Convert rotation matrix from camera to world frame
        '''
        view_matrix = self.camera.get_view_matrix_ros()
        R_wc = np.linalg.inv(view_matrix)[0:3, 0:3]  # 3x3
        R_world = R_wc @ R_cam
        return R_world


    def collect_rgb_graph_for_video(self):
        '''
        take RGB graph from recording_camera and collect them for gif generation.
        '''
        # when capture flag is True, make camera capture photos
        self.capture = True
        while self.capture:
            data = self.camera.get_rgb()
            if len(data):
                self.video_frame.append(data)

            # take rgb photo every 500 ms
            time.sleep(0.1)
            # print("get rgb successfully")
        cprint("stop get rgb", "green")


    def create_gif(self, save_path:str=get_unique_filename(base_filename=f"Assets/Replays/carry_garment/animation/animation",extension=".gif")):
        '''
        [Not Recommend]
        create gif according to video frame list.
        Args:
            save_path(str): The path you wanna save, remember to include file name and file type(suffix).
        '''
        self.capture = False
        with imageio.get_writer(save_path, mode='I', duration=0.1) as writer:
            for frame in self.video_frame:
                # write each video frame into gif
                writer.append_data(frame)

        print(f"GIF has been save into {save_path} [DEBUG: len(video_frame) = {len(self.video_frame)}]")
        # clear video frame list
        self.video_frame.clear()

    def create_mp4(self, save_path:str=get_unique_filename(base_filename=f"Assets/Replays/carry_garment/animation/animation",extension=".mp4"), fps:int=10):
        '''
        create mp4 according to video frame list. (not mature yet, don't use)
        Args:
            save_path(str): The path you wanna save, remember to include file name and file type(suffix).
        '''
        self.capture = False

        container = av.open(save_path, mode='w')
        stream = container.add_stream('h264', rate=fps)
        stream.width = self.resolution[0]
        stream.height = self.resolution[1]
        stream.pix_fmt = 'yuv420p'

        for frame in self.video_frame:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            packet = stream.encode(frame)
            if packet:
                container.mux(packet)

        packet = stream.encode(None)
        if packet:
            container.mux(packet)

        container.close()

        cprint(f"MP4 has been save into {save_path} [{len (self.video_frame)}]", "green")
        # clear video frame list
        self.video_frame.clear()
