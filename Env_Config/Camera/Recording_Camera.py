import os
import sys
import numpy as np
import open3d as o3d
import imageio
import av
import time
from termcolor import cprint

import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles, quat_to_rot_matrix

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Code_Tools import get_unique_filename
from Env_Config.Utils_Project.Point_Cloud_Manip import furthest_point_sampling

from pxr import UsdGeom, Gf
import omni.usd
import numpy as np

class Recording_Camera:
    def __init__(self,
        camera_position:np.ndarray=np.array([0.0, 6.0, 2.6]),
        camera_orientation:np.ndarray=np.array([0, 20.0, -90.0]),
        frequency=20, resolution=(1280, 720),
        prim_path="/World/recording_camera", ori_type="angle"
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
            # self.annotator_semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            # self.annotator_semantic.attach(self.render_product)


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
            cprint(f"RGB image has been save into {save_path}", "green")
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
        workspace_z_limit:list=[0.005, None],
        ):
        '''
        get environment pointcloud data (remove the ground) from recording_camera, down_sample the number of points to be 2048.
        '''
        point_cloud = self.camera.get_pointcloud()
        print ()
        color = self.camera.get_rgb().reshape(-1, 3).astype(np.float32) / 255.0  # (N, 3)
        if show_original_pc_online:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([pcd])

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

        if sample_flag:
            down_sampled_point_cloud, down_sampled_color = furthest_point_sampling(point_cloud, colors=color, n_samples=sampled_point_num)
            if show_downsample_pc_online:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(down_sampled_point_cloud)
                pcd.colors = o3d.utility.Vector3dVector(down_sampled_color)
                o3d.visualization.draw_geometries([pcd])
            # down_sampled_point_cloud = np.hstack((down_sampled_point_cloud, down_sampled_color))
            self.point_cloud = down_sampled_point_cloud,
            self.color = down_sampled_color
            return down_sampled_point_cloud, down_sampled_color
        else:
            # point_cloud = np.hstack((point_cloud, color))
            self.point_cloud = point_cloud,
            self.color = color
            return point_cloud, color

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
        self.collecting = True
        self.capture = False        
        while self.collecting:
            if self.capture == True:
                data = self.camera.get_rgb ()
                if len (data):
                    self.video_frame.append (data)
            # take rgb photo every 500 ms
            time.sleep (0.5)
                
        cprint ("stop get rgb", "green")


    def create_gif(self, save_path:str=get_unique_filename(base_filename=f"Assets/Replays/carry_garment/animation/animation",extension=".gif")):
        '''
        [Not Recommend]
        create gif according to video frame list.
        Args:
            save_path(str): The path you wanna save, remember to include file name and file type(suffix).
        '''
        self.collecting = False 
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
        # create file path
        save_path_dir = os.path.dirname(save_path)
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        
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

        cprint(f"MP4 has been save into {save_path}", "green")
        # clear video frame list
        self.video_frame.clear()
