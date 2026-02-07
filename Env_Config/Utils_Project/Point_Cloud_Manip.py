import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import open3d as o3d

def furthest_point_sampling(points, colors=None, n_samples=2048,need_idx=False):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    points = torch.Tensor(points).cuda()  # [N, 3]
    if colors is not None:
        colors = torch.Tensor(colors).cuda()

    # Number of points
    num_points = points.size(0)  # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points).cuda() * float('inf')  # [N]

    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum((points[last_added] - points) ** 2, dim=-1)  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected
    if need_idx == True:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy(), sample_inds.cpu ().numpy ()
    if colors is None:
        return points[sample_inds].cpu().numpy()
    else:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()  # [S, 3]

def normalize_pcd_points_xy(pcd_points, x_range=(-1, 1), y_range=(-1, 1)):
    '''
    Normalize point cloud points to a given range.
    '''
    # calcaulate centroid
    centroid = np.mean(np.asarray(pcd_points), axis=0)

    # move centroid to origin
    normalized_points = np.asarray(pcd_points) - centroid

    # calculate scale factor
    min_coords = np.min(normalized_points[:, :2], axis=0)
    max_coords = np.max(normalized_points[:, :2], axis=0)
    scale_x = (x_range[1] - x_range[0]) / (max_coords[0] - min_coords[0]) if (max_coords[0] - min_coords[0]) != 0 else 1
    scale_y = (y_range[1] - y_range[0]) / (max_coords[1] - min_coords[1]) if (max_coords[1] - min_coords[1]) != 0 else 1
    scale = min(scale_x, scale_y)

    # scale
    normalized_points = normalized_points * scale

    return normalized_points, centroid, scale

import numpy as np

def normalize_pcd_points_xyz(pcd_points, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1)):
    '''
    Normalize point cloud points to a given XYZ range.
    '''
    pcd_points = np.asarray(pcd_points)

    # 计算质心
    centroid = np.mean(pcd_points, axis=0)

    # 平移到原点
    normalized_points = pcd_points - centroid

    # 获取最小最大值（XYZ）
    min_coords = np.min(normalized_points, axis=0)
    max_coords = np.max(normalized_points, axis=0)

    # 分别计算缩放因子
    scale_x = (x_range[1] - x_range[0]) / (max_coords[0] - min_coords[0]) if max_coords[0] != min_coords[0] else 1
    scale_y = (y_range[1] - y_range[0]) / (max_coords[1] - min_coords[1]) if max_coords[1] != min_coords[1] else 1
    scale_z = (z_range[1] - z_range[0]) / (max_coords[2] - min_coords[2]) if max_coords[2] != min_coords[2] else 1

    # 统一使用最小缩放因子，保持比例
    scale = min(scale_x, scale_y, scale_z)

    # 应用缩放
    normalized_points *= scale

    return normalized_points, centroid, scale


def calculate_chamfer_distance(pc1, pc2):
    """
    pc1: (N, D) array-like
    pc2: (M, D) array-like
    return: (N,) array — distance from each point in pc1 to its nearest point in pc2
    """
    # pairwise squared distances (N, M)
    d2 = ((pc1[:, None, :] - pc2[None, :, :]) ** 2).sum(axis=2)
    # nearest neighbor distance for each point in pc1
    return np.sqrt(d2.min(axis=1))


def rotate_point_cloud_relative_to_origin_point(points, euler_angles):
        '''
        rotate point cloud relative to origin point
        '''
        points = np.asarray(points)

        roll, pitch, yaw = np.deg2rad(euler_angles)  # 将角度转换为弧度

        # 绕 x 轴旋转的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 绕 y 轴旋转的旋转矩阵
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # 绕 z 轴旋转的旋转矩阵
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 计算总的旋转矩阵，旋转顺序为 R_z * R_y * R_x
        rotation_matrix = R_z @ R_y @ R_x

        # 应用旋转矩阵到点云
        rotated_points = points @ rotation_matrix.T  # 转置矩阵进行点乘

        return rotated_points

def rotate_point_cloud(pcd, euler_angles, center_point=np.array([0, 0, 0])):
    '''
    rotate point cloud around a given center point
    '''
    points = copy.deepcopy(pcd)

    points = np.asarray(points)
    center_point = np.asarray(center_point)

    # 将点云平移到原点
    points -= center_point

    # 旋转点云
    rotated_points = rotate_point_cloud_relative_to_origin_point(points, euler_angles)

    # 将点云平移回原始位置
    rotated_points += center_point

    return rotated_points

def colormap(pointcloud, scale=0.4):
    '''
    colormap for pointcloud.
    '''
    base_point = np.copy(pointcloud[0])
    distance = np.zeros((pointcloud.shape[0],1))
    point1 = np.copy(pointcloud[0])
    point2 = np.copy(pointcloud[0])
    for i in range(pointcloud.shape[0]):#最左下的点
        if pointcloud[i][0]+pointcloud[i][1]<base_point[0]+base_point[1]:
            base_point=pointcloud[i]
    for i in range(pointcloud.shape[0]):#最左上的点(255,0,255)
        if pointcloud[i][0]-pointcloud[i][1]<point1[0]-point1[1]:
            point1 = pointcloud[i]
    for i in range(pointcloud.shape[0]):#最右上的点(170,0,255)
        if pointcloud[i][0]+pointcloud[i][1]>point2[0]+point2[1]:
            point2 = pointcloud[i]

    base_point[0]-=0.02
    for i in range(pointcloud.shape[0]):
        distance[i] = np.linalg.norm(pointcloud[i] - base_point)
    max_value = np.max(distance)
    min_value = np.min(distance)
    cmap = plt.cm.get_cmap('jet_r')
    colors = cmap((-distance+max_value)/(max_value-min_value))
    colors = np.reshape(colors,(-1,4))
    color_map = np.zeros((pointcloud.shape[0], 3))
    i=0
    for color in colors:
        color_map[i] = color[:3]
        i=i+1
    color_map2 = np.zeros_like(color_map)
    for i in range(pointcloud.shape[0]):
        distance1 = np.linalg.norm(point1-pointcloud[i])
        distance2 = np.linalg.norm(point2-pointcloud[i])
        dis = np.abs(point1[1]-pointcloud[i][1])
        if dis < scale:
            color_map2[i] = np.array([75.0/255.0,0.0,130.0/255.0])*distance2/(distance1+distance2) + np.array([1.0,20.0/255.0,147.0/255.0])*distance1/(distance1+distance2)


    for i in range(pointcloud.shape[0]):
        distance1 = np.linalg.norm(point1-pointcloud[i])
        distance2 = np.linalg.norm(point2-pointcloud[i])
        distance3 = np.linalg.norm(point1-point2)
        dis = np.abs(point1[1]-pointcloud[i][1])
        if dis<scale:
            color_map[i] = color_map[i]*(dis)/(scale) + (color_map2[i])*(scale-dis)/(scale)

    return color_map

def visualize_pointcloud_with_colors(points, colors, save_or_not=False, save_path:str=None):

    if type(points) == torch.Tensor:
        points = points.cpu().numpy()
    if type(colors) == torch.Tensor:
        colors = colors.cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])

    if save_or_not:
        o3d.io.write_point_cloud(save_path, point_cloud)

def compute_similarity(pc, point, sigma=0.1):
    """
    Compute similarity between each point in a point cloud and a reference point,
    using a Gaussian decay function based on Euclidean distance.

    Args:
        pc: np.ndarray, shape (N, 3) - Point cloud coordinates
        point: np.ndarray, shape (3,) - Reference point coordinates
        sigma: float - Standard deviation of the Gaussian kernel, controls decay rate

    Returns:
        similarity: np.ndarray, shape (N, 1) - Similarity value for each point
    """
    # Compute Euclidean distances
    dist = np.linalg.norm(pc - point, axis=1)  # shape: (N,)

    # Compute similarity using Gaussian decay
    similarity = np.exp(- (dist ** 2) / (2 * sigma ** 2))  # shape: (N,)

    # Reshape to column vector (N, 1)
    similarity = similarity.reshape(-1, 1)

    return similarity
