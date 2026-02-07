import os
import re
import sys
import random

sys.path.append("Model_Train")

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from Data_Utils.utils import furthest_point_sampling


class DataLoader_AffModel(Dataset):
    def __init__(self, mode=None, data_dir: str=None):

        assert mode in ["train", "test", "val"]
        self.mode = mode

        # get pcd files
        folder_re = re.compile(r'^\d+-\d+$')
        pcd_paths = []
        for sub in os.listdir(data_dir):
            sub_path = os.path.join(data_dir, sub)
            if os.path.isdir(sub_path) and folder_re.match(sub):
                for fn in os.listdir(sub_path):
                    if fn.lower().endswith('.pcd'):
                        pcd_paths.append(os.path.join(sub_path, fn))

        def pcd_sort_key(path):
            folder = os.path.basename(os.path.dirname(path))
            filename = os.path.basename(path)
            i_str, j_str = folder.split('-')
            num_str, _ = os.path.splitext(filename)
            return (int (i_str), int (j_str), int (num_str))
        # sort pcd file and get final result
        self.pcd = sorted(pcd_paths, key=pcd_sort_key)

        # get pcd_file_nums
        self.pcd_nums = len(self.pcd)
        # get pick_points and labels from record
        pick_points = []
        labels = []
        for pcd_path in self.pcd:
            folder = os.path.basename(os.path.dirname(pcd_path))
            i_str, _  = folder.split('-')
            i_val = int(i_str)
            assert i_val == 0 or i_val == 1
            label = 0 if i_val == 0 else 1

            pcd = o3d.io.read_point_cloud(pcd_path)
            pointcloud = np.asarray(pcd.points, dtype = np.float32)
            colors = np.asarray(pcd.colors, dtype = np.float32)
            pick_point = pointcloud[np.all(np.isclose(colors, [1.0, 0.0, 0.0]), axis=1)]

            pick_points.append(pick_point)
            labels.append(label)

        self.pick_points = np.array (pick_points)
        self.labels = np.array (labels, dtype = np.float32)

    def __getitem__(self, index):
        """
        return:
            - point_cloud_data: sampled (4096)
            - label: 1/0(Success/Failure)
        """
        # get index pcd_path
        pcd_path = self.pcd[index]
        # read np_data from pcd
        pcd = o3d.io.read_point_cloud(pcd_path)
        pointcloud = np.asarray(pcd.points, dtype = np.float32)
        colors = np.asarray(pcd.colors, dtype = np.float32)
        # push pick_point_position to data
        col0 = colors[:, 0]  # shape (N,)
        mask = np.where(col0 == 0.0, 0.0, 1.0).astype(np.float32).reshape(-1, 1)
        data = np.concatenate([pointcloud, mask], axis=1)
        data[0] = np.append(self.pick_points[index], 1.0)
        # get label
        label = self.labels[index]

        # normalization
        xy = data[:, :2]
        mi, ma = xy.min(0), xy.max(0)
        d = ma - mi
        d[d == 0] = 1.0
        data[:, :2] = 2 * (xy - mi) / d - 1
       
        if self.mode == "train":
            refl_flag = [0, 1]
            random_number = random.choice (refl_flag)
            if random_number == 1:
                data[ : , 0] = - data[ : , 0]

        return data, label

    def __len__(self):
        """
        return:
            - the num of point_clouds
        """
        if self.mode == "train":
            return self.pcd_nums
        else:
            return self.pcd_nums
