import mrcfile
import numpy as np
import open3d as o3d

from preprocessing import get_amino_acid_coordinates, get_seg_label

my_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.6, 0.0], [0.3, 0.3, 0.3], [0.2, 0.0, 0.3],
             [0.3, 0.9, 0.6], [1.0, 0.8, 0.6], [0.0, 0.6, 0.3], [0.6, 0.1, 0.7], [0.2, 0.9, 0.6],
             [0.6, 0.6, 0.2], [0.2, 0.3, 0.4], [0.5, 0.6, 0.6], [0.6, 0.3, 0.1], [0.4, 0.6, 0.1],
             [0.9, 0.3, 0.1], [0.0, 0.5, 0.1], [0.9, 0.6, 0.5], [0.6, 0.2, 0.3], [0.2, 0.3, 0.6]]


def read_amino_acid_label(label_path):
    amino_acid_coordinates = []
    with open(label_path, 'r') as label_file:
        for line in label_file:
            [x1, x2, y1, y2, z1, z2, amino_acid_id] = list(map(int, line.strip('\n').split(',')))
            amino_acid_coordinates.append([x1, x2, y1, y2, z1, z2, amino_acid_id])
    return amino_acid_coordinates


def visualize():
    # draw axis
    for a in range(0, shape[0]):
        points.append([a, 0, 0])
        colors.append([0.0, 0.0, 0.0])
    for a in range(0, shape[1]):
        points.append([0, a, 0])
        colors.append([1.0, 0.0, 0.0])
    for a in range(0, shape[2]):
        points.append([0, 0, a])
        colors.append([1.0, 0.0, 0.0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])


def add_amino_acid_label(amino_acid_coordinates):
    for x1, x2, y1, y2, z1, z2, amino_acid_id in amino_acid_coordinates:
        for x in range(x1, x2 + 1):
            points.append([x, y1, z1])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x, y2, z1])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x, y1, z2])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x, y2, z2])
            colors.append(my_colors[amino_acid_id - 1])
        for y in range(y1, y2 + 1):
            points.append([x1, y, z1])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x1, y, z2])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x2, y, z1])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x2, y, z2])
            colors.append(my_colors[amino_acid_id - 1])
        for z in range(z1, z2 + 1):
            points.append([x1, y1, z])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x1, y2, z])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x2, y1, z])
            colors.append(my_colors[amino_acid_id - 1])
            points.append([x2, y2, z])
            colors.append(my_colors[amino_acid_id - 1])


def add_seg_label(data):
    for x in range(0, data.shape[0]):
        for y in range(0, data.shape[1]):
            for z in range(0, data.shape[2]):
                if data[x][y][z] > 0:
                    points.append([x, y, z])
                    colors.append(my_colors[data[x][y][z] - 1])


def add_amino_acid(data):
    for x in range(0, data.shape[0]):
        for y in range(0, data.shape[1]):
            for z in range(0, data.shape[2]):
                if data[x][y][z] > 1:
                    points.append([x, y, z])
                    colors.append([1.0, 0.0, 0.0])


if __name__ == '__main__':
    pdb_id = '6Od0'
    # [full, full_crop, seg, seg_crop]
    visualize_type = 'seg_crop'
    pdb_path = 'debug/{}/{}.rebuilt.pdb'.format(pdb_id, pdb_id)
    map_path = 'debug/{}/normalized_map.mrc'.format(pdb_id)
    label_path = 'debug/{}/{}_0_2_1.txt'.format(pdb_id, pdb_id)
    npy_path = 'debug/{}/{}_0_2_1.npy'.format(pdb_id, pdb_id)
    npy_seg_path = 'debug/{}/{}_0_2_1_seg.npy'.format(pdb_id, pdb_id)

    points = []
    colors = []
    if visualize_type == 'full':
        data = mrcfile.open(map_path, mode='r').data
        add_amino_acid_label(get_amino_acid_coordinates(map_path, pdb_path))
        add_amino_acid(data)
        shape = data.shape
    elif visualize_type == 'full_crop':
        add_amino_acid_label(read_amino_acid_label(label_path))
        add_amino_acid(np.load(npy_path))
        shape = [64, 64, 64]
    elif visualize_type == 'seg':
        data = get_seg_label(map_path, pdb_path)
        add_amino_acid_label(get_amino_acid_coordinates(map_path, pdb_path))
        add_seg_label(data)
        shape = data.shape
    elif visualize_type == 'seg_crop':
        add_amino_acid_label(read_amino_acid_label(label_path))
        data = np.load(npy_seg_path)
        add_seg_label(np.load(npy_seg_path).astype(int))
        shape = [64, 64, 64]

    visualize()
