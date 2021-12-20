import open3d as o3d

my_colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.6, 0.0], [0.3, 0.3, 0.3], [0.2, 0.0, 0.3],
             [0.3, 0.9, 0.6], [1.0, 0.8, 0.6], [0.0, 0.6, 0.3], [0.6, 0.1, 0.7], [0.2, 0.9, 0.6],
             [0.6, 0.6, 0.2], [0.2, 0.3, 0.4], [0.5, 0.6, 0.6], [0.6, 0.3, 0.1], [0.4, 0.6, 0.1],
             [0.9, 0.3, 0.1], [0.0, 0.5, 0.1], [0.9, 0.6, 0.5], [0.6, 0.2, 0.3], [0.2, 0.3, 0.6]]


def visualize_atom_label(amino_acid_coordinates):
    points = []
    colors = []
    for x1, x2, y1, y2, z1, z2, amino_acid_id in amino_acid_coordinates:
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    points.append([x, y, z])
                    colors.append(my_colors[amino_acid_id])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
