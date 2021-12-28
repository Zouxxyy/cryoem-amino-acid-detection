import argparse
import math
import shutil

import mrcfile

from configs import *
from utils.pdb_utils import *

amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


def distance(z1, z2, y1, y2, x1, x2):
    z_diff = z1 - z2
    y_diff = y1 - y2
    x_diff = x1 - x2
    sum_squares = math.pow(z_diff, 2) + math.pow(y_diff, 2) + math.pow(x_diff, 2)
    return math.sqrt(sum_squares)


def label_sphere(arr, location, label, sphere_radius):
    box_size = np.shape(arr)
    for x in range(-sphere_radius + location[0], sphere_radius + location[0]):
        for y in range(-sphere_radius + location[1], sphere_radius + location[1]):
            for z in range(-sphere_radius + location[2], sphere_radius + location[2]):
                if (0 <= x < box_size[0] and 0 <= y < box_size[1] and 0 <= z < box_size[2]
                        and distance(location[2], z, location[1], y, location[0], x) < sphere_radius):
                    arr[x][y][z] = label


def get_seg_label(map_path, pdb_path):
    normalized_map = mrcfile.open(map_path, mode='r')
    origin = normalized_map.header.origin.item(0)
    shape = normalized_map.data.shape
    seg_label = np.zeros(shape, dtype=np.uint8)

    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                coordinates = parse_coordinates(line)
                # pdb 是按 z, y, x 排列的
                z = int(coordinates[0] - origin[0]) * 2
                y = int(coordinates[1] - origin[1]) * 2
                x = int(coordinates[2] - origin[2]) * 2
                amino_acid = parse_amino_acid(line)
                atom_type = parse_atom(line)
                # pass C, N, O
                if atom_type != 'C' and atom_type != 'N' and atom_type != 'O':
                    label_sphere(seg_label, [x, y, z], amino_acids.index(amino_acid) + 1, 3)
    return seg_label


def get_amino_acid_coordinates(map_path, pdb_path):
    normalized_map = mrcfile.open(map_path, mode='r')
    origin = normalized_map.header.origin.item(0)
    shape = normalized_map.data.shape
    amino_acid_coordinates = []
    pad = 2
    with open(pdb_path, 'r') as pdb_file:
        amino_acid_num = 'None'
        amino_acid = 'None'
        x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
        for line in pdb_file:
            if line.startswith("ATOM"):
                if amino_acid_num == 'None':
                    amino_acid_num = parse_amino_acid_num(line)
                    amino_acid = parse_amino_acid(line)
                if parse_amino_acid_num(line) != amino_acid_num:
                    [x1, x2, y1, y2, z1, z2] = [x1 - pad, x2 + pad, y1 - pad, y2 + pad, z1 - pad, z2 + pad]
                    if amino_acid in amino_acids and x1 >= 0 and y1 >= 0 and z1 >= 0 \
                            and x2 < shape[0] and y2 < shape[1] and z2 < shape[1]:
                        # 0 is background
                        amino_acid_coordinates.append([x1, x2, y1, y2, z1, z2, amino_acids.index(amino_acid) + 1])
                    amino_acid = parse_amino_acid(line)
                    amino_acid_num = parse_amino_acid_num(line)
                    x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
                # pass C, N, O
                atom_type = parse_atom(line)
                if atom_type != 'C' and atom_type != 'N' and atom_type != 'O':
                    coordinates = parse_coordinates(line)
                    # pdb 是按 z, y, x 排列的
                    z = int(coordinates[0] - origin[0]) * 2
                    y = int(coordinates[1] - origin[1]) * 2
                    x = int(coordinates[2] - origin[2]) * 2
                    x1 = min(x1, x)
                    x2 = max(x2, x)
                    y1 = min(y1, y)
                    y2 = max(y2, y)
                    z1 = min(z1, z)
                    z2 = max(z2, z)
            # add the last amino acid
            elif amino_acid_num != 'None':
                [x1, x2, y1, y2, z1, z2] = [x1 - pad, x2 + pad, y1 - pad, y2 + pad, z1 - pad, z2 + pad]
                if amino_acid in amino_acids and x1 >= 0 and y1 >= 0 and z1 >= 0 \
                        and x2 < shape[0] and y2 < shape[1] and z2 < shape[1]:
                    amino_acid_coordinates.append([x1, x2, y1, y2, z1, z2, amino_acids.index(amino_acid) + 1])
                amino_acid = 'None'
                amino_acid_num = 'None'
                x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
    return amino_acid_coordinates


def crop_map_and_save_labels(map_path, pdb_path, output_path):
    amino_acid_coordinates = get_amino_acid_coordinates(map_path, pdb_path)

    full_image = mrcfile.open(map_path, mode='r').data
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size), dtype=np.float64)
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = full_image

    seg_label = get_seg_label(map_path, pdb_path).data
    padded_image_seg = np.zeros(padded_image.shape, dtype=np.uint8)
    padded_image_seg[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = seg_label

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        chunk_seg = padded_image_seg[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        chunk_coordinates = []
        r_x = cur_x - box_size
        r_y = cur_y - box_size
        r_z = cur_z - box_size
        for x1, x2, y1, y2, z1, z2, amino_acid_id in amino_acid_coordinates:
            if x1 >= r_x and x2 <= r_x + box_size and y1 >= r_y \
                    and y2 <= r_y + box_size and z1 >= r_z and z2 <= r_z + box_size:
                chunk_coordinates.append([x1 - r_x, x2 - r_x, y1 - r_y, y2 - r_y, z1 - r_z, z2 - r_z, amino_acid_id])
        if len(chunk_coordinates) > 0:
            # save chunk and coordinates
            chunk_name = pdb_path.split('/')[-1].split('.')[0] + '_' + str(int(r_x / core_size)) + '_' \
                         + str(int(r_y / core_size)) + '_' + str(int(r_z / core_size))
            np.save(os.path.join(output_path, chunk_name), chunk)
            np.save(os.path.join(output_path, '{}_seg'.format(chunk_name)), chunk_seg)
            with open(os.path.join(output_path, chunk_name + '.txt'), 'w') as label_file:
                for coordinate in chunk_coordinates:
                    label_file.write(','.join([str(i) for i in coordinate]) + '\n')
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point
                cur_x = start_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMdata_dir', type=str, required=True)
    parser.add_argument('--pp_dir', type=str, required=True)
    args = parser.parse_args()

    cf = configs()
    box_size = cf.box_size
    core_size = cf.core_size

    if os.path.exists(args.pp_dir):
        shutil.rmtree(args.pp_dir)
    os.makedirs(args.pp_dir, exist_ok=True)

    for pdb_id in os.listdir(args.EMdata_dir):
        map_path = os.path.join(args.EMdata_dir, pdb_id, 'simulation/normalized_map.mrc')
        pdb_path = os.path.join(args.EMdata_dir, pdb_id, 'simulation/{}.rebuilt.pdb'.format(pdb_id))
        if os.path.exists(map_path) and os.path.exists(pdb_path):
            crop_map_and_save_labels(map_path, pdb_path, args.pp_dir)
            print('finish {}'.format(pdb_id))
