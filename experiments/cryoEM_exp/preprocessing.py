import mrcfile

from configs import *
from pdb_utils import *

amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


def get_amino_acid_coordinates(map_path, pdb_path):
    normalized_map = mrcfile.open(map_path, mode='r')
    origin = normalized_map.header.origin.item(0)
    shape = normalized_map.data.shape
    amino_acid_coordinates = []
    with open(pdb_path, 'r') as pdb_file:
        amino_acid = 'None'
        for line in pdb_file:
            if line.startswith("ATOM"):
                if parse_amino_acid(line) != amino_acid:
                    if amino_acid in amino_acids and x1 >= 0 and y1 >= 0 and z1 >= 0 \
                            and x2 < shape[0] and y2 < shape[1] and z2 < shape[1]:
                        amino_acid_coordinates.append([x1, x2, y1, y2, z1, z2, amino_acids.index(amino_acid)])
                    amino_acid = parse_amino_acid(line)
                    x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
                coordinates = parse_coordinates(line)
                # pdb 是按 z, y, x 排列的
                z = int(coordinates[0] - origin[0])
                y = int(coordinates[1] - origin[1])
                x = int(coordinates[2] - origin[2])
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
                z1 = min(z1, z)
                z2 = max(z2, z)
    return amino_acid_coordinates


def crop_map_and_save_coordinates(map_path, pdb_path, output_path):
    amino_acid_coordinates = get_amino_acid_coordinates(map_path, pdb_path)

    full_image = mrcfile.open(map_path, mode='r').data
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size), dtype=np.float64)
    padded_image[box_size:box_size + image_shape[0], box_size:box_size + image_shape[1],
    box_size:box_size + image_shape[2]] = full_image

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        chunk_coordinates = []
        r_x = cur_x - start_point
        r_y = cur_y - start_point
        r_z = cur_z - start_point
        for x1, x2, y1, y2, z1, z2, amino_acid_id in amino_acid_coordinates:
            if x1 >= r_x and x2 <= r_x + box_size and y1 >= r_y \
                    and y2 <= r_y + box_size and z1 >= r_z and z2 <= r_z + box_size:
                chunk_coordinates.append([x1 - r_x, x2 - r_x, y1 - r_y, y2 - r_y, z1 - r_z, z2 - r_z, amino_acid_id])
        if len(chunk_coordinates) > 0:
            # save chunk and coordinates
            chunk_name = pdb_path.split('/')[-1].split('.')[0] + '_' + str(int(r_x / core_size)) + '_' \
                         + str(int(r_y / core_size)) + '_' + str(int(r_z / core_size))
            np.save(os.path.join(output_path, chunk_name), chunk)
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
    cf = configs()
    EMdata_path = cf.EMdata_dir
    output_path = cf.pp_dir
    box_size = cf.box_size
    core_size = cf.core_size

    for pdb_id in os.listdir(EMdata_path):
        map_path = os.path.join(EMdata_path, pdb_id, 'simulation/normalized_map.mrc')
        pdb_path = os.path.join(EMdata_path, pdb_id, '{}.pdb'.format(pdb_id))
        if os.path.exists(map_path) and os.path.exists(pdb_path):
            crop_map_and_save_coordinates(map_path, pdb_path, output_path)
            print('finish {}'.format(pdb_id))
