import os
import shutil


def get_lost_pdb_ids():
    pdb_ids = os.listdir(EMdata_dir)
    lost_pdb_ids = []
    list.sort(pdb_ids)
    for pdb_id in pdb_ids:
        map_path = os.path.join(EMdata_dir, pdb_id, 'simulation/normalized_map.mrc')
        pdb_path = os.path.join(EMdata_dir, pdb_id, 'simulation/{}.rebuilt.pdb'.format(pdb_id))
        if not os.path.exists(map_path) or not os.path.exists(pdb_path):
            lost_pdb_ids.append(pdb_id)
            shutil.rmtree(os.path.join(EMdata_dir, pdb_id))
    return lost_pdb_ids


def remove_lost_pdb_ids(pdb_ids_to_remove):
    with open(old_txt, 'r') as old_txt_file, open(new_txt, 'w') as new_txt_file:
        old_pdbs = []
        for line in old_txt_file:
            for pdb_id in line.split(','):
                if len(pdb_id) == 4:
                    old_pdbs.append(pdb_id)
        print('before remove {} pdbs'.format(len(old_pdbs)))
        new_pdbs = [i for i in old_pdbs if i not in pdb_ids_to_remove]
        print('after remove {} pdbs'.format(len(new_pdbs)))
        for i in range(0, len(new_pdbs), 20):
            new_txt_file.write(','.join(new_pdbs[i:i + 20]) + '\n')


if __name__ == '__main__':
    dataset = '400_500'
    EMdata_dir = '/mnt/data/zxy/amino-acid-detection/EMdata_dir/{}'.format(dataset)

    old_txt = '/home/zxy/cryoem-tianchi/dataset/csv/{}.txt'.format(dataset)
    new_txt = '/home/zxy/cryoem-tianchi/dataset/csv/{}_remove.txt'.format(dataset)

    lost_pdb_ids = get_lost_pdb_ids()
    print('lost {} pdbs'.format(len(lost_pdb_ids)))
    remove_lost_pdb_ids(lost_pdb_ids)
