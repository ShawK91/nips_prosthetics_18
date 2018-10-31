import numpy as np, os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-data_folder', help='Data folder',  required=True)
data_folder = vars(parser.parse_args())['data_folder']



######## READ DATA #########
list_files = os.listdir(data_folder)

corrupt_list = []
for index, file in enumerate(list_files):
        print('Checking file', file)
        try:
            data = np.load(data_folder + file)
            s = data['state'];
            ns = data['next_state'];
            a = data['action'];
            r = data['reward'];
            done_dist = data['done_dist']
        except:
            corrupt_list.append(data_folder + file)
            os.remove(data_folder + file)

print('Corrupt files removed', corrupt_list)




