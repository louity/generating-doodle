import numpy as np
import os
'''
Purpose of this script is to take a list of file name with extension .npz serving as input of the
sketchrnn and put together the categories.
Input:
-----
    - l_category : list of file name to merge
    - name_mixture : name attributed to this mixture
'''
l_category = ['cat.npz', 'broccoli.npz', 'car.npz']
name_mixture = 'broccoli_car_cat.npz'
if not all(file[-4:] == '.npz' for file in l_category):
    raise ValueError('One of the filename is not a .npz extension')
files_admissible = os.listdir(os.curdir)
if not all(file in files_admissible for file in l_category):
    raise ValueError('One of the filemane is not in the directory')

# Check that the list l_category countains who we needs

def mix_category(l_category, name_mixture):
    # check that name_mixture as indeed a .npz extension
    if name_mixture[-4:] != '.npz':
        raise ValueError('name_mixture should have a .npz extension')
    l_train = []
    l_test = []
    l_valid = []
    for data_location in l_category:
        dataset = np.load(data_location, encoding='latin1')
        l_train = l_train + list(dataset['train'])
        l_test = l_test + list(dataset['test'])
        l_valid = l_valid + list(dataset['valid'])
    train = np.array(l_train)
    test = np.array(l_test)
    valid = np.array(l_valid)
    np.savez(name_mixture, train=train, test=test, valid=valid)

if __name__ == '__main__':
    mix_category(l_category, name_mixture)
