import os
import shutil
import sys
import functools, operator
from glob import glob
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocessing of SMD dataset, pretrain or end-to-end.')
parser.add_argument('--data_dir', '-d', type=str,
                    default='../data/vexas_cnn_data/raw',
                    help='Absolute path to the folder containing the dataset and where logs will be created.')
args = parser.parse_args()

def ignore_files(dir,files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def copy_structure(src, dest):
    shutil.copytree(src, dest, ignore=ignore_files)

def augment_image(img, mode='all'):
    """
    Perform data augmentation on images for pretrain set
    :param img: image to augment, numpy array
    :param mode: Number of corner permutations for training set augmentation
    """
    if mode == 'all':
        # Perform all corner permutations
        augmented_img_list = [img]
        augmented_img_list.append(np.flip(img, 0))
        augmented_img_list.append(np.flip(img, 1))
        augmented_img_list.append(np.flip(img, (1,0)))
        augmented_img_list.append(np.rot90(img))
        augmented_img_list.append(np.flip(np.rot90(img), 0))
        augmented_img_list.append(np.flip(np.rot90(img), 1))
        augmented_img_list.append(np.rot90(img,3))
    elif mode == 'semi':
        # Only perform vertical and horizontal flip
        augmented_img_list = [img]
        augmented_img_list.append(np.flip(img, 0))
        augmented_img_list.append(np.flip(img, 1))

    return augmented_img_list

def save_img_reid(df: pd.DataFrame, data_dir: str, path: str):
    """"
    Save imgs to processed dataset, renaming with patient id
    :param img_list: list of images to save
    :param path: str, output path
    """
    img_list = df['path'].values
    one_hot_label = df.iloc[:,1:6].values
    augmented_img_list= ['']*len(img_list)

    copy_structure(data_dir+'/train/', path+'/train/')

    n_repeats = [0]*len(img_list)
    for i in tqdm(range(len(img_list))) :
        img = cv2.imread(os.path.join(data_dir, img_list[i].replace('./','')))

        if (one_hot_label[i,1] == 1) or (one_hot_label[i,2] == 1) :
            augmented_imgs = augment_image(img, 'semi')
        else :
            augmented_imgs = augment_image(img)
            
        for j, augmented_img in enumerate(augmented_imgs):
            cv2.imwrite(path + img_list[i].replace('.jpg',f'{chr(65 + j)}.jpg'), augmented_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        augmented_img_list[i] = [img_list[i].replace('.jpg',f'{chr(65 + j)}.jpg') for j in range(len(augmented_imgs))]
            
        n_repeats[i] = len(augmented_imgs)

    new_df = df.copy()
    new_df = new_df.loc[new_df.index.repeat(n_repeats)].reset_index(drop=True)
    new_df['path'] = functools.reduce(operator.iconcat, augmented_img_list, [])
    return new_df
    
def preprocess_multilabel(data_path):
    """
    Main function for pretrain experiment. Upon execution, does all necessary operations and 
    creates a dataset ready to train upon with logs.
    """
    #Get absolute path to data dir
    data_path = os.path.abspath(data_path)

    #Make output dirs
    out_path = data_path.replace('raw', f'processed/')
    os.makedirs(out_path, exist_ok=True)

    #Get path to images of healthy and dysplastic PNNs
    print('\nProcessing train set :')
    df = pd.read_csv(f'{data_path}/train.csv')
    processed_df = save_img_reid(df, data_path, out_path)
    processed_df.to_csv(f'{out_path}train.csv')

    # No augmentations for testing, simply copy log and test dir to processed folder
    print('\nProcessing test set...')
    shutil.copytree(f'{data_path}/test/', f'{out_path}/test/')
    shutil.copy(f'{data_path}/test.csv', f'{out_path}/')
   
    print('')
    print('-'*30)
    print(f'\nDone with preprocessing and log creation in {out_path}.\n')

if __name__ == '__main__':

    preprocess_multilabel(args.data_dir)
