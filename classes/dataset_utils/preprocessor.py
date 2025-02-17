import os
from tqdm import tqdm
import nibabel as nib
from nilearn.image import resample_img
import numpy as np

class Preprocess():

    def __init__(self, dataset_in_dir: str, re_dataset_out_dir: str="./dataset/affine_transformed",
                 target_affine=(2, 2, 2), target_shape=(128,232,232)):
        
        self.in_dir = dataset_in_dir
        self.out_dir = re_dataset_out_dir
        target_affine_space = np.array(([0, 0, -target_affine[0], 0],[0, -target_affine[1], 0, 0],[-target_affine[2], 0, 0, 0],[0, 0, 0, 1]))
        self.target_affine = target_affine_space
        self.target_shape = target_shape
        
        self.case_dirs = []
        self.case_names = []
        for case_name in sorted(os.listdir(self.in_dir)):
            if case_name.startswith("case"):
                self.case_dirs.append(os.path.join(self.in_dir, case_name))
                self.case_names.append(case_name)
        if len(self.case_dirs)== 0:
            raise Exception(f"No file found at Dir: {self.in_dir}")
        
        # create dir if not exist
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
            print("Create directory: {}".format(self.out_dir))
        else:
            print("Files output dir: {}".format(self.out_dir))
    
    def resample_datasets(self):
        """
        Transform and resample dataset (input and output)
        """
        for i, case_dir in enumerate(tqdm(self.case_dirs)):
            case_name = self.case_names[i]
            case_img_path = '/'.join([case_dir, "imaging.nii.gz"])
            case_seg_path = '/'.join([case_dir, "segmentation.nii.gz"])
            img_loader = nib.load(case_img_path)
            seg_loader = nib.load(case_seg_path)
            resampled_img_loader = resample_img(img_loader, target_affine=self.target_affine, fill_value=-1024, target_shape=self.target_shape, interpolation='nearest')
            resampled_seg_loader = resample_img(seg_loader, target_affine=self.target_affine, fill_value=0, target_shape=self.target_shape, interpolation='nearest')
            if i == 0:
                print("Output Affine : \n{}".format(resampled_img_loader.affine))
            print("{}: 3D image from shape {} to shape {}".format(case_name, img_loader.shape, resampled_img_loader.shape))
            print("{}: 3D seg from shape {} to shape {}".format(case_name, seg_loader.shape, resampled_seg_loader.shape))
            
            this_case_dir = os.path.join(self.out_dir, case_name)
            if not os.path.isdir(this_case_dir):
                os.mkdir(this_case_dir)
            nib.save(resampled_img_loader, os.path.join(this_case_dir, "imaging.nii.gz"))
            nib.save(resampled_seg_loader, os.path.join(this_case_dir, "segmentation.nii.gz"))


