from torch.utils.data import Dataset
import os
import nibabel as nib
from sklearn.model_selection import train_test_split

class ProcessedKit23TorchDataset(Dataset):
    def __init__(self, dataset_dir: str="./dataset/affine_transformed", train_data:bool=True, test_size=0.25, transform=None):

        self.in_dataset_dir = dataset_dir
        self.case_dirs = []
        self.case_names = []
        self.transform =transform
        all_cases = sorted(os.listdir(self.in_dataset_dir))
        train_cases, test_cases = train_test_split(all_cases, test_size=test_size, random_state=42)
        cases_to_load = test_cases
        if train_data:
            cases_to_load = train_cases
                           
        for case_name in cases_to_load:
            if case_name.startswith("case"):
                self.case_dirs.append(os.path.join(self.in_dataset_dir, case_name))
                self.case_names.append(case_name)
        
        # self.transform = transform
        # self.target_transform = target_transform
    
    def __len__(self):
        return len(self.case_dirs)
    
    def __getitem__(self, idx):
        # case_name = self.case_names[idx]
        case_dir = self.case_dirs[idx]
        case_img_path = '/'.join([case_dir, "imaging.nii.gz"])
        case_seg_path = '/'.join([case_dir, "segmentation.nii.gz"])
        if self.transform:
            img_loader = nib.load(case_img_path).get_fdata().double()[None]
            seg_loader = nib.load(case_seg_path).get_fdata().double()[None]
        else:
            img_loader = nib.load(case_img_path).get_fdata()[None]
            seg_loader = nib.load(case_seg_path).get_fdata()[None]
            return img_loader, seg_loader