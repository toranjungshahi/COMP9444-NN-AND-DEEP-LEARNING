# Kidney and Kidney Tumor 3D Image Segmentation
The main objective of this project is to create a deep learning network-based model capable of identifying 
and distinguishing between kidneys, renal tumors, and renal cysts when given a 3D segmented CT scan of a patient.

At the time of doing this project, the methods of identifying kidney tumors are:
- Labor intensive
- Frequently subjective and imprecise
- Resulting in it being difficult to definitively distinguish between malignant and benign tumors.

Our project aims to assist in this process, by creating a reliable method of automated early detection and accurate diagnosis. Helping detect malignant tumors earlier with less labor required.
Our research has found two promising models; UNet and ResNet that are able to solve the task highly efficiently. Our process is to implement and experiment with ResNet and variations of UNet; nnU-Net and UNet 3D models, compare their accuracies in solving the task, then take the more accurate model and modify its parameters and method with the goal of increasing its performance.


## Data and Data Exploration
- Data Set from: the 2023 Kidney and Kidney Tumor Segmentation challenge (abbreviated KiTS23)
- Raw data can be downloaded at https://kits-challenge.org/kits23/ or URL https://kits23-data.s3.us-east-2.amazonaws.com/repo-tarballs/kits23-v0.1.2.tar
- Kidney 3D CT NII medical image ( 588 cases)
- Kits 23 provides browse page for dataset at https://kits-challenge.org/kits21/browse

**Data Structure**
- All data provided are in “.nii.gz” format, which is gzipped NIFTI datatype used to store medical images. 
- Data provided are 3D images in (z, y, x) where they are layers/thickness, width and depth respectively. 
- Python Module *NiBabel* is used to extract data.
- Python Numpy Unique is used to check labels/classes in the data sample. -> see sample information on the right
- To visualise the 3D-image in 2D, case can be plotted at selective layers.

**Preprocess – Scaling and Down-sampling**
To accommodate the time and compute resource constrains, the data has been:
- 3D Image clipped from (zlayer, 512,512) to target space of (128,232,232) for (zlayer, y-axis, x-axis)
- 3D Image padding with -1024 for input data and 0 for segmentation data.
- Uniform affine matrix with (2,2,2) for (z, y, x)

## Usage

### python packages 
torch, numpy, scikit-learn and jupyterlab are the core packages (for now 25 Oct 2023)
If pip is used in virtual environments, package can be installed with `pip install -r requirements.txt` 
```
numpy==1.26.1
torch==2.1.0
scikit-learn==1.3.2
jupyterlab==4.0.7
```

### Data location
- Raw data folder is to be located at ./dataset
- Put the tar file at dataset folder and extract it.

### Folder Structure
At Google Colab, the base_dir is set to "/content/drive/MyDrive/Colab Notebooks/" for following file. This can be changed at notebook. Due to computational requirement, works are mainly done in Google Colab.

```
./dataset
./dataset/affine_transformed
./notebook_imgs/affine_img.png
./notebook_imgs/case_00000_case_00002.png
./classes/config_class.py
./classes/dataset_utils/preprocessor.py
./classes/dataset_utils/toTorchDataset.py
./classes/models/resnet.py
./classes/models/resnet_model_generator.py
./classes/models/unet3d.py
./pretrainedModel/resnet_10_23dataset.pth
./pretrainedModel/resnet_50_23dataset.pth
./training_checkpoints/Model_resnet*
./training_checkpoints/Model_UNET*
main_notebook.ipynb
resample_dataset.ipynb
resnet_training_cuda.ipynb
Unet3D_train_vanila_model.ipynb
```

### Pretrained Weight
- folder /training_checkpoints/ contains our trained RESNET and UNET pretrained weight. 
- folder /pretrainedModel/ contains RESNET pretrained weight from https://github.com/Tencent/MedicalNet

### Transform Dataset
- Transform of dataset can be performed by using resample_dataset.ipynb
- Transformed dataset can be located at ./dataset/affine_tranformed.

### Classes Folder
- Contain helper classes
- Contain model generators

### Model Training

NN-UNET can be cloned from https://github.com/MIC-DKFZ/nnUNet . Installation and Usage is available on the linked repo. We have used 3d_fullres network as a baseline model. 
Below are the details of how NN-UNET is used for this task. 
- Training Setup
```
- First 100 cases of KiTS23 converted to nnU-Net dataset format.
- Train on Google Colab using Tesla T4 GPU
- Split into 80 cases as train and 20 cases as test set.
- Training epochs 100 with 5-fold cross validation, checkpoint every 25 epochs

```
- Hyper Parameters (Fixed Parameters)
```
- Learning rate = 0.01
- Weight decay = 0.00003
- Compound loss = Dice loss + Cross Entropy loss function
- Optimizer  = SGD
- Oversample foreground percent = 0.33
```
- Training Workflow Details
```
- Extract fingerprint of first100 cases of KiTS23 dataset
    "foreground_intensity_properties_per_channel/Modality (CT in this dataset) ": 
    "0": { "max": 2152.0, "mean": 105.07093345744681, "median": 105.0, "min": -251.0, "percentile_00_5": -74.0, "percentile_99_5": 291.0, "std": 75.0807730601272 }
     "shapes_after_crop": [ 128,232,232]  "spacings": [2.0,2.0,2.0] 
- Plan experiment
    Only 3d fullres network architecture used from nnUNet, Below is the configuration for 3d fullres
    "3d_fullres": { "batch_size": 2,"patch_size": [96,160,160],"normalization_schemes":["CTNormalization"],"UNet_class_name": "PlainConvUNet","UNet_base_num_features": 32
    PlainConvUNet as shown in 3d UNet architecture
    Use region-based segmentation
- Pre-processing**
    CTNormalization : Clip values to 0.5 and 99.5 percentile, followed by subtraction of the mean and division by standard deviation. Normalized between [-2.385, 2.476] 
```
- Mean Validation dice of 0.78 was achieved with NN-UNET

*Notebook of NN-UNET has been lost, while cleaning gdrive*

**Notebook for Resnet and Unet 3D training are given below**
```
resnet_training_cuda.ipynb
Unet3D_train_vanila_model.ipynb

```

Notebook below contains the development processes and findings when taking the kits23 challenge as the assignment project.
```
main_notebook.ipynb
```
