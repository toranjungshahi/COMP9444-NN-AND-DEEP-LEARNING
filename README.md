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
- Kidney 3D CT NII medical image ( 588 cases)
- Kits 23 provides browse page for dataset at https://kits-challenge.org/kits21/browse

**Data Structure**
- All data provided are in “.nii.gz” format, which is gzipped NIFTI datatype used to store medical images. 
- Data provided are 3D images in (z, y, x) where they are layers/thickness, width and depth respectively. 
- Python Module *NiBabel* is used to extract data.
- Python Numpy Unique is used to check labels/classes in the data sample. -> see sample information on the right
- To visualise the 3D-image in 2D, case can be plotted at selective layers.

## Installation
Step-by-step installation instructions

## Usage
How to use your project with examples

## Technologies
- Tech 1
- Tech 2

## Contributing
How others can contribute (if open-source)