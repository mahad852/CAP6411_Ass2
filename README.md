# CAP6411_Ass2
Replicating the ViT paper as part of Assignment 2 of the course CAP6411 at UCF.

# Available datasets
The code provides fine tuning and evaluation for 4 different models (more about them below) on 6 different datasets. These include:
1. ImageNet21k Winter 2021 (imagenet)
2. ImageNet1k (imagenet_real)
3. CIFAR10 (cifar10)
4. CIFAR100 (cifar100)
5. Flowers 102 (flowers102)
6. Oxford IIIT Pet (oxford_iiit_pet)

The above datasets have their corresponding name in the brackets. These names are used as the dataset argument when running inference and training.

# Available Models
The code provides with 4 models:
1. ViT Huge - 14 patch, 224x224 (vit_h_14)
2. ViT Large - 16 patch, 224x224 (vit_l_16)
3. Resnet 50x3 (bit_l_res)
4. Efficient Net B2 (eff_net_l2)

The above models have their corresponding name in the brackets. These names are used as the model argument when running inference and training. There is a discrepancy in eff_net_l2 and Efficient Net B2, this is because we originally planned to use the L2 (large 2) model for the efficient net, but unfortunately couldn't find the pretrained weights very easily.

# Dataset config and dataset paths
To run this code, you need to make sure that the paths for the datasets that you provide in the dataset config are valid and have the relevant datasets in the directories. Check out the dataset_loaders/datasset_config.py (there's a typo in the name, will fix) file and the get_dataset_dir_newton() and get_dataset_dir_local() functions for more information. We will also be updating the README and add the format (files/folders/hierarchies) expected for the different datasets.

# Environments
We support 2 environments, namely:
1. Local (local)
2. Newton (newton)

You can add support for more environments, but this would require going through the code and can take some time. The names in brackets represent the environment name being passed when running training and inference.

# Installation
First, create a conda environment named v
'''
conda create --name vit_ass -c conda-forge python=3.9
'''
Then, actiavte the conda environment using (you can skip this step and all the next steps if you are using newtown by instead just running the script.slurm as a job, the script has commands towards the end that run the python code, you can modify it to run the commands of your choice):
'''
conda activate vit_ass
'''
Now that we have created and activated our conda environmnet, we'll install packages using either pip or conda
## Using pip (recommended)
From the root directory, run:
'''
pip install -r requirements_pip.txt
'''
## Using conda:
Form the root directory, run:
'''
conda install --file requirements.txt
'''

## Train
We provide the option to train (fine tune) datasets. All datasets are pretrained over the ImageNet datasets. Here's an example of the train command:
'''
python3 train.py --num_epochs 40 --dataset oxford_iiit --model_type vit_h_14 --env newton --metric_path models/metrics/vit_h_14_oxford_iiit_train.txt --checkpoint_path models/checkpoints/oxford_iiit_viiit_vit_h_14.bin
'''
Here, the num_epochs argument refers to the number of epochs the training would run for (default is 10), the dataset refers to the dataset being trained on, the model_type argument refers to the type of model (check available models section or train.py for more options), env refers to the env and either be "local" or "newton", metric_path refers to the file where metrics (e.g: loss and accuracy after each epoch) would be written, and finally checkpoint_path is where the model would be saved (it is saved after each epoch). There is another parameter pretrained_dir that train.py accepts, which allows the model to load pretrained weights (bin file). If none is specified, pretrained weights from the ImageNet dataset are used.

## Eval
We provide the option to evaluate pretrained models. Here is an example command:
'''
python3 eval.py --dataset flowers102 --model_type vit_l_16 --env newton --metric_path models/metrics/vit_l_16_flowers102.txt --pretrained_dir models/checkpoints/flowers102_vit_l_16.bin
'''
All the arguments here are the same as the train.py file, except the num_epochs argument. Also, the pretrained_dir argument is optional here as well, and by default the model would evaluate the model over the provided dataset using the pretrained ImageNet weights.

## Seeing the outcome of inference for random (sort of) images from the test set:
The following command runs inference over the test set and saves a bunch of random images from the set such that the name of the file is ground_truth.png and there is a predicted class text overlayed on the image:
'''
python3 generate_outputs.py --model_type vit_h_14 --dataset flowers102 --image_folder models/images/flowers102 --env newton --pretrained_dir models/checkpoints/flowers102_vit_h_14.bin
'''
Here the image_folder is a new argument that specifies the directory where the images will be saved.The rest of the arguments are the same as eval.py
