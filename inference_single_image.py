import torch
import os
from dataset_loaders.load_dataset import get_loader
from models.models import get_model

from dataset_loaders.datasset_config import DatasetConfig

from utils.data_utils import get_dataset_constant
from utils.model_utils import get_model_constant, load_model, freeze_layers_except_last
from utils.env_utils import get_env_constant

import argparse
from torch.cuda.amp import autocast

import urllib.parse

from PIL import Image
from torchvision import transforms
from io import BytesIO
import requests


def get_inference_label(model, config: DatasetConfig, image:torch.Tensor):
    model.eval()

    id_to_label = config.get_id_to_label()
    images = image.unsqueeze(0)
    images = images.to('cuda')

    with autocast(dtype=torch.float16):
        outputs = model(images)
        predicted = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)

    idx = predicted[0].item()
    return id_to_label[idx]


def is_valid_url(image_loc):
    try:
        result = urllib.parse.urlparse(image_loc)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def is_dir(image_loc):
    return os.path.exists(image_loc)

def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        return img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def get_image_from_dir(path):
    try:
        img = Image.open(path)
        return img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def apply_transformation(image):
    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform_image(image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "imagenet_real", "flowers102", "oxford_iiit"], default="imagenet_real",
                        help="Which model to evaluate on.", required=True)
    
    parser.add_argument("--model_type", choices=["vit_l_16", "vit_h_14", "eff_net_l2", "bit_l_res"], default="vit_l_16", help="Which variant to use.", required=True)
    
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Where to search for pretrained ViT models.")

    parser.add_argument("--image_loc", type=str, default=None, help="Image URL or path", required=True)

    parser.add_argument("--env", choices=["local", "newton"], default="local", help="Which variant to use.", required=True)

    
    args = parser.parse_args()

    dataset_name = get_dataset_constant(args.dataset)
    model_name = get_model_constant(args.model_type)
    env_name = get_env_constant(args.env)
    
    dataset_config = DatasetConfig(dataset_name, env_name)
    
    model = get_model(model_name, dataset_name)

    if args.pretrained_dir:
        model = load_model(model, args.pretrained_dir)

    model = freeze_layers_except_last(model)

    if is_valid_url(args.image_loc):
        image = get_image_from_url(args.image_loc)
    elif is_dir(args.image_loc):
        image = get_image_from_dir(args.image_loc)
    else:
        raise ValueError('image_loc:', args.image_locl, 'not a valid a valid path to a local image or a URL')

    if not image:
        raise ValueError('Unable to fetch image. Make sure image_loc contains a valid image URL or path to a local image')

    image = apply_transformation(image)

    print('This is an image of a', get_inference_label(model, dataset_config, image))

if __name__ == '__main__':
    main()