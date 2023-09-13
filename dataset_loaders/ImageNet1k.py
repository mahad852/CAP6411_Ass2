import os
import torch
import torch.utils.data
from torchvision.datasets.folder import default_loader

from torchvision.datasets.vision import VisionDataset
from dataset_loaders.datasset_config import DatasetConfig

class ImageNet1k(VisionDataset):
    """
    Custom Vision Dataset class.
    """

    def __init__(self, root, config:DatasetConfig, transform=None, target_transform=None, loader=default_loader, is_train = False):
        """
        Args:
            root (str): Root directory where the dataset folder exists.
            config (DatasetConfig): contains important information with regards to the dataset (e.g: file paths)
            transform (callable, optional): A function/transform to apply to the input data.
            target_transform (callable, optional): A function/transform to apply to the target data.
            loader (callable, optional): A function to load an image from a file.

        Note:
            You need to implement your data loading logic here, such as reading image paths and labels,
            and storing them appropriately.
        """
        super(ImageNet1k, self).__init__(root, transform=transform,
                                                  target_transform=target_transform)

        # Initialize your dataset here
        self.data = []  # List to store image paths
        self.targets = []  # List to store corresponding labels
        
        self.is_train = is_train
    
        self.config = config


        # Example: You can scan the root directory to populate self.data and self.targets
        # Replace this with your dataset-specific logic
        self._load_dataset()
        self.loader = loader
        
    def _load_dataset(self):
        """
        This method should populate self.data and self.targets with appropriate values.
        You should implement your dataset-specific logic here.
        """
        # Example: Scanning the root directory for image files and labels
        # Replace this with your own logic

        path = os.path.join(self.root, 'train' if self.is_train else 'validation')
        all_labels = os.listdir(path)
        label_to_idx = self.config.get_label_to_id()

        for label in all_labels:
            for image_file_name in os.listdir(os.path.join(path, label)):
                idx = label_to_idx[label]
                self.data.append(os.path.join(path, label, image_file_name))
                self.targets.append(idx)

        self.data = self.data
        self.targets = self.targets
        
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data point to fetch.

        Returns:
            tuple: (image, target) where target is the label of the data point.
        """
        img_path = self.data[index]
        target = self.targets[index]

        # Load the image and apply transformations
        image = self.loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.ToTensor()(image)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        """
        Returns the number of data points in the dataset.
        """
        return len(self.data)
