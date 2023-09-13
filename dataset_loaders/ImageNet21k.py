import os
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader

from torchvision.datasets.vision import VisionDataset
from dataset_loaders.datasset_config import DatasetConfig

class ImageNet21k(VisionDataset):
    """
    Custom Vision Dataset class.
    """

    def __init__(self, root, config : DatasetConfig, transform=None, target_transform=None, loader=default_loader, is_train = False, split = 0.7):
        """
        Args:
            root (str): Root directory where the dataset folder exists.
            config (DatasetConfig): contains important information with regards to the dataset (e.g: file paths)
            transform (callable, optional): A function/transform to apply to the input data.
            target_transform (callable, optional): A function/transform to apply to the target data.
            loader (callable, optional): A function to load an image from a file.
        """
        super(ImageNet21k, self).__init__(root, transform=transform,
                                                  target_transform=target_transform)

        self.data = []  # List to store image paths
        self.targets = []  # List to store corresponding labels
        
        self.is_train = is_train
    
        self.config = config

        if not 0 <= split <= 1:
            raise ValueError("The 'split' parameter must be in the range [0, 1]")
        self.split = split

        self._load_dataset()
        self.loader = loader
        
    def _load_dataset(self):
        """
        This method should populate self.data and self.targets with appropriate values.
        """
        # Example: Scanning the root directory for image files and labels
        # Replace this with your own logic

        all_labels = os.listdir(self.root)
        
        images = []
        labels = []
        label_to_idx = self.config.get_label_to_id()

        for label in all_labels:
            if label not in label_to_idx:
                continue

            for image_file_name in os.listdir(os.path.join(self.root, label)):
                idx = label_to_idx[label]
                images.append(os.path.join(self.root, label, image_file_name))
                labels.append(idx)
        
        if self.is_train:
            self.data, _, self.targets, _ = train_test_split(images, labels, test_size=1-self.split, random_state=1, shuffle=True)
        else:
            _, self.data, _, self.targets = train_test_split(images, labels, test_size=1-self.split, random_state=1, shuffle=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data point to fetch.

        Returns:
            tuple: (image, target) where target is the label of the data point.
        """
        img_path = self.data[index]
        target = self.targets[index]

        try:
            # Load the image and apply transformations
            image = self.loader(img_path)
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = torch.ToTensor()(image)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
        
            return image, target
        
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            # Adjust the following line to your specific needs.
            return self.__getitem__(index + 1)  # Skip to the next item

        

    def __len__(self):
        """
        Returns the number of data points in the dataset.
        """
        return len(self.data)