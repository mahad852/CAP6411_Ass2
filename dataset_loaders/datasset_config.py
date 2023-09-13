import os
from consts import DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET, DATASET_IMAGENET_REAL, DATASET_OXFORD102, DATASET_OXFORDIIIT
from consts import ENV_LOCAL, ENV_NEWTON
import json

class DatasetConfig(object):
    def __init__(self, dataset_name, env):
        self.dataset_name = dataset_name
        self.env = env

        self.dataset_dir = None
        self.id_to_label = None
        self.label_to_id = None
    
    def get_name(self) -> int:
        return self.dataset_name
    
    def get_batch_size(self) -> int:
        return 10 if self.env == ENV_LOCAL else 128

    def get_dataset_dir_newton(self) -> str:
        if self.dataset_name in [DATASET_CIFAR10, DATASET_CIFAR100]:
            return os.path.join(os.path.sep, 'lustre', 'fs1', 'groups', 'course.cap6411')
        elif self.dataset_name in [DATASET_OXFORD102, DATASET_OXFORDIIIT]:
            return os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        elif self.dataset_name == DATASET_IMAGENET:
            return os.path.join(os.path.sep, 'datasets', 'ImageNet', 'data', 'winter21_whole')
        elif self.dataset_name == DATASET_IMAGENET_REAL:
            return os.path.join(os.path.sep, 'datasets', 'ImageNet2012nonpub/')

    def get_dataset_dir_local(self) -> str:
        if self.dataset_name in [DATASET_CIFAR10, DATASET_CIFAR100]:
            return os.path.join('/home/mahad/Downloads/CAP6411/ass1/data')
        elif self.dataset_name in [DATASET_OXFORD102, DATASET_OXFORDIIIT]:
            return os.path.join(os.path.dirname(__file__), '..', '..')
        elif self.dataset_name == DATASET_IMAGENET:
            return os.path.join(os.path.dirname(__file__), '..', '..', 'ImageNet')
        elif self.dataset_name == DATASET_IMAGENET_REAL:
            return os.path.join(os.path.dirname(__file__), '..', '..', 'ImageNetReal')

    def get_dataset_dir(self) -> str:
        if self.dataset_dir:
            return self.dataset_dir

        if self.env == ENV_LOCAL:
            self.dataset_dir = self.get_dataset_dir_local()
        elif self.env == ENV_NEWTON:
            self.dataset_dir = self.get_dataset_dir_newton()
        else:
            raise ValueError('Environment needs to be either newton or local. Incorrect value provided')
        
        return self.dataset_dir
    
    def get_id_to_label(self) -> dict[int:any]|dict[int:str]:
        if self.id_to_label:
            return self.id_to_label

        self.set_mappings()
        return self.id_to_label
    
    def get_label_to_id(self) -> dict[any:int]|dict[str:int]:
        if self.label_to_id:
            return self.label_to_id
        
        self.set_mappings()
        return self.label_to_id
    

    def set_mappings(self) -> None:
        if self.dataset_name in [DATASET_IMAGENET, DATASET_IMAGENET_REAL]:
            with open(os.path.join(os.path.dirname(__file__), 'mappings', 'imagenet_class_index.json')) as f:
                idx_to_label = json.load(f)
            self.id_to_label = {int(idx):idx_to_label[idx][1] for idx in idx_to_label.keys()}
            self.label_to_id = {idx_to_label[idx][0]:int(idx) for idx in idx_to_label.keys()}
        elif self.dataset_name == DATASET_CIFAR10:
            self.label_to_id = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
            self.id_to_label = {self.label_to_id[label]:label for label in self.label_to_id.keys()}
        elif self.dataset_name == DATASET_CIFAR100:
            str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 
                          'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 
                          'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 
                          'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
                          'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 
                          'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 
                          'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                          'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
                          'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
            
            self.id_to_label = {i:label for i, label in enumerate(str_labels)}
            self.label_to_id = {label:i for i, label in enumerate(str_labels)}

        elif self.dataset_name == DATASET_OXFORD102:
            with open(os.path.join(os.path.dirname(__file__), 'mappings', 'cat_to_name.json')) as f:
                idx_to_label = json.load(f)
            self.id_to_label = {int(idx) - 1:idx_to_label[idx] for idx in idx_to_label.keys()}
            self.label_to_id = {idx_to_label[idx]:int(idx) - 1 for idx in idx_to_label.keys()}
        
        elif self.dataset_name == DATASET_OXFORDIIIT:
            self.label_to_id = {'Abyssinian': 0, 'American Bulldog': 1, 'American Pit Bull Terrier': 2, 'Basset Hound': 3, 'Beagle': 4, 'Bengal': 5, 'Birman': 6, 'Bombay': 7, 'Boxer': 8, 'British Shorthair': 9, 'Chihuahua': 10, 'Egyptian Mau': 11, 'English Cocker Spaniel': 12, 'English Setter': 13, 'German Shorthaired': 14, 'Great Pyrenees': 15, 'Havanese': 16, 'Japanese Chin': 17, 'Keeshond': 18, 'Leonberger': 19, 'Maine Coon': 20, 'Miniature Pinscher': 21, 'Newfoundland': 22, 'Persian': 23, 'Pomeranian': 24, 'Pug': 25, 'Ragdoll': 26, 'Russian Blue': 27, 'Saint Bernard': 28, 'Samoyed': 29, 'Scottish Terrier': 30, 'Shiba Inu': 31, 'Siamese': 32, 'Sphynx': 33, 'Staffordshire Bull Terrier': 34, 'Wheaten Terrier': 35, 'Yorkshire Terrier': 36}
            self.id_to_label = {self.label_to_id[label]:label for label in self.label_to_id.keys()}
            
        

