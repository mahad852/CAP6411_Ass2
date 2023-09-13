from consts import DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET, DATASET_IMAGENET_REAL, DATASET_OXFORD102, DATASET_OXFORDIIIT

def get_dataset_constant(ds):
    if ds == 'imagenet':
        return DATASET_IMAGENET
    elif ds == 'imagenet_real':
        return DATASET_IMAGENET_REAL
    elif ds == 'cifar10':
        return DATASET_CIFAR10
    elif ds == 'cifar100':
        return DATASET_CIFAR100
    elif ds == 'flowers102':
        return DATASET_OXFORD102
    elif ds == 'oxford_iiit':
        return DATASET_OXFORDIIIT
    else:
        raise ValueError('Invalid dataset:', ds, '; Please pass a valid dataset')
    
def get_num_classes(dataset_name):
    if dataset_name == DATASET_CIFAR10:
        return 10
    elif dataset_name == DATASET_CIFAR100:
        return 100
    elif dataset_name == DATASET_IMAGENET:
        return 1000
    elif dataset_name == DATASET_IMAGENET_REAL:
        return 1000
    elif dataset_name == DATASET_OXFORD102:
        return 102
    elif dataset_name == DATASET_OXFORDIIIT:
        return 37