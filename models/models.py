from consts import MODEL_BIT_L_RES, MODEL_EFF_NET_L2, MODEL_VIT_H_14, MODEL_VIT_L_16
from utils.data_utils import get_num_classes
from timm import create_model

# a function to determine whether a string is a URL or not

def get_model(model_name, dataset_name):
    num_classes = get_num_classes(dataset_name)
    if model_name == MODEL_VIT_L_16:
        model = create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
    elif model_name == MODEL_VIT_H_14:
        model = create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True, num_classes=num_classes)
    elif model_name == MODEL_EFF_NET_L2:
        model = create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
    elif model_name == MODEL_BIT_L_RES:
        model = create_model('resnetv2_50x3_bit.goog_in21k_ft_in1k', pretrained=True, num_classes=num_classes)

    model.cuda()
    return model
