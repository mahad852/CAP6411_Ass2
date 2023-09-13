from consts import MODEL_BIT_L_RES, MODEL_EFF_NET_L2, MODEL_VIT_H_14, MODEL_VIT_L_16
import torch


def get_model_constant(m:str):
    if m == 'vit_l_16':
        return MODEL_VIT_L_16
    elif m == 'vit_h_14':
        return MODEL_VIT_H_14
    elif m == 'bit_l_res':
        return MODEL_BIT_L_RES
    elif m == 'eff_net_l2':
        return MODEL_EFF_NET_L2
    else:
        raise ValueError('Invalid model_type:', m, '; Please pass a valid model name.')
    
def save_model(model, path:str):
    torch.save(model.state_dict(), path)

def load_model(model, path:str):
    model.load_state_dict(torch.load(path))
    return model

def freeze_layers_except_last(model):
     # Freeze all layers except the last one
    for idx, child in enumerate(model.children()):
        if idx < len(list(model.children())) - 1:  # Exclude the last layer
            for param in child.parameters():
                param.requires_grad = False
    
    return model
