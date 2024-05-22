import torch
import collections
import numpy as np
from copy import deepcopy
from yiku.nets.training_utils import weights_init
from yiku.nets.model.Labs.labs import Labs

import os
def load_weights(device,model_path="",pretrained=False,**kwargs):
    kwargs=deepcopy(kwargs)
    load_key, no_load_key, temp_dict = [], [], {}
    if model_path != '' and os.path.exists(model_path):
        metad_dict = torch.load(model_path, map_location=device)
        if isinstance(metad_dict,dict):
            weights=metad_dict["weights"]
            kwargs.update(metad_dict)
            kwargs.__delitem__("weights")
            kwargs.__delitem__("pretrained")
        elif isinstance(metad_dict,collections.OrderedDict):
            weights=metad_dict
            kwargs.__delitem__("pretrained")
        else:
            raise NotImplementedError(f"Unkown weight type:{type(metad_dict)}")
        model = Labs(**kwargs)
        if not pretrained:
            weights_init(model)
        model_dict = model.state_dict()

        for k, v in weights.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        return model,load_key, no_load_key, temp_dict
    else:
        kwargs["pretrained"]=True
        model = Labs(**kwargs)
        return model,load_key, no_load_key, temp_dict,kwargs


