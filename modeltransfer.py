import warnings


def transfer_model(pretrained_model, model):
    pretrained_dict = pretrained_model.state_dict() if isinstance(pretrained_model,Module)  else pretrained_model# get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict,strict=False)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict,name_array=[''],verbose=True,strict=True):  
    def find_nearest_key(key,key_list):
        depth=len(key.split('.'))
        for d in range(1,depth+1):
            keys=list(filter(lambda x:''.join(x.split('.')[-d:])==''.join(key.split('.')[-d:]), key_list))
            if len(keys)<=1:break
        return keys
    keys_trans = list(model_dict.keys())
    valid_pretrained_keys=list(filter(lambda x:any([f in x for f in name_array ]),list(pretrained_dict.keys())))
    for k in keys_trans:
        keys=find_nearest_key(k,valid_pretrained_keys)
        if strict:
            assert len(keys)==1, f"{k} has no match keys"
        if len(keys)>1:
            warnings.warn(f"{k} has duplicate matching keys:{{'\n'.join(keys)}}\nignored!")
            break
        if len(keys)==0:
            warnings.warn(f"{k} has no matching keys!")
            break
        key=keys[0]
        model_dict[k] = pretrained_dict[key]
        if verbose:
            print(f"{key} ==> {k}")
        valid_pretrained_keys.remove(key)
    return model_dict
