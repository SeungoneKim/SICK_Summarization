import torch

# Function to map weights to extra decoder
def load_checkpoint(orig_model, new_model):

    orig_model_dict = orig_model.state_dict()
    new_model_dict = new_model.state_dict()

    decoder_weights = {k: v for k,v in orig_model_dict.items() if k[0:13]=='model.decoder'}

    name_list = []
    for k,v in decoder_weights.items():
        name_list.append((k,k[0:6]+"extra_"+k[6:]))
    
    extra_decoder_weights = {}
    for old_name, new_name in name_list:
        extra_decoder_weights[new_name] = decoder_weights.pop(old_name)

    new_model_dict.update(extra_decoder_weights)

    new_model.load_state_dict(new_model_dict)

    return new_model