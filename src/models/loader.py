import torch as pt
import torchao
from functools import lru_cache
from huggingface_hub import hf_hub_download
from src.core.config_loader import model_configurator
from src.models.translation.model import Transformer
from src.models.coding.coder import CodingModel



def download_model(model_name:str,variant:str):
    model_dict=model_configurator(model_name=model_name)
    model_variant = model_dict["weights"][variant]
    return model_dict["architecture"],hf_hub_download(repo_id="THENDO1977/from-scratch-models", filename=model_variant)

def _instatiate_model(config_param,model_name):
    if model_name == "translation":
        return Transformer(**config_param)
    elif model_name == "coding":
        return CodingModel(**config_param)

@lru_cache
def model_loader(model_name: str, variant: str):
    model_dict, model_weights = download_model(model_name=model_name, variant=variant)
    model = _instatiate_model(config_param=model_dict, model_name=model_name)
    trained_weights = pt.load(model_weights, map_location=pt.device('cpu'), mmap=True)
    model.load_state_dict(trained_weights)
    del trained_weights  # ← free the buffer immediately
    model.eval() 
    torchao.quantize_(model, torchao.int8_weight_only())
    return model
    

    
    
    
