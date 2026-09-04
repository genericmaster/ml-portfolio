from huggingface_hub import hf_hub_download
import json

def model_configurator(model_name:str):
   try:
       path=hf_hub_download(repo_id="THENDO1977/from-scratch-models", filename=f"{model_name}/config.json")
   except Exception as e:
      raise ValueError(f"config path error: {e}")
       
   with open(path) as f:
       config = json.load(f)
   return config


