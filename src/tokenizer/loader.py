import sentencepiece as spm
from huggingface_hub import hf_hub_download
from functools import lru_cache



def download_tokenizer(model_name):
    try:
          path=hf_hub_download(repo_id="THENDO1977/from-scratch-models", filename=f"{model_name}/sp.model")
          return path
    except Exception as e:
         raise ValueError(f"config path error: {e}")
 
@lru_cache     
def tokenizer_loader(model_name):
    path=download_tokenizer(model_name)     
    tokenizer = spm.SentencePieceProcessor(model_file=path)
    return tokenizer
