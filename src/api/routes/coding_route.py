from fastapi import APIRouter
from pydantic import BaseModel
from src.models.loader import model_loader
from src.tokenizer.loader import  tokenizer_loader
from src.services.coding import ARInference

coding_router = APIRouter()
class Parameters(BaseModel):
    code:str
    max_length:int
    temperature:float
    top_p : float
    repetition_penalty: float
    variant:str
    
@coding_router.post("/coding")
def coding_model_endpoint(data:Parameters)->str:
    tokenizer = tokenizer_loader("coding")
    model = model_loader(model_name="coding",variant=data.variant)
    inference=ARInference(sp=tokenizer,model=model,prompt=data.code,reptition_penalty=data.repetition_penalty,top_p=data.top_p,max_length=data.max_length,temperature=data.temperature)
    return inference  