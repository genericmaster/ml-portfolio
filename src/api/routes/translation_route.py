from fastapi import APIRouter
from pydantic import BaseModel
from src.models.loader import model_loader
from src.tokenizer.loader import tokenizer_loader
from src.services.translation import translate
translation_router = APIRouter()

class Parameters(BaseModel):
    sentence:str
    max_length: int
    temperature:float
    top_p:float
    variant:str

@translation_router.post("/translate")
def translation_endpoint(data:Parameters)->str:
    tokenizer = tokenizer_loader("translation")
    model = model_loader(model_name ="translation",variant=data.variant)
    return translate(model=model,tokenizer=tokenizer,english_sentence=data.sentence,temperature=data.temperature,top_p=data.top_p,max_len=data.max_length)
    
    