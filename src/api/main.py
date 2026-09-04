from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.routes.translation_route import translation_router
from src.api.routes.coding_route import coding_router
from src.api.routes.post_route import  writing_router
app = FastAPI()
app.include_router(translation_router)
app.include_router(coding_router)
app.include_router(writing_router)

app.mount(path='/', app=StaticFiles(directory="frontend", html=True), name='static')