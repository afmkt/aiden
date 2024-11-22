
from src.backend import router as api_router
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter
import solara.server.fastapi

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow these origins
    allow_credentials=True,  # Allow cookies or Authorization headers
    allow_methods=["*"],  # Allowed HTTP methods
    allow_headers=["*"],  # Allowed headers
)
app.include_router(api_router, prefix='/api')
app.mount('/web/', app = solara.server.fastapi.app)


