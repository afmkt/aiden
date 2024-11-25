
from src.backend import router as recog_router
from src.user import router as user_router
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends
import solara.server.fastapi
from fastapi.security import OAuth2PasswordBearer
from src.db import create_db_tables
from contextlib import asynccontextmanager

load_dotenv()


@asynccontextmanager
async def lifespan(app:FastAPI):
    create_db_tables()
    yield


tags_metadata = [
    {
        'name': 'user',
        'description': 'Operations with users.'
    },
    {
        'name': 'recognition',
        'description': 'Recognize dental clinical images.'
    }
]

def global_dep():
    pass


app = FastAPI(
    title='AiDen',
    description='''
## Aiden Clinical Image Recognition API

* Upload an image and get an annotated image back
* Upload an image and receive the annotations in JSON.
''',
    version='0.0.1',
    contact = {
        "name": "Michael Shen",
        "email": 'michael@esacca.com'
    },
    license_info = {
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags = tags_metadata,
    openapi_url="/api/v1/openapi.json",
    docs_url = "/api/v1/docs",
    redoc_url = '/api/v1/redoc',
    lifespan=lifespan,
    dependencies=[Depends(global_dep)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allow these origins
    allow_credentials=True,  # Allow cookies or Authorization headers
    allow_methods=["*"],  # Allowed HTTP methods
    allow_headers=["*"],  # Allowed headers
)
app.include_router(recog_router, prefix='/api/v1')
app.include_router(user_router, prefix='/api/v1')
app.mount('/web/', app = solara.server.fastapi.app)

