
from sqlmodel import SQLModel, create_engine, Session
import src.admin
import src.user

engine = create_engine("sqlite:///database.db", echo=True)

def create_db_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
