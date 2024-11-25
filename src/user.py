from typing import Optional, List
from sqlmodel import Field, SQLModel, Session, select
from fastapi import APIRouter, HTTPException, Query, Depends
from src.db import get_session



router = APIRouter(
    prefix='/user',
    tags=["user"],
)


class UserBase(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    full_name: str
    email: str = Field(default=None, index=True, unique=True)
    
    
class User(UserBase, table = True):
    hashed_password: Optional[str] = Field(default=None)

class Role(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)





@router.post('/', tags=["user"], response_model=User)
def new_user(*, session: Session = Depends(get_session), usr: UserBase):
    usr = User.model_validate(usr)
    session.add(usr)
    session.commit()
    session.refresh(usr)
    return usr

@router.get('/', response_model= List[User])
def get_users(*, session: Session = Depends(get_session), offset:int = 0, limit:int = Query(default=100, le=100)):
    return session.exec(select(User).offset(offset).limit(limit)).all() 
    

@router.get('/{id}', response_model=User)
def get_one_user(*, session: Session = Depends(get_session), id: int):
    ret = session.get(User, id)
    if not ret:
        raise HTTPException(status_code=404, detail=f'User {id} not found')
    return ret
    
@router.patch('/{id}', response_model=User)
def update_user(*, session: Session = Depends(get_session), id: int, usr: UserBase):
    db_usr = session.get(User, id)
    if not db_usr:
        raise HTTPException(status_code=404, detail=f'User {id} not found')
    usr_data = usr.model_dump(exclude_unset=True)
    db_usr.sqlmodel_update(usr_data)
    session.add(db_usr)
    session.commit()
    session.refresh(db_usr)
    return db_usr

@router.delete('/{id}')
def del_user(*, session: Session = Depends(get_session), id: int):
    usr = session.get(User, id)
    if not usr:
        raise HTTPException(status_code=404, detail=f'User {id} not found')
    session.delete(usr)
    session.commit()
    return {'ok': True}