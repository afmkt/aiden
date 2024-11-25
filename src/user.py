from typing import Optional, List, Annotated
from sqlmodel import Field, SQLModel, Session, select
from fastapi import APIRouter, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, OAuth2AuthorizationCodeBearer
from src.db import get_session
import os
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone



router = APIRouter(
    prefix='/user',
    tags=["user"],
)

class Token(SQLModel):
    access_token: str
    token_type: str


class TokenData(SQLModel):
    username: str | None = None

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




SECRET_KEY = os.environ['SECRET_KEY']
ALGORITHM = os.environ['ALGORITHM']
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ['ACCESS_TOKEN_EXPIRE_MINUTES'])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(session: Session, username: str):
    stmt = select(User).where(User.name == username)
    results = session.exec(stmt)
    for r in results:
        return r
    
    



def authenticate_user(session: Session,username: str, password: str)->User:
    user = get_user(session, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(session: Annotated[Session, Depends(get_session)], token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(session, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user


@router.post("/token")
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()], 
        session: Session = Depends(get_session)
    ) -> Token:
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.name}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")




@router.get("/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user

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