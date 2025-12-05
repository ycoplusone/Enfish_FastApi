from pydantic   import BaseModel , Field , validator
from typing     import Union , List , Optional 
from fastapi    import Query , HTTPException
from datetime   import datetime


class User(BaseModel):
    ''' User의 기본 모델 '''
    seq             : int   # 자동
    client_id       : str   # 옵션
    user_id         : str   # 필수
    user_nm         : str   # 필수
    email           : str   # 필수
    password        : str   # 필수
    tel_phone       : str   # 옵션
    cell_phone      : str   # 옵션
    role            : int   # 옵션
    use_yn          : str   # 옵션
    created_dt      : datetime  # 옵션
    updated_dt      : datetime  # 옵션
    class Config:
        orm_mode = True  # ⭐ 이 설정이 있어야 SQLAlchemy 객체를 직접 반환 가능 ,response_modeld에 연동할수 있다.

class User_Create(BaseModel):
    '''User 생성'''
    user_id         : str   # 필수
    user_nm         : str   # 필수
    email           : str   # 필수
    password        : str   # 필수
    tel_phone       : str   # 옵션
    cell_phone      : str   # 옵션
    @validator('user_id','user_nm','email','password')
    def check_empty(cls , v):
        if not v or v.isspace():
            raise HTTPException(status_code=422 , detail='필수항목을 입력하세요.')
        return v
  
class User_Login(BaseModel):
    '''User 로그인'''
    client_id   : Union[str,None] = None
    user_id     : str
    password    : str

class User_password_chg(BaseModel):
    '''User 비밀번호 변경'''
    user_id         : str   # id
    password_cur    : str   # 현재 비밀번호
    password_new1   : str   # 비밀번호 1
    password_new2   : str   # 비밀번호 2

class User_Update(BaseModel):
    '''User 생성'''
    user_id         : str   # 필수
    user_nm         : str   # 필수
    email           : str   # 필수
    tel_phone       : str   # 옵션
    cell_phone      : str   # 옵션
    @validator('user_id','user_nm','email','cell_phone','tel_phone')
    def check_empty(cls , v):
        if not v or v.isspace():
            raise HTTPException(status_code=422 , detail='필수항목을 입력하세요.')
        return v


class Menu(BaseModel):
    ''' Menu의 기본 모델 '''
    seq             : int
    menu_id         : str
    menu_nm         : str
    parent_nm       : str 
    url_path        : str
    use_yn          : str 
    created_dt      : datetime  # 옵션
    updated_dt      : datetime  # 옵션
    class Config:
        orm_mode = True  # ⭐ 이 설정이 있어야 SQLAlchemy 객체를 직접 반환 가능 ,response_modeld에 연동할수 있다.    


class Menu_Create(BaseModel):
    ''' menu 생성 모델'''
    menu_id         : str
    menu_nm         : str
    parent_nm       : str 
    url_path        : str
    use_yn          : Union[str,None] = None     


class Menu_Update(BaseModel):
    ''' menu 수정 모델 '''
    seq             : int
    menu_id         : str
    menu_nm         : str
    parent_nm       : str 
    url_path        : str
    use_yn          : str 

class Menu_Delete(BaseModel):
    ''' menu 삭제 모델 '''
    seq             : int    


class Role(BaseModel):
    ''' Role의 기본 모델 '''
    seq             : int
    role_nm         : str
    created_dt      : datetime  # 옵션
    updated_dt      : datetime  # 옵션
    class Config:
        orm_mode = True  # ⭐ 이 설정이 있어야 SQLAlchemy 객체를 직접 반환 가능 ,response_modeld에 연동할수 있다.        

class RoleCreate(BaseModel):
    '''권한 생성'''
    role_nm : str

class RoleUpdate(BaseModel):
    '''role 수정 모델'''
    seq             : int
    role_nm         : str
    use_yn          : str

class RoleDelete(BaseModel):
    '''role 삭제 모델'''
    seq             : int

class RoleMenu(BaseModel):
    ''' Role 세부 menu 기본 모델 '''
    role_seq        : int = 0
    menu_seq        : Union[int,None] = 0
    use_yn          : Union[str,None] = 'Y'
    created_dt      : Union[datetime,None] = None  # 옵션
    updated_dt      : Union[datetime,None] = None  # 옵션

class RoleMenuCreate(BaseModel):
    ''' roel menu 생성 '''
    role_seq         : int
    menu_seq         : int
    use_yn           : str = 'Y'

class RoleMenuUpdate(BaseModel):
    ''' roel menu 수정 '''
    role_seq         : int
    menu_seq         : int
    use_yn           : str = 'Y'

class RoleMenuDelete(BaseModel):
    ''' roel menu 삭제 '''
    role_seq         : int
    menu_seq         : int      
