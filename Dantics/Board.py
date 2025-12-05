from pydantic   import BaseModel , Field 
from typing     import Union , List , Optional 
from fastapi    import Query
from datetime   import datetime

from enum import Enum

class YesNo(str, Enum):
    Y = "Y"
    N = "N"
    def __str__(self):
        return f"{self.value} ({'예' if self.value == 'Y' else '아니오'})"

class Board(BaseModel):
    ''' 보드의 기본 모델 '''
    seq         : int
    content     : str
    user_id     : str
    use_yn      : str  #Union[str,None] = None
    create_dt   : datetime
    class Config:
        orm_mode = True  # ⭐ 이 설정이 있어야 SQLAlchemy 객체를 직접 반환 가능 ,response_modeld에 연동할수 있다.

class BoardList(BaseModel):
    ''' 보드의 리스트 모델 '''
    seq         : int    
    user_id     : str
    create_dt   : datetime
    class Config:
        orm_mode = True  # ⭐ 이 설정이 있어야 SQLAlchemy 객체를 직접 반환 가능 ,response_modeld에 연동할수 있다.

class CreateBoard(BaseModel):
    ''' board 에 글쓰기 '''
    content     : str
    user_id     : str

class UpdateBoard(BaseModel):
    '''수정'''
    seq         : int
    content     : Union[str,None] = None
    user_id     : Union[str,None] = None

class BoardPatch(BaseModel):
    '''수정'''
    seq         : int
    use_yn      : YesNo #Optional[str] = Query(None , enum=['Y','N'])

class BoardSeq(BaseModel):
    '''순서키 하나만 받는다.'''
    seq         : int
