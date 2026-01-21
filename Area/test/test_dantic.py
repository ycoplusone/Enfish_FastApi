import datetime
from pydantic import BaseModel , field_validator,  EmailStr , ConfigDict
from typing     import Union , List , Optional 

class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class User(ORMBase):
    id      : int
    user_nm : str
    email   : str

    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)
    

class Answer(ORMBase):
    id              : int
    content         : str
    create_date     : datetime.datetime   
    modify_date     : Union[datetime.datetime , None] = None
    user            : Union[ User , None] = None
    question_id     : int
    voter           : list[User] = []
    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)

class AnswerCreate(ORMBase):
    content         : str

    @field_validator('content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class AnswerUpdate(AnswerCreate): # 답변 수정 스키마
    answer_id: int    

class AnswerDelete(ORMBase): # 답변 삭제 스키마
    answer_id: int

class AnswerVote(ORMBase): # 답변 추천
    answer_id: int

class Question(ORMBase):
    id              : int
    subject         : str
    content         : str
    create_date     : datetime.datetime    
    modify_date     : Union[datetime.datetime , None] = None
    answers         : list[Answer] = []
    user            : Union[ User , None] = None
    voter           : list[User] = [] # 추천인 스키마 추가.
    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)

class QuestionCreate(ORMBase):
    subject: str
    content: str

    @field_validator('subject', 'content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class QuestionList(ORMBase):
    total: int = 0
    question_list: list[Question] = []

class QuestionUpdate(QuestionCreate): # 질문 수정 스키마
    question_id: int

class QuestionDelete(ORMBase): # 질문 삭제 스키마
    question_id: int

class QuestionVote(ORMBase): # 질문 추천 
    question_id: int

class UserCreate(ORMBase):
    user_nm         : str
    password        : str
    password_chk    : str
    email           : EmailStr

    @field_validator('user_nm', 'password', 'password_chk', 'email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @field_validator('password_chk')
    def passwords_match(cls, v, values):
        if "password" in values and v != values["password"]:
            raise ValueError("비밀번호가 일치하지 않습니다")
        return v

class Token(ORMBase):
    access_token    : str
    token_type      : str
    user_nm         : str      
    user_id         : str
    email           : str  

