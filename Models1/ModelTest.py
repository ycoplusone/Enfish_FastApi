from sqlalchemy import Column , Integer , String , Boolean , DateTime , ForeignKey,func , Text
from sqlalchemy.orm import relationship,backref
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "user"
    id              = Column(Integer    , primary_key=True , autoincrement=True)
    user_nm         = Column(String     , unique=False, nullable=False)
    password        = Column(String     , nullable=False)
    email           = Column(String     , unique=True, nullable=False)    
    create_date     = Column(DateTime   , nullable=True , default= datetime.now , comment='생성일시')

class Question(Base):
    __tablename__   = "question"
    id              = Column(Integer    , primary_key=True  , index=True , autoincrement=True)
    subject         = Column(String     , nullable=False)
    content         = Column(Text       , nullable=False)
    create_date     = Column(DateTime   , nullable=False)
    '''
    user_id         = Column(Integer    , ForeignKey("user.id") , nullable=True)
    user            = relationship("User"
                                   , backref = backref("ansquestion_userswers")
                                ) #  자식 객체 붙임.    
    '''
class Answer(Base):
    __tablename__   = "answer"
    id              = Column(Integer    , primary_key=True  , index=True , autoincrement=True)
    content         = Column(Text       , nullable=False)
    create_date     = Column(DateTime   , nullable=False)
    question_id     = Column(Integer    , ForeignKey("question.id")) # join 식부분.
    question        = relationship("Question"
                                   , backref = backref("answers" , order_by="Answer.create_date.desc()")
                                ) #  자식 객체 붙임.
    '''
    user_id         = Column(Integer    , ForeignKey("user.id"), nullable=True)
    user            = relationship("User"
                                   , backref = backref("answer_users")
                                ) #  자식 객체 붙임.        
    '''


