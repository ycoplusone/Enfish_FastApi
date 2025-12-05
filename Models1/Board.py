from sqlalchemy import Column , Integer , String , Boolean , DateTime,VARCHAR
from datetime import datetime
from database import Base

class board(Base):
    __tablename__ = 'sys_boards'
    seq             = Column(Integer    , primary_key=True , index=True , autoincrement=True  ) # 기본키 , 자동증가설정
    content         = Column(VARCHAR(128)   , nullable=False )
    user_id         = Column(VARCHAR(32)    , nullable=False )
    use_yn          = Column(VARCHAR(1) , nullable=True , default='Y' )
    create_dt       = Column(DateTime   , nullable=True , default= datetime.now )
    

