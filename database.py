from sqlalchemy import create_engine , MetaData
from sqlalchemy.orm import sessionmaker , declarative_base , Session 
from sqlalchemy import update
from fastapi import Depends
#import contextlib

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db" # 현재 디렉토리에 app.db 생성

# sqlite파일로 연결되는 엔진 객체 생성.
engine = create_engine(
    url=SQLALCHEMY_DATABASE_URL
    , connect_args={"check_same_thread": False} # check_same_thread 한개의 쓰레드에만 연결되어 있다는 옵션.
    , echo=True # sql 로그 생성
)

# 세션팩토리, 세션 객체를 생성하여 DB 작업에 사용
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 

# declarative_base모든 모델이 상속받는 베이스 클래스 생성
Base = declarative_base() 

# sqlite 에서만 문제가 발생해서 적용한다.
naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

Base.metadata = MetaData(naming_convention=naming_convention)
# sqlite 에서만 문제가 발생해서 적용한다.

#@contextlib.contextmanager
def getDb():    
    # 의존성을 주입하기 위한 함수.
    # 향후 세션을 db:Session=Depends(getDb)     형태로 받을수 있다.    
    db : Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()