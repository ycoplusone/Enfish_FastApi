from sqlalchemy import Column , Integer , String , Boolean , DateTime,VARCHAR  , ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
'''
class Sys_Users(Base):
    __tablename__ = 'sys_users'
    seq             = Column(Integer        , primary_key=True  , index=True , autoincrement=True , comment='키값' ) # 기본키 , 자동증가설정
    client_id       = Column(VARCHAR(32)    , nullable=True , default='NONE' , comment='클라이언트 Id 향후 업체 인증 아이디로 사용할예정')
    user_id         = Column(VARCHAR(32)    , nullable=False , comment='사번 혹은 id')
    user_nm         = Column(VARCHAR(32)    , nullable=False , comment='이름')
    email           = Column(VARCHAR(256)   , nullable=False , comment='메일주소' )
    password        = Column(VARCHAR(128)   , nullable=False , comment='개인키')
    tel_phone       = Column(VARCHAR(128)   , nullable=True , comment='전화번호')
    cell_phone      = Column(VARCHAR(128)   , nullable=True , comment='휴대전화번호')
    role_id         = Column(Integer        , ForeignKey('sys_roles.seq') , default='basic' , comment='권한') #Column(VARCHAR(32)    , nullable=True , default='basic' , comment='권한' )    
    use_yn          = Column(VARCHAR(1)     , nullable=True , default='Y' , comment='사용여부(Y , N)' )    
    created_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='생성일시')
    updated_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='수정일시')
    user_role       = relationship( 'Role' , back_populates='role_user' )   


class Menu(Base):
    #메뉴 관리
    __tablename__ = 'sys_menus'
    seq             = Column(Integer        , primary_key=True  , index=True , autoincrement=True , comment='키값' ) # 기본키 , 자동증가설정    
    menu_id         = Column(VARCHAR(32)    , nullable=False , comment='메뉴id')
    menu_nm         = Column(VARCHAR(32)    , nullable=False , comment='메뉴명')
    parent_nm       = Column(VARCHAR(32)    , nullable=False , comment='상위명')
    url_path        = Column(VARCHAR(32)    , nullable=False , comment='경로')
    use_yn          = Column(VARCHAR(1)     , nullable=True , default='Y' , comment='사용여부(Y , N)' )    
    created_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='생성일시')
    updated_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='수정일시') 
    menu_rolemenu   = relationship( 'RoleMenu' , back_populates='rolemenu_menu' )   

class Role(Base):
    #권한
    __tablename__   = 'sys_roles'    
    seq             = Column(Integer        , primary_key=True  , index=True , autoincrement=True , comment='키값' ) # 기본키 , 자동증가설정
    role_nm         = Column(VARCHAR(32)    , nullable=False , comment='권한 명칭')
    use_yn          = Column(VARCHAR(1)     , nullable=True , default='Y' , comment='사용여부(Y , N)' )    
    created_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='생성일시')
    updated_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='수정일시')    
    role_user       = relationship( 'User' , back_populates='user_role' ) 
    role_rolemenu   = relationship( 'RoleMenu' , back_populates='rolemenu_role' ) 

class RoleMenu(Base):
    # 권한의 세부사항
    __tablename__   = 'sys_role_menu'
    role_seq        = Column( Integer       , ForeignKey('sys_roles.seq') , primary_key=True )
    menu_seq        = Column( Integer       , ForeignKey('sys_menus.seq') , primary_key=True )
    use_yn          = Column(VARCHAR(1)     , nullable=True , default='Y' , comment='사용여부(Y , N)' )
    created_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='생성일시')
    updated_dt      = Column(DateTime       , nullable=True , default= datetime.now , comment='수정일시')  
    rolemenu_role   = relationship( 'Role'  , back_populates='role_rolemenu' )        
    rolemenu_menu   = relationship( 'Menu'  , back_populates='menu_rolemenu' )        
'''

