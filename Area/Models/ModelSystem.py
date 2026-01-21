from sqlalchemy import Column , Integer , String , Boolean , DateTime,VARCHAR  , ForeignKey , Text , Table , Index, func , and_ 
from sqlalchemy.orm import relationship,backref
from sqlalchemy.ext.associationproxy import association_proxy
from database import Base
from datetime import datetime


System_Boards_Voter = Table(
    'system_boards_voter',
    Base.metadata,
    Column('user_id'            , Integer, ForeignKey('system_users.id')        , primary_key=True),
    Column('board_id'           , Integer, ForeignKey('system_boards.id')       , primary_key=True),
)
System_Comments_Voter = Table(
    'system_comments_voter',
    Base.metadata,
    Column('user_id'            , Integer, ForeignKey('system_users.id')        , primary_key=True),
    Column('comment_id'         , Integer, ForeignKey('system_comments.id')     , primary_key=True),
)


class System_Users(Base):
    __tablename__ = 'system_users'
    id              = Column(Integer        , primary_key=True  , index=True , autoincrement=True , comment='키값' ) # 기본키 , 자동증가설정    
    user_email      = Column(VARCHAR(256)   , unique=True       , nullable=False    , comment='메일주소' )     # 접속 키로 사용할 값 
    user_id         = Column(VARCHAR(32)    , nullable=False    , comment='사번 혹은 id') #
    user_nm         = Column(VARCHAR(32)    , nullable=False    , comment='사용자 이름')    
    password        = Column(VARCHAR(128)   , nullable=False    , comment='개인키')
    tel_phone       = Column(VARCHAR(128)   , nullable=True     , comment='전화번호')
    cell_phone      = Column(VARCHAR(128)   , nullable=True     , comment='휴대전화번호')    
    use_yn          = Column(VARCHAR(1)     , nullable=True     , default='Y' , comment='사용여부(Y , N)' )    
    created_dt      = Column(DateTime       , nullable=True     , default= datetime.now , comment='생성일시')
    updated_dt      = Column(DateTime       , nullable=True     , default= datetime.now , comment='수정일시')
    role_id         = Column(VARCHAR(32)    , ForeignKey("system_roles.role_id") ,  nullable=True) 
    role_ref        = relationship('System_Roles'           , foreign_keys=[role_id]    , back_populates = 'user_role' )
    board           = relationship('System_Boards'          , back_populates = 'user' ) 
    board_voter     = relationship('System_Boards'          , back_populates = 'voter' ) 
    comment         = relationship('System_Comments'        , back_populates = 'user' ) 
    comment_voter   = relationship('System_Comments'        , back_populates = 'voter' )
    menu            = relationship('System_Menus'           , foreign_keys='System_Menus.reged_user_id'         , back_populates = 'user' ) 
    role            = relationship('System_Roles'           , foreign_keys='System_Roles.reged_user_id'         , back_populates = 'user' )
    permissions     = relationship('System_Permission'      , foreign_keys='System_Permission.reged_user_id'    , back_populates = 'user' )

class System_Boards(Base):
    __tablename__   = "system_boards"    
    id              = Column(Integer    , primary_key=True  , index=True , autoincrement=True)
    kind            = Column(String     , nullable=False    , index=True , comment='board 구분')
    subject         = Column(String     , nullable=False)
    content         = Column(Text       , nullable=False)
    create_date     = Column(DateTime   , nullable=False , default= datetime.now)    
    modify_date     = Column(DateTime   , nullable=True , comment='수정일시')
    user_id         = Column(Integer    , ForeignKey("system_users.id") , nullable=True)
    user            = relationship('System_Users'       , back_populates = 'board' ) 
    comments        = relationship('System_Comments'    , back_populates = 'board' )
    voter           = relationship('System_Users'       , secondary=System_Boards_Voter , back_populates='board_voter' ) # 추천
    
class System_Comments(Base):
    __tablename__   = "system_comments"
    id              = Column(Integer    , primary_key=True  , index=True , autoincrement=True)
    content         = Column(Text       , nullable=False)
    create_date     = Column(DateTime   , nullable=False)
    modify_date     = Column(DateTime   , nullable=True , comment='수정일시')
    board_id        = Column(Integer    , ForeignKey("system_boards.id") ) # join 식부분.
    board           = relationship('System_Boards' , back_populates = 'comments' ) #  자식 객체 붙임.    comments 로 접근했을시 자식을 조인해서 가져온다    
    user_id         = Column(Integer    , ForeignKey("system_users.id") , nullable=True)
    user            = relationship('System_Users'       , back_populates = 'comment' )
    voter           = relationship('System_Users'       , secondary=System_Comments_Voter , back_populates='comment_voter' )      


class System_Permission(Base):
    __tablename__ = "system_permission"
    role_id         = Column(VARCHAR(32)    , ForeignKey("system_roles.role_id")    , primary_key=True)
    menu_id         = Column(Integer        , ForeignKey("system_menus.id")         , primary_key=True)
    use_yn          = Column(VARCHAR(1)     , nullable=False , default="Y"    , comment="사용여부(Y , N)")
    created_dt      = Column(DateTime, nullable=False, default=datetime.now, comment="생성일시")
    updated_dt      = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now, comment="수정일시")
    reged_user_id   = Column(Integer        , ForeignKey("system_users.id") , nullable=True , comment='최종등록자')
    # N:1 관계 (Permission -> Role/Menu)
    role            = relationship('System_Roles'           , foreign_keys=[role_id]        , back_populates = 'permissions' )
    menu            = relationship('System_Menus'           , foreign_keys=[menu_id]        , back_populates = 'permissions' , order_by="System_Menus.sort_no" )
    user            = relationship('System_Users'           , foreign_keys=[reged_user_id]  , back_populates = 'permissions' )
    #permission_user     = relationship('System_Permission'      , foreign_keys='system_users.role_id'    , back_populates = 'user_permission' )
    __table_args__ = (
        Index("ix_system_permission_role_id", "role_id"),
        Index("ix_system_permission_menu_id", "menu_id"),
        Index("ix_system_permission_use_yn", "use_yn"),
    )

class System_Menus(Base):
    __tablename__ = "system_menus"
    id              = Column(Integer        , primary_key=True, index=True, autoincrement=True)
    menu_nm         = Column(VARCHAR(64)    , nullable=False, comment="메뉴명")
    group_nm        = Column(VARCHAR(64)    , nullable=False, comment="상위명")
    url_path        = Column(VARCHAR(64)    , nullable=False, comment="경로")
    sort_no         = Column(VARCHAR(8)     , nullable=False, default='9999' , comment='정렬순서(앞두자리 GROUP_NM , 뒤두자리 MENU_NM)' )    
    use_yn          = Column(VARCHAR(1)     , nullable=False, default="Y", comment="사용여부(Y , N)")
    created_dt      = Column(DateTime       , nullable=False, default=datetime.now, comment="생성일시")
    updated_dt      = Column(DateTime       , nullable=False, default=datetime.now, onupdate=datetime.now, comment="수정일시")    
    reged_user_id   = Column(Integer        , ForeignKey("system_users.id") , nullable=True , comment='최종등록자')
    user            = relationship('System_Users'       , foreign_keys=[reged_user_id] , back_populates = 'menu')
    permissions     = relationship('System_Permission'  , foreign_keys='System_Permission.menu_id' , back_populates = 'menu')


class System_Roles(Base):
    __tablename__ = "system_roles"
    role_id         = Column(VARCHAR(32)    , primary_key=True  , comment="권한id")
    role_nm         = Column(VARCHAR(32)    , nullable=False    , comment="권한명")
    use_yn          = Column(VARCHAR(1)     , nullable=False    , default="Y", comment="사용여부(Y , N)")
    created_dt      = Column(DateTime       , nullable=False    , default=datetime.now, comment="생성일시")
    updated_dt      = Column(DateTime       , nullable=False    , default=datetime.now, onupdate=datetime.now , comment="수정일시")
    reged_user_id   = Column(Integer        , ForeignKey("system_users.id") , nullable=True , comment='최종등록자')
    user            = relationship('System_Users'               , foreign_keys=[reged_user_id] , back_populates = 'role' )
    permissions     = relationship('System_Permission'          , foreign_keys='System_Permission.role_id'  , back_populates = 'role' )
    user_role       = relationship("System_Users"               , foreign_keys='System_Users.role_id'       ,  back_populates="role_ref")




