import datetime
from pydantic import BaseModel ,  EmailStr , Field ,field_validator,ConfigDict
from typing     import Union , List , Optional 

class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

#----------------- system_permission : begin
class MenuOut(ORMBase):
    id          : int
    menu_nm     : str
    group_nm    : str
    url_path    : str
    sort_no     : str
    use_yn      : str
    #class Config:
    #    orm_mode = True  
    #model_config = ConfigDict(from_attributes=True)

class RoleOut(ORMBase):
    role_id     : Union[ str , None] = None
    role_nm     : Union[ str , None] = None
    use_yn      : Union[ str , None] = None
    #class Config:
    #    orm_mode = True  
    #model_config = ConfigDict(from_attributes=True)

class PermissionOut(ORMBase):
    role_id     : Union[ str , None] = None
    menu_id     : Union[ int , None] = None
    use_yn      : Union[ str , None] = None
    created_dt  : Union[ datetime.datetime , None] = None
    updated_dt  : Union[ datetime.datetime , None] = None
    role        : Union[ RoleOut , None ] = None 
    menu        : Union[ MenuOut , None ] = None 
    #class Config:
    #    orm_mode = True  
    #model_config = ConfigDict(from_attributes=True)

class UserPermissionsOut(ORMBase):
    user_id     : int
    role_id     : Optional[str]
    permissions : List[PermissionOut] = []
    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)
#----------------- system_permission : end

#----------------- system_users : begin
class System_Users(ORMBase): # 사용자 기본 쿼리폼
    id              : int
    user_email      : str
    user_id         : str
    user_nm         : str
    tel_phone       : str
    cell_phone      : str
    use_yn          : str
    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)

class System_Users_Create(ORMBase): # 사용자 생성
    user_email      : EmailStr
    user_id         : str
    user_nm         : str
    password        : str
    password_chk    : str    
    tel_phone       : str
    cell_phone      : str       

    
    @field_validator('user_nm', 'password', 'password_chk', 'user_email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @field_validator('password_chk')
    def passwords_match(cls, v, values):
        if "password" in values and v != values["password"]:
            raise ValueError("비밀번호가 일치하지 않습니다")
        return v
    

class Token(ORMBase): # 로그인후 발송 토큰 
    access_token    : str
    token_type      : str
    user_email      : str
    user_id         : str
    user_nm         : str
    permissions     : Union[ List[PermissionOut] , None] = None

class System_Users_Update_role(ORMBase): #사용자 권한 업데이트
    user_email      : str
    role_id         : str
    use_yn          : str
    @field_validator('user_email', 'role_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v     

class System_Users_List(ORMBase):
    datas           : list[System_Users] = []

#----------------- system_users : end

#----------------- system_comments : begin
class System_Comments(ORMBase):
    id              : int
    content         : str
    create_date     : datetime.datetime   
    board_id        : int
    modify_date     : Union[ datetime.datetime , None] = None
    user            : Union[ System_Users , None] = None    
    voter           : list[System_Users] = []
    #class Config:
    #    orm_mode = True    

class System_Comments_Create(ORMBase):
    content         : str
    @field_validator('content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class System_Comments_Update(System_Comments_Create): # 답변 수정 스키마
    comment_id : int    

class System_Comments_Delete(ORMBase): # 답변 삭제 스키마
    comment_id : int

class System_Comments_Vote(ORMBase): # 답변 추천
    comment_id: int
#----------------- system_comments : end

#----------------- system_boards : begin
class System_Boards(ORMBase): # 게시판 
    id              : int
    kind            : str
    subject         : str
    content         : str
    create_date     : Union[ datetime.datetime , None] = None
    modify_date     : Union[ datetime.datetime , None] = None
    user            : Union[ System_Users , None] = None
    comments        : list[System_Comments] = []    
    voter           : list[ System_Users ] = [] # 추천인 스키마 추가.
    #class Config:
    #    orm_mode = True    
    #model_config = ConfigDict(from_attributes=True)

class System_Boards_List(ORMBase):
    total           : int = 0
    boards_list     : list[System_Boards] = []

class System_Boards_Create(ORMBase): # 게시판 생성 
    kind    : str
    subject : str
    content : str
    @field_validator('kind','subject', 'content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class System_Boards_Update(ORMBase): # boards 수정
    system_boards_id: int
    subject : str
    content : str
    @field_validator('subject', 'content')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v    

class System_Boards_Delete(ORMBase): # boards 삭제
    system_boards_id: int

class System_Boards_Vote(ORMBase): # 글 추천 
    system_boards_id: int
#----------------- system_boards : end

#----------------- system_menu : start
class System_Menus_Base(ORMBase): # system_menus Base
    id              : int
    menu_nm         : str
    group_nm        : str
    url_path        : str
    use_yn          : str
    sort_no         : str
    created_dt      : Union[ datetime.datetime , None] = None
    updated_dt      : Union[ datetime.datetime , None] = None
    user            : Union[ System_Users , None] = None
    #class Config:
    #    orm_mode = True
    #model_config = ConfigDict(from_attributes=True)

class System_Menus_List(ORMBase): # system_menus list
    datas : list[System_Menus_Base] = []

class System_Menus_Create(ORMBase): # system_menus create
    menu_nm     : str = Field(default="Menu Name"   , description="메뉴명")
    group_nm    : str = Field(default="Group Name"  , description="그룹명")
    url_path    : str = Field(default="url_path"    , description="url")
    sort_no     : str = Field(default="9999"        , description="순서")
    @field_validator('menu_nm','group_nm', 'url_path','sort_no')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v    

class System_Menus_Update(ORMBase): # system_menus update
    id          : int = Field(default=0             )
    menu_nm     : str = Field(default="Menu Name"   , description="메뉴명")
    group_nm    : str
    url_path    : str
    use_yn      : str = Field(default="Y", pattern="^(Y|N)$" , description="사용유무 Y N") 
    sort_no     : str
    @field_validator('menu_nm', 'group_nm','url_path','sort_no')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v     
      
    @field_validator('use_yn')
    def validate_use_yn(cls, v):
        if v not in ("Y", "N"):
            raise ValueError("use_yn must be 'Y' or 'N'")
        return v

class System_Menus_Delete(ORMBase): # system_menus delete
    id          : int
#----------------- system_menus : end

#----------------- system_roles : start
class System_Roles_Base(ORMBase): # System_Roles Base
    role_id         : str
    role_nm         : str    
    use_yn          : str
    created_dt      : Union[ datetime.datetime , None] = None
    updated_dt      : Union[ datetime.datetime , None] = None
    user            : Union[ System_Users , None] = None
    #class Config:
    #    orm_mode = True
    #model_config = ConfigDict(from_attributes=True)

class System_Roles_List(ORMBase): # System_Roles list
    datas : list[System_Roles_Base] = []

class System_Roles_Create(ORMBase): # System_Roles create
    role_id     : str = Field(default="role id"   , description="권한ID 유니크하게")
    role_nm    : str = Field(default="role Name"  , description="권한명")
    @field_validator('role_id','role_nm')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v    

class System_Roles_Update(ORMBase): # System_Roles update
    role_id     : str = Field(default="role_id"             , description="권한id 유니크하게" )
    role_nm     : str = Field(default="role Name"           , description="권한명")
    use_yn      : str = Field(default="Y", pattern="^(Y|N)$"  , description="사용유무 Y N") 
    @field_validator('role_id', 'role_nm')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v     
      
    @field_validator('use_yn')
    def validate_use_yn(cls, v):
        if v not in ("Y", "N"):
            raise ValueError("use_yn must be 'Y' or 'N'")
        return v    

class System_Roles_Delete(ORMBase): # System_Roles delete
    role_id : str
#----------------- system_roles : end

#----------------- system_permission : start
class System_Permission_Base(ORMBase): # System_Roles Base
    role_id         : str
    menu_id         : int
    use_yn          : str
    created_dt      : Union[ datetime.datetime , None] = None
    updated_dt      : Union[ datetime.datetime , None] = None
    user            : Union[ System_Users , None] = None
    role            : Union[ System_Roles_Base , None ] = None
    menu            : Union[ System_Menus_Base , None ] = None
    #class Config:
    #    orm_mode = True
    #model_config = ConfigDict(from_attributes=True)

class System_Permission_List(ORMBase):
    datas : list[System_Permission_Base] = []

class System_Permission_Create(ORMBase): # 글 추천 
    role_id : str
    menu_id : int
    @field_validator('role_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v         

    @field_validator('menu_id')
    def not_empty_positive_int(cls, v):
        if v is None:
            raise ValueError('값은 필수입니다.')
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError('정수만 입력 가능합니다.')
        if v <= 0:
            raise ValueError('0보다 큰 값만 허용됩니다.')
        return v          

class System_Permission_Update(ORMBase): 
    role_id     : str
    menu_id     : int
    use_yn      : str = Field(default="Y", pattern="^(Y|N)$"  , description="사용유무 Y N") 
 
    @field_validator('role_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v     
    
    @field_validator('menu_id')
    def not_empty_positive_int(cls, v):
        if v is None:
            raise ValueError('값은 필수입니다.')
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise ValueError('정수만 입력 가능합니다.')
        if v <= 0:
            raise ValueError('0보다 큰 값만 허용됩니다.')
        return v      
      
    @field_validator('use_yn')
    def validate_use_yn(cls, v):
        if v not in ("Y", "N"):
            raise ValueError("use_yn must be 'Y' or 'N'")
        return v  

class System_Permission_Delete(ORMBase):
    role_id     : str
    menu_id     : int

      
#----------------- system_permission : end
