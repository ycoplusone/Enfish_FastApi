from fastapi            import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , Response
from sqlalchemy.orm     import Session,joinedload
from typing             import List
from fastapi.responses  import FileResponse , HTMLResponse , RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette          import status
from fastapi.security   import OAuth2PasswordRequestForm , OAuth2PasswordBearer
from datetime import datetime, timedelta

# 사용자 라이브러리
from database           import engine , getDb
from utils              import utils
from ..Models           import ModelSystem      as sm
from .                  import system_dantic    as sd
from .                  import system_crud      as sc

router  = APIRouter()
util    = utils()

#----------------- system_users : begin
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/system/users/login")

def get_current_user(token: str = Depends(oauth2_scheme) , db: Session = Depends(getDb)):    
    # 접속자 id 가져오기 위한 함수.       
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )    
    try:
        payload = util.token_decoe( token=token )
        email : str = payload.get("email")
        if email is None:
            raise credentials_exception
    except Exception as e:
        raise credentials_exception
    else:
        user = sc.get_user(db=db, email=email)
        if user is None:
            raise credentials_exception
        return user 

@router.post("/users/create", status_code=status.HTTP_200_OK , description='user 1건 생성')
def user_create( _user : sd.System_Users_Create , db: Session = Depends(getDb)):    
    user = sc.get_existing_user(db=db , _user = _user) # 사용자 검색
    if user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT , detail="이미 존재하는 사용자입니다.")
    user = sc.system_users_create(db=db , _user = _user ) # 사용자 생성


@router.post('/users/login', response_model= sd.Token ,description='로그인' )
def login_for_access_token( form_data : OAuth2PasswordRequestForm = Depends() , db: Session = Depends(getDb)  ):    
    email = form_data.username
    user = sc.get_user( db=db , email=email )    
    if not user:
        raise HTTPException( status_code=status.HTTP_400_BAD_REQUEST , detail='Check Your Infomation' )
    
    res = util.verifyHash( form_data.password , user.password )  # 입력 패스워드와 등록 패스워드 확인
    if not res:
        raise HTTPException( status_code=status.HTTP_400_BAD_REQUEST , detail='Check Your Infomation' )
    
    print( {'user_nm' :  user.user_nm , 'user_id' : user.user_id , 'id': user.id , 'email':user.user_email} )
    
    access_token = util.create_access_token( data={'user_nm' :  user.user_nm , 'user_id' : user.user_id , 'id': user.id , 'email':user.user_email} , expires_delta=0 )     
    
    __result = sc.user = sc.system_permission_user(db=db , user_id=user.id)
    perms = __result.role_ref.permissions if  __result and __result.role_ref else [] 

    return {
        "access_token"  : access_token,
        "token_type"    : "bearer" ,
        "id"            : user.id , 
        "user_id"       : user.user_id ,
        "user_nm"       : user.user_nm,
        "user_email"    : user.user_email,
        "permissions"   : perms,
    }

@router.put("/users/update/role", status_code=status.HTTP_200_OK , description='system_users role update')
def system_users_update_role( _data : sd.System_Users_Update_role , current_user: sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    print('*'*50)
    print('_data', _data)
    _db = sc.get_existing_user(db=db , _user=_data)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_users_update_role(db=db , db_data=_db , _update=_data)

@router.get("/users/list" ,  description="system_users select all")
def system_users_list( db : Session = Depends(getDb) ):
    __result = sc.system_users_list(db=db)
    return {'datas' : __result}
#----------------- system_users : end

#----------------- system_boards : begin
@router.get("/boards/get/{kind}" ,  response_model=sd.System_Boards_List ,description="system_boards 전체 리스트")
def boards_get( db : Session = Depends(getDb) , page: int = 0, size: int = 10 , kind:str='', keyword: str = ''):    
    __total, __list = sc.get_boards_list( db=db , skip=page*size , limit=size , kind=kind, keyword=keyword)
    return { 'total': __total , 'boards_list': __list }  

@router.get('/boards/detail/{id}' , response_model=sd.System_Boards , description='boards one row select')
def boards_get_one( id : int , db : Session = Depends(getDb) ):    
    _data = sc.get_boards_row(db=db, id = id  )
    return _data

@router.post("/boards/set", status_code=status.HTTP_200_OK , description='boards one row insert')
def boards_set( _data : sd.System_Boards_Create , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb)  ):
    sc.set_boards(db=db , board_create= _data , user=current_user)

@router.put("/boards/update", status_code=status.HTTP_200_OK , description='boards one row update')
def boards_update( _data : sd.System_Boards_Update , current_user: sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb)):
    db_board = sc.get_boards_row(db=db , id = _data.system_boards_id )
    if not db_board:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_board.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="수정 권한이 없습니다.")
    sc.update_boards(db=db, db_boards=db_board , data = _data)

@router.delete("/boards/delete", status_code=status.HTTP_200_OK , description='boards one row delete')
def boards_delete( _data : sd.System_Boards_Delete , db: Session = Depends(getDb) , current_user: sd.System_Users = Depends(get_current_user) ):
    db_board = sc.get_boards_row(db=db , id = _data.system_boards_id )
    if not db_board:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_board.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="삭제 권한이 없습니다.")
    sc.delete_boards(db=db , data = db_board )

@router.post("/boards/vote", status_code=status.HTTP_200_OK , description='board vote')
def question_vote( _vote: sd.System_Boards_Vote , db: Session = Depends(getDb) , current_user: sd.System_Users = Depends(get_current_user) ):
    db_board = sc.get_boards_row(db=db , id=_vote.system_boards_id)
    if not db_board:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")    
    sc.vote_boards(db=db, db_board=db_board , db_user=current_user)
#----------------- system_boards : end

#----------------- system_comments : begin
@router.get("/comments/detail/{comment_id}", response_model=sd.System_Comments , description='comments one row select')
def comments_get_one(comment_id: int, db: Session = Depends(getDb)): 
    answer = sc.get_comment(db=db, comment_id=comment_id)
    return answer

@router.post("/comments/set/{board_id}",  description='comments one row insert')
def comments_set( board_id : int , _comment_create: sd.System_Comments_Create , current_user : sd.System_Users = Depends(get_current_user), db: Session = Depends(getDb) ):
    board = sc.get_boards_row(db=db , id=board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    _data = sc.set_comment(db=db , board=board , comment= _comment_create , user=current_user)
    return _data

@router.put("/comments/update", status_code=status.HTTP_200_OK , description='comment update')
def comments_update(_comment_update: sd.System_Comments_Update , db: Session = Depends(getDb) , current_user : sd.System_Users = Depends(get_current_user) ):
    _db = sc.get_comment(db=db , comment_id=_comment_update.comment_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != _db.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="수정 권한이 없습니다.")
    sc.update_comment(db=db, db_comment= _db , comment_update=_comment_update )

@router.delete("/comments/delete", status_code=status.HTTP_200_OK , description='comment delete')
def comments_delete( comment_delete : sd.System_Comments_Delete , db: Session = Depends(getDb) , current_user : sd.System_Users = Depends(get_current_user) ):
    _db = sc.get_comment(db=db, comment_id=comment_delete.comment_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != _db.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="삭제 권한이 없습니다.")
    sc.delete_comment(db=db , db_comment=_db)


@router.post("/comments/vote", status_code=status.HTTP_200_OK , description='질문 추천')
def answer_vote( _vote: sd.System_Comments_Vote , db: Session = Depends(getDb) , current_user: sd.System_Users = Depends(get_current_user) ):
    db_comment = sc.get_comment(db=db , comment_id=_vote.comment_id)
    if not db_comment:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.vote_comment(db=db , db_comment=db_comment , db_user= current_user)
#----------------- system_comments : end

#----------------- system_menus : start
@router.get("/menus/list" ,  response_model=sd.System_Menus_List ,description="system_menus select all")
def system_menus_list( db : Session = Depends(getDb) ):
    __result = sc.system_menus_list(db=db)
    return {'datas' : __result}

@router.post("/menus/create", status_code=status.HTTP_200_OK ,  description='system_menus_create')
def system_menus_create( _input : sd.System_Menus_Create , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb)  ):
    sc.system_menus_create(db=db , _create_data = _input , user=current_user)
    
@router.put("/menus/update", status_code=status.HTTP_200_OK , description='system_menus_update')
def system_menus_update( _update : sd.System_Menus_Update ,  current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_menus_get(db=db , menu_id= _update.id )
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_menus_update(db=db , menus_db=_db , update_data=_update , user=current_user )

@router.delete("/menus/delete", status_code=status.HTTP_200_OK , description='comment delete')
def system_menus_delete( _delete : sd.System_Menus_Delete , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_menus_get(db=db, menu_id=_delete.id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_menus_delete(db=db , db_data=_db)
#----------------- system_menus : end

#----------------- system_roles : start
@router.get("/roles/list" ,  response_model=sd.System_Roles_List ,description="system_roles select all")
def system_roles_list( db : Session = Depends(getDb) ):
    __result = sc.system_roles_list(db=db)
    return {'datas' : __result}

@router.post("/roles/create", status_code=status.HTTP_200_OK ,  description='system_roles create')
def system_roles_create( _input : sd.System_Roles_Create , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb)  ):
    sc.system_roles_create(db=db , _create_data = _input , user=current_user)
    
@router.put("/roles/update", status_code=status.HTTP_200_OK , description='system_roles update')
def system_roles_update( _update : sd.System_Roles_Update ,  current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_roles_get(db=db , id=_update.role_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_roles_update(db=db , input_db=_db , update_data=_update , user=current_user )

@router.delete("/roles/delete", status_code=status.HTTP_200_OK , description='system_roles delete')
def system_roles_delete( _delete : sd.System_Roles_Delete , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_roles_get(db=db , id=_delete.role_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_roles_delete(db=db , db_data=_db)
#----------------- system_roles : end

#----------------- system_permission : start
@router.get("/permission/list" ,  response_model=sd.System_Permission_List ,description="system permission select all")
def system_permission_list( db : Session = Depends(getDb) ):
    __result = sc.system_permission_list(db=db)
    return {'datas' : __result}

@router.get("/permission/get/{role_id}" ,  response_model=sd.System_Permission_List ,description="system permission select get")
def system_permission_list( role_id:str , db : Session = Depends(getDb)   ):
    __result = sc.system_permission_get_list(db=db,role_id=role_id)
    return {'datas' : __result}


@router.post("/permission/create", status_code=status.HTTP_200_OK ,  description='system permission create')
def system_permission_create( _input : sd.System_Permission_Create , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb)  ):
    sc.system_permission_create(db=db , _create_data = _input , user=current_user)

    
@router.put("/permission/update", status_code=status.HTTP_200_OK , description='system permission update')
def system_permission_update( _update : sd.System_Permission_Update ,  current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_permission_get(db=db , role_id=_update.role_id , menu_id=_update.menu_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_permission_update(db=db , input_db=_db , update_data=_update , user=current_user )

@router.delete("/permission/delete", status_code=status.HTTP_200_OK , description='system permission delete')
def system_permission_delete( _delete : sd.System_Permission_Delete , current_user : sd.System_Users = Depends(get_current_user) , db: Session = Depends(getDb) ):
    _db = sc.system_permission_get(db=db , role_id=_delete.role_id , menu_id=_delete.menu_id)
    if not _db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    sc.system_permission_delete(db=db , db_data=_db)
#----------------- system_permission : end

#----------------- system_test : start
@router.get("/system/{user_pk}/permissions", response_model=sd.UserPermissionsOut)
def get_user_permissions(user_pk: int, db: Session = Depends(getDb)):
    '''
    user = (
        db.query(sm.System_Users)
        .options(
            joinedload(sm.System_Users.role_ref)
            .joinedload(sm.System_Roles.permissions)
            .joinedload(sm.System_Permission.menu),
        )
        .filter(sm.System_Users.id == user_pk)
        .one_or_none()
    )
    '''
    user = sc.system_permission_user(db=db , user_id=user_pk)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not user.role_ref:
        return {"user_id": user.id, "role_id": user.role_id, "permissions": []}

    # role 기반 permissions
    perms = user.role_ref.permissions

    # 원하면 use_yn 필터 적용 가능 (DB에서 필터하려면 별도 쿼리로)
    # perms = [p for p in perms if p.use_yn == "Y" and p.menu and p.menu.use_yn == "Y"]

    return {
        "user_id": user.id,
        "role_id": user.role_ref.role_id,
        "permissions": perms,
    }
#----------------- system_test : end