from fastapi    import HTTPException , status
from database   import Session
from utils      import utils
from datetime   import datetime
from sqlalchemy import and_
from sqlalchemy.orm import Session, selectinload , joinedload
from typing import List, Optional, Sequence, Iterable, Dict, Tuple

from .          import system_dantic as sd
from ..Models   import ModelSystem as sm

util = utils()


#----------------- system_users : begin
def system_users_create(db: Session, _user: sd.System_Users_Create): # user insert
    _db = sm.System_Users(
                    user_email  = _user.user_email , 
                    user_id     = _user.user_id,
                    user_nm     = _user.user_nm , 
                    password    = util.getHash( _user.password ) ,
                    tel_phone   = _user.tel_phone ,
                    cell_phone  = _user.cell_phone ,
                )    
    db.add( _db)
    db.commit()

def get_existing_user(db: Session , _user : sd.System_Users_Create): # user check    
    return db.query(sm.System_Users).filter( sm.System_Users.user_email == _user.user_email ).first()

def get_user(db: Session, email: str): # user select 1
    return db.query(sm.System_Users).filter(sm.System_Users.user_email == email).first()

def system_users_update_role(db:Session , db_data: sm.System_Users , _update : sd.System_Users_Update_role): # system_users role update
    db_data.role_id     = _update.role_id
    db_data.use_yn      = _update.use_yn
    db.add( db_data )
    db.commit()   

def system_users_list(db: Session ): # system_users select all
    _result = db.query(sm.System_Users).all()
    return _result    
#----------------- system_users : end

#----------------- system_boards : begin
def get_boards_list(db: Session , skip: int = 0, limit: int = 10 , keyword : str = '', kind:str=''): # board select all
    _list = db.query(sm.System_Boards).filter(sm.System_Boards.kind == kind )
    if keyword:
        search = '%%{}%%'.format(keyword)
        sub_query = db.query(
                        sm.System_Comments.board_id, 
                        sm.System_Comments.content, 
                        sm.System_Users.user_nm
                    ).outerjoin(
                        sm.System_Users, 
                        and_(sm.System_Comments.user_id == sm.System_Users.id)
                    ).subquery()
        _list = _list.outerjoin(
                            sm.System_Users
                        ).outerjoin(
                            sub_query, 
                            and_(sub_query.c.board_id == sm.System_Boards.id)
                        ).filter(
                            sm.System_Boards.subject.ilike(search) |        # 질문제목
                            sm.System_Boards.content.ilike(search) |        # 질문내용
                            sm.System_Users.user_nm.ilike(search) |           # 질문작성자
                            sub_query.c.content.ilike(search) |     # 답변내용
                            sub_query.c.user_nm.ilike(search)      # 답변작성자
                        )
    total = _list.distinct().count()
    _list = _list.order_by( sm.System_Boards.id.desc() ).offset(skip).limit(limit).distinct().all()
    return total, _list  # (전체 건수, 페이징 적용된 질문 목록)    

def get_boards_row(db: Session , id: int): # board select 1
    _data = db.query(sm.System_Boards).get(id)
    return _data

def set_boards(db: Session , board_create : sd.System_Boards_Create , user : sm.System_Users ): # board insert
    __db = sm.System_Boards(
                        kind    = board_create.kind ,
                        subject = board_create.subject ,
                        content = board_create.content ,
                        create_date = datetime.now() ,
                        user = user
                        )
    db.add( __db )
    db.commit()

def update_boards(db: Session, db_boards : sm.System_Boards , data: sd.System_Boards_Update): # board update
    db_boards.subject       = data.subject
    db_boards.content       = data.content
    db_boards.modify_date   = datetime.now()
    db.add(db_boards)
    db.commit()    

def delete_boards(db: Session , data : sm.System_Boards): # board delete
    db.delete( data )
    db.commit()  

def vote_boards(db: Session, db_board : sm.System_Boards , db_user:sm.System_Users ):  # board vote
    # 질문 추천
    #db_board.voter.append(db_user)
    db_board.voter.append(db_user)
    db.commit()
#----------------- system_boards : end

#----------------- system_comments : begin
def get_comment(db: Session, comment_id : int): # comment select
    return db.query(sm.System_Comments).get(comment_id)

def set_comment(db: Session, board: sm.System_Boards , comment : sd.System_Comments_Create , user : sm.System_Users): # comment insert
    _db = sm.System_Comments(
                    board_id    = board.id , 
                    content     = comment.content ,
                    create_date = datetime.now() ,
                    user        = user
                    )
    db.add( _db )
    db.commit()

def update_comment(db: Session, db_comment: sm.System_Comments , comment_update: sd.System_Comments_Update): # comment update
    db_comment.content       = comment_update.content
    db_comment.modify_date   = datetime.now()
    db.add( db_comment )
    db.commit()

def delete_comment(db: Session, db_comment: sd.System_Comments): # comment delete
    db.delete( db_comment )
    db.commit()


def vote_comment(db: Session, db_comment: sm.System_Comments , db_user: sm.System_Users):
    # 답변 추천
    db_comment.voter.append(db_user)
    db.commit()   
#----------------- system_comments : end

#----------------- system_menus : start
def system_menus_list(db: Session ): # system_menus select all
    _result = db.query(sm.System_Menus).order_by(sm.System_Menus.sort_no).all()
    return _result

def system_menus_get(db: Session, menu_id : int): # comment select
    return db.query(sm.System_Menus).get(menu_id)

def system_menus_create(db:Session , _create_data : sd.System_Menus_Create , user : sm.System_Users ): # system_menus create
    _db = sm.System_Menus(
                    menu_nm         = _create_data.menu_nm ,
                    group_nm        = _create_data.group_nm , 
                    url_path        = _create_data.url_path  ,
                    sort_no         = _create_data.sort_no , 
                    reged_user_id   = user.id ,
                    )
    db.add( _db )
    db.commit()    

def system_menus_update(db: Session, menus_db : sm.System_Menus , update_data: sd.System_Menus_Update , user : sm.System_Users ): 
    menus_db.menu_nm        = update_data.menu_nm 
    menus_db.group_nm       = update_data.group_nm
    menus_db.url_path       = update_data.url_path
    menus_db.use_yn         = update_data.use_yn
    menus_db.sort_no        = update_data.sort_no
    menus_db.reged_user_id  = user.id    
    db.add( menus_db )
    db.commit()

def system_menus_delete(db: Session, db_data: sd.System_Menus_Delete): # comment delete
    db.delete( db_data )
    db.commit() 
#----------------- system_menus : end

#----------------- system_roles : start
def system_roles_list(db: Session ): # system_roles select all
    _result = db.query(sm.System_Roles).all()
    return _result

def system_roles_get(db: Session, id : str): # system_roles select
    return db.query(sm.System_Roles).get(id)

def system_roles_create(db:Session , _create_data : sd.System_Roles_Create , user : sm.System_Users ): # sysystem_rolesstem_menus create
    _db = sm.System_Roles(
                    role_id         = _create_data.role_id ,
                    role_nm         = _create_data.role_nm , 
                    reged_user_id   = user.id ,
                    )
    db.add( _db )
    db.commit()    

def system_roles_update(db: Session, input_db : sm.System_Roles , update_data: sd.System_Roles_Update , user : sm.System_Users ): 
    input_db.role_nm        = update_data.role_nm
    input_db.use_yn         = update_data.use_yn
    input_db.reged_user_id  = user.id    
    db.add( input_db )
    db.commit()

def system_roles_delete(db: Session, db_data: sd.System_Roles_Delete): # comment delete
    db.delete( db_data )
    db.commit() 
#----------------- system_roles : end

#----------------- system_permission : start
def system_permission_list(db: Session ):
    _result = db.query(sm.System_Permission).all()
    return _result

def system_permission_get_list(db:Session , role_id : str):
    _result = db.query(sm.System_Permission).filter(sm.System_Permission.role_id == role_id).all()
    return _result

def system_permission_get(db: Session, role_id : str , menu_id : int):
    return db.query(sm.System_Permission).filter(
        sm.System_Permission.role_id == role_id , 
        sm.System_Permission.menu_id == menu_id ).one()

def system_permission_create(db:Session , _create_data : sd.System_Permission_Create , user : sm.System_Users ):
    _db = sm.System_Permission(
                    role_id         = _create_data.role_id ,
                    menu_id         = _create_data.menu_id , 
                    reged_user_id   = user.id ,
                    )
    db.add( _db )
    db.commit()    

def system_permission_update(db: Session, input_db : sm.System_Permission , update_data: sd.System_Permission_Update , user : sm.System_Users ): 
    input_db.role_id        = update_data.role_id
    input_db.menu_id        = update_data.menu_id
    input_db.use_yn         = update_data.use_yn
    input_db.reged_user_id  = user.id    
    db.add( input_db )
    db.commit()

def system_permission_delete(db: Session, db_data: sd.System_Permission_Delete): # comment delete
    db.delete( db_data )
    db.commit() 

def system_permission_user(db:Session , user_id : int ):
    data = (
        db.query(sm.System_Users)
        .options(
            joinedload(sm.System_Users.role_ref)
            .joinedload(sm.System_Roles.permissions)
            .joinedload(sm.System_Permission.menu),
        )
        .filter(sm.System_Users.id == user_id)
        .one_or_none()
    )
    return data    

#----------------- system_permission : end
