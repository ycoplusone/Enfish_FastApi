from fastapi import HTTPException , status

from database   import Session , update
from Models     import Sys as models_sys
from Dantics    import Sys as dantics_sys
from utils      import utils
from datetime import datetime

util = utils()

def getUser( input : dantics_sys.User_Create , db : Session ):
    '''user 1건 조회'''
    data = db.query(models_sys.User).filter(
                                            models_sys.User.user_id == input.user_id
                                            , models_sys.User.use_yn == 'Y'
                                              ).one_or_none()
    return data

def setUser( input : dantics_sys.User_Create  , db : Session ):
    '''user 1건 생성'''
    user = models_sys.User(
        user_id         = input.user_id
        , user_nm       = input.user_nm
        , email         = input.email
        , password      = util.getHash(input.password) # 암호화
        , tel_phone     = input.tel_phone
        , cell_phone    = input.cell_phone
    )
    db.add(user)
    db.commit()

def setUserPassword(input : dantics_sys.User_password_chg , db : Session):
    ''' user 비밀번호 변경'''
    row = db.query( models_sys.User).filter(
            models_sys.User.user_id == input.user_id
            , models_sys.User.use_yn == 'Y'
        ).one_or_none()
    
    if row is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='확인되지 않는 ID 입니다.'  )
    
    res = util.verifyHash( input.password_cur , row.password )     
    if not(res):
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='현재 비밀번호가 맞지 않습니다.'  )

    if input.password_new1 != input.password_new2:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='확인 비밀번호가 맞지 않습니다.'  )

    row.password = util.getHash(input.password_new1)
    row.updated_dt = datetime.now() 

    db.commit()
    db.refresh(row)
    return HTTPException( status_code=status.HTTP_200_OK , detail='변경되었습니다'  )
    

def putUser( input : dantics_sys.User_Update  , db : Session ):
    '''user 1건 수정'''
    row = db.query( models_sys.User).filter(
            models_sys.User.user_id == input.user_id
            , models_sys.User.use_yn == 'Y'
        ).one_or_none()
    row.email = input.email
    row.user_nm = input.user_nm
    row.tel_phone = input.tel_phone
    row.cell_phone = input.cell_phone
    row.updated_dt = datetime.now()    
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='변경되었습니다'  )

# menu - start
def MeneSelects( db : Session ):
    '''Menu 전체 조회'''
    datas = db.query(models_sys.Menu).filter().all()    
    return HTTPException( status_code=status.HTTP_200_OK , detail= datas )

def MenuCreate( input : dantics_sys.Menu_Create  , db : Session ):
    '''Menu 1건 생성'''
    temp = db.query(models_sys.Menu).filter( models_sys.Menu.menu_id == input.menu_id ).one_or_none()
    if temp is not None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
        
    data = models_sys.Menu(
        menu_id         = input.menu_id
        , menu_nm       = input.menu_nm
        , parent_nm     = input.parent_nm
        , url_path      = input.url_path
        , use_yn        = input.use_yn
        , created_dt    = datetime.now()
        , updated_dt    = datetime.now()        
    )
    db.add(data)
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='등록되었습니다.'  )

def MenuUpdate( input : dantics_sys.Menu_Update  , db : Session ):
    '''Menu 1건 수정'''
    data = db.query( models_sys.Menu ).filter( models_sys.Menu.seq == input.seq ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
    data.menu_id        = input.menu_id
    data.menu_nm        = input.menu_nm
    data.parent_nm      = input.parent_nm
    data.url_path       = input.url_path
    data.use_yn         = input.use_yn
    data.updated_dt     = datetime.now()
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='변경되었습니다'  )

def MenuDelete( seq :int , db : Session):
    '''menu 1건 삭제'''
    data = db.query(models_sys.Menu).filter( models_sys.Menu.seq == seq ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
    
    db.delete(data)
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='완료되었습니다.'  )
# menu - end

# role - start
def RoleSelects( db : Session ):
    '''Menu 전체 조회'''
    datas = db.query(models_sys.Role).filter().all()    
    return HTTPException( status_code=status.HTTP_200_OK , detail= datas )

def RoleCreate( input : dantics_sys.RoleCreate  , db : Session ):
    '''role 1건 생성'''    
    temp = db.query(models_sys.Role).filter( models_sys.Role.role_nm == input.role_nm ).one_or_none()
    if temp is not None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
        
    data = models_sys.Role(
        role_nm = input.role_nm
    )
    db.add(data)
    db.commit()        
    return HTTPException( status_code=status.HTTP_200_OK , detail='등록되었습니다.'  )

def RoleUpdate( input : dantics_sys.RoleUpdate  , db : Session ):
    '''Role 1건 수정'''
    data = db.query( models_sys.Role ).filter( models_sys.Role.seq == input.seq ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
    data.role_nm        = input.role_nm
    data.use_yn         = input.use_yn
    data.updated_dt     = datetime.now()
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='변경되었습니다'  )

def RoleDelete( input : dantics_sys.RoleDelete , db : Session):
    '''Role 1건 삭제'''
    data = db.query(models_sys.Role).filter( models_sys.Role.seq == input.seq ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
    
    db.delete(data)
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='완료되었습니다.'  )
# role - end

# rolemenu - start 
def RoleMenuSelect( roel_seq : int , db : Session ):
    '''roleMenu 조회'''
    #datas = db.query(models_sys.RoleMenu , models_sys.Menu ).outerjoin( models_sys.RoleMenu.menu_seq == models_sys.Menu.seq ).all()
    results = db.query( models_sys.RoleMenu.role_seq
                    , models_sys.RoleMenu.menu_seq
                    , models_sys.RoleMenu.use_yn
                    , models_sys.Menu.menu_nm
                    , models_sys.Menu.parent_nm
                    , models_sys.Menu.url_path
                    , models_sys.Menu.use_yn.label('use_yn_menu')
                    ).outerjoin(models_sys.RoleMenu.rolemenu_menu).filter(models_sys.RoleMenu.role_seq == roel_seq ).all()
    #datas = [{"role_seq": r[0], "menu_seq": r[1] , "use_yn": r[2] , "menu_nm": r[3] , "parent_nm": r[4] , "menurl_pathu_seq": r[5] , "menuse_yn_menuu_seq": r[6]  } for r in results]
    datas = [dict(row._mapping) for row in results]
    return HTTPException( status_code=status.HTTP_200_OK , detail= datas )



def RoleMenuCreate( input : dantics_sys.RoleMenuCreate  , db : Session ):
    #role 1건 생성
    temp = db.query(models_sys.RoleMenu).filter(  models_sys.RoleMenu.role_seq == input.role_seq
                                                , models_sys.RoleMenu.menu_seq == input.menu_seq
                                                ).one_or_none()
    if temp is not None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='중복 발생 확인 바랍니다.'  )
        
    data = models_sys.RoleMenu(
          role_seq = input.role_seq
        , menu_seq = input.menu_seq
        , use_yn   = input.use_yn
    )
    db.add(data)
    db.commit()        
    return HTTPException( status_code=status.HTTP_200_OK , detail='등록되었습니다.'  )


def RoleMenuUpdate( input : dantics_sys.RoleMenuUpdate  , db : Session ):
    # Role 1건 수정
    data = db.query( models_sys.RoleMenu ).filter( models_sys.RoleMenu.role_seq == input.role_seq
                                                  , models_sys.RoleMenu.menu_seq == input.menu_seq
                                                   ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )    
    data.use_yn         = input.use_yn
    data.updated_dt     = datetime.now()
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='변경되었습니다'  )

def RoleMenuDelete( input : dantics_sys.RoleMenuDelete , db : Session):
    '''Role 1건 삭제'''
    data = db.query(models_sys.RoleMenu).filter( models_sys.RoleMenu.role_seq == input.role_seq
                                                , models_sys.RoleMenu.menu_seq == input.menu_seq
                                                 ).one_or_none()
    if data is None:
        return HTTPException( status_code=status.HTTP_409_CONFLICT , detail='이상 발생 확인 바랍니다.'  )
    
    db.delete(data)
    db.commit()
    return HTTPException( status_code=status.HTTP_200_OK , detail='완료되었습니다.'  )


# rolemenu - end