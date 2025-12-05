from fastapi import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , status , Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
import urllib.parse

from datetime import datetime , timedelta

from database       import engine , getDb

from Models         import Sys as models_sys
from Dantics        import Sys as dantics_sys
from Cruds          import Sys as cruds_sys
from utils          import utils

router = APIRouter()
templates = Jinja2Templates(directory="Templates")
util  = utils()

@router.get(path='/' , description='로그인화면'  )
async def loging():
    template = templates.get_template("Sys/login.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)    

@router.get(path='/mypage' , description='마이페이지 화면'  )
async def mypage():
    template = templates.get_template("Sys/mypage.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)    

@router.get(path='/password' , description='비밀번호 변경 화면'  )
async def password():
    template = templates.get_template("Sys/password.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)    

@router.put(path='/password' , description='비밀번호 변경 프로세스'  )
async def password( input : dantics_sys.User_password_chg , db : Session=Depends(getDb)  ):
    return cruds_sys.setUserPassword(input, db)


@router.post(path='/user', description='가입')
async def post_user( user : dantics_sys.User_Create , db : Session=Depends(getDb)  ):
    tmp = cruds_sys.getUser(user , db)
    if tmp:
        raise HTTPException( status_code=status.HTTP_409_CONFLICT , detail='user_id 중복'  )
    
    # 회원 가입하기
    cruds_sys.setUser( user , db )    
    return HTTPException( status_code=status.HTTP_200_OK , detail='welcome join us' )

@router.post(path='/login' , description='로그인 프로세서')
async def post_login( response : Response , request : Request ,  input : OAuth2PasswordRequestForm = Depends( dantics_sys.User_Login) , db : Session=Depends(getDb) ):
    ''''''
    # 회원 존재 여부 확인
    user = cruds_sys.getUser( input , db)
    if not user:
        raise HTTPException( status_code=status.HTTP_400_BAD_REQUEST , detail='Invalid user or password'  )

    # 로그인
    res = util.verifyHash( input.password , user.password ) 

    # 토큰 생성 => 향후 화면 권한 까지 쿠키에 넣어서 처리한다.
    access_token_expires = timedelta( util.getTokenExpireMinutes() )
    access_token = util.create_access_token( data={'user_id' :  user.user_id , 'user_nm':user.user_nm} , expires_delta= access_token_expires )

    # 쿠키에 저장
    response.set_cookie(key='access_token'  , value=access_token , expires=access_token_expires , httponly=True)    
    response.set_cookie(key='user_nm'       , value= urllib.parse.quote(user.user_nm)   )
    response.set_cookie(key='user_id'       , value= urllib.parse.quote(user.user_id)   )
    response.set_cookie(key='email'         , value= urllib.parse.quote(user.email)     )
    response.set_cookie(key='tel_phone'     , value= urllib.parse.quote(user.tel_phone) )
    response.set_cookie(key='cell_phone'    , value= urllib.parse.quote(user.cell_phone))

    if not res:
        raise HTTPException( status_code=status.HTTP_400_BAD_REQUEST , detail='Invalid user or password'  )
    #return dantics_sys.Token(access_token=access_token , token_type='bearer')
    return HTTPException(status_code=status.HTTP_200_OK , detail='sucessed'  )

@router.put(path='/' , description='개인정보 변경 프로세스'  )
async def userinfochg(res:Response , input : dantics_sys.User_Update , db : Session=Depends(getDb)  ):
    res.set_cookie(key='user_nm'       , value= urllib.parse.quote(input.user_nm)   )
    res.set_cookie(key='user_id'       , value= urllib.parse.quote(input.user_id)   )
    res.set_cookie(key='email'         , value= urllib.parse.quote(input.email)     )
    res.set_cookie(key='tel_phone'     , value= urllib.parse.quote(input.tel_phone) )
    res.set_cookie(key='cell_phone'    , value= urllib.parse.quote(input.cell_phone))
    return cruds_sys.putUser(input, db)
    

@router.delete(path='/logout' , description='로그아웃')
async def get_logout( response : Response , request : Request ):
    ''''''
    access_token = request._cookies.get('access_token')
    str = util.token_decoe( access_token ) # 쿠키값 decode 한 결과 위 생성한 쿠키의 access_token 에서 생성된 내용이 저장되어 있다.
    
    # 쿠키 삭제
    for cookie in request.cookies:
        response.delete_cookie(key=cookie)
    return HTTPException(status_code=status.HTTP_200_OK , detail='Logout successful')

# menu - start
@router.get(path='/menu' , description='메뉴 화면'  )
async def mypage():
    template = templates.get_template("Sys/menu.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)    

@router.get(path='/menus' , description='전체 조회'  )
async def menuSelect( db : Session = Depends(getDb) ):
    return cruds_sys.MeneSelects(db)


@router.post(path='/menu', description='메뉴 생성')
async def menuCreate( input : dantics_sys.Menu_Create , db : Session=Depends(getDb)  ):
    '메뉴 생성'
    return cruds_sys.MenuCreate( input , db )

@router.put(path='/menu', description='메뉴 수정')
async def menuUpdate( input : dantics_sys.Menu_Update , db : Session=Depends(getDb)  ):
    '메뉴 수정'
    return cruds_sys.MenuUpdate( input , db )

@router.delete(path='/menu/{seq}', description='메뉴 삭제')
async def menuDelete( seq : int , db : Session=Depends(getDb)  ):
    '메뉴 삭제'
    return cruds_sys.MenuDelete( seq , db )
# menu - end

# role - start
#@router.get(path='/role' , description='권한 화면'  )
#async def role():
#    template = templates.get_template("Sys/role.html")    
#    rendered = template.render()  # context 전달
#    return HTMLResponse(content=rendered)    

@router.get(path='/role' , description='조회'  )
async def roleSelect( db : Session = Depends(getDb) ):
    return cruds_sys.RoleSelects(db)

@router.post(path='/role', description='권한 생성')
async def roleCreate( input : dantics_sys.RoleCreate , db : Session=Depends(getDb)  ):
    '권한메뉴 생성'
    return cruds_sys.RoleCreate( input , db )

@router.put(path='/role', description='권한 수정')
async def roleUpdate( input : dantics_sys.RoleUpdate , db : Session=Depends(getDb)  ):
    '권한 수정'
    return cruds_sys.RoleUpdate( input , db )

@router.delete(path='/role/', description='권한 삭제')
async def menuDelete( input : dantics_sys.RoleDelete , db : Session=Depends(getDb)  ):
    '메뉴 삭제'
    return cruds_sys.RoleDelete( input , db)

# role - end

# rolemenu - start
@router.get(path='/rolemenu/{role_seq}' , description='조회'  )
async def rolemenuSelect( role_seq : int , db : Session = Depends(getDb) ):
    return cruds_sys.RoleMenuSelect(role_seq , db)

@router.post(path='/rolemenu' , description='생성'  )
async def rolemenuCreate( input : dantics_sys.RoleMenuCreate , db : Session = Depends(getDb) ):
    return cruds_sys.RoleMenuCreate(input , db)

@router.put(path='/rolemenu', description='수정')
async def rolemenuUpdate( input : dantics_sys.RoleMenuUpdate , db : Session=Depends(getDb)  ):
    '수정'
    return cruds_sys.RoleMenuUpdate( input , db )

@router.delete(path='/rolemenu/', description='삭제')
async def rolemenuDelete( input : dantics_sys.RoleMenuDelete , db : Session=Depends(getDb)  ):
    '메뉴 삭제'
    return cruds_sys.RoleMenuDelete( input , db)
# rolemenu - end