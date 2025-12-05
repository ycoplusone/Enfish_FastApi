from fastapi import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , status , Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import FileResponse , HTMLResponse , RedirectResponse
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

@router.get("/index" , description='메인 화면')
async def index( res : Response , req : Request ):
    access_token = req._cookies.get('access_token')
    if access_token == None: # 로그인 토큰 값이 없으면 로그인 창으로 이동한다.
        return RedirectResponse(url='/sys')
    
    template = templates.get_template("Dashboard/index.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)      

@router.get("/path/{uid}" , description='dash board 화면들')
async def path( uid: str , res : Response , req : Request ):
    template = templates.get_template(f"Dashboard/{uid}.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)     