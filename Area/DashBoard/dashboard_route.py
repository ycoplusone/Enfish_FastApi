# Ways.Test

from fastapi            import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , Response
from sqlalchemy.orm     import Session
from typing             import List
from fastapi.responses  import FileResponse , HTMLResponse , RedirectResponse

from starlette          import status
from fastapi.security   import OAuth2PasswordRequestForm , OAuth2PasswordBearer
from database           import engine , getDb
from utils              import utils
from datetime import datetime, timedelta



#from ..Models           import ModelTest as m_test
#from . import test_dantic   as d_test
#from . import test_crud     as c_test



router  = APIRouter()
util    = utils()




@router.get("/test1" ,  description="dashboard test1")
def test1():
    return {
            "summary": {
                "total_users": 1250,
                "active_sessions": 42
            },
            "recent_activities": [
                {"time": "2023-10-27 10:00", "message": "새로운 사용자가 가입했습니다."},
                {"time": "2023-10-27 09:45", "message": "시스템 업데이트가 완료되었습니다."}
            ]
        }

