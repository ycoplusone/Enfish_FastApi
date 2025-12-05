from fastapi import FastAPI , Query , HTTPException , Depends ,Request
from fastapi.responses import HTMLResponse , JSONResponse , RedirectResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates



from Area.test import test_route as ways_test
#from Models import Board as models_board , Sys as models_sys
#from Ways import Basic as ways_basic, Sys as ways_sys, Board as ways_board , Dashboard as ways_dashboard
from Area.Models import ModelTest
from database import engine


app = FastAPI(
    swagger_ui_parameters={
        'defaultModelsExpandDepth'  : -1    ,  # 왼쪽 Models 섹션 숨김 [0 , -1]
        'docExpansion'              : 'none',  # 전체 collapse [none , list]
        'displayRequestDuration'    : True  ,  # 요청 시간 표시
        'filter'                    : True  ,  # 필터 기능 활성화
    }
)

app.add_middleware(
    CORSMiddleware ,
    allow_origins=["http://localhost:5173","http://localhost:4173",'*'],  # 개발 중이면 일단 *
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)



@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def block_devtools_file(): # 크롬 dev 호출 부분 우회 시킨다.
    raise HTTPException(status_code=404, detail="Not available")


app.mount("/statics", StaticFiles(directory="statics")              , name="statics")



#models_board.Base.metadata.create_all(bind=engine)  # 만약 이미 테이블이 생성되어 있다면 건너 뛴다.
ModelTest.Base.metadata.create_all(bind=engine)    # 만약 이미 테이블이 생성되어 있다면 건너 뛴다.

app.include_router(ways_test.router     , prefix="/test"    , tags=["test router"] , include_in_schema=True) # include_in_schema=True는 스웨거에 포함시키는것

#app.include_router(ways_basic.router    , tags=["basic router"] , include_in_schema=True , ) # include_in_schema=True는 스웨거에 포함시키는것
#app.include_router(ways_sys.router     , prefix="/sys"    , tags=["System Part"] , include_in_schema=True) # include_in_schema=True는 스웨거에 포함시키는것
#app.include_router(ways_board.router    , prefix="/board"   , tags=["board router"] , include_in_schema=True) # include_in_schema=True는 스웨거에 포함시키는것
#app.include_router(ways_dashboard.router     , prefix="/dashboard"    , tags=["분석 화면 Router"] , include_in_schema=True) # include_in_schema=True는 스웨거에 포함시키는것

