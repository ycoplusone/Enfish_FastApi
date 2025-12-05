# Ways.Test

from fastapi            import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , Response
from sqlalchemy.orm     import Session
from typing             import List
from fastapi.responses  import FileResponse , HTMLResponse , RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette          import status
from fastapi.security   import OAuth2PasswordRequestForm , OAuth2PasswordBearer
from database           import engine , getDb
from utils              import utils

from datetime import datetime, timedelta

from ..Models           import ModelTest as m_test
from . import test_dantic   as d_test
from . import test_crud     as c_test


#templates = Jinja2Templates(directory="Templates")
router  = APIRouter()
util    = utils()

# FastSvelte 시작 지점 ------------------------------------------------------------------------------------------- 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/test/user/login")
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
        user = c_test.get_user(db=db, email=email)
        if user is None:
            raise credentials_exception
        return user    


@router.get("/question/list" ,  response_model=d_test.QuestionList ,description="Question 전체 리스트")
def question_list( db : Session = Depends(getDb) , page: int = 0, size: int = 10 , keyword: str = ''):    
    #_question_list = c_test.get_question_list(db=db)
    #return _question_list
    total, _question_list = c_test.get_question_list(db, skip=page*size, limit=size , keyword=keyword)
    return { 'total': total , 'question_list': _question_list }    

@router.get('/question/detail/{question_id}' , response_model=d_test.Question, description='question의 개별 상세 내역을 조회한다.')
def detail_id( question_id : int , db : Session = Depends(getDb) ):    
    _data = c_test.get_question(db=db, question_id = question_id  )
    return _data




@router.post("/question/create", status_code=status.HTTP_200_OK , description='질문 1개 생성')
def question_create(_question_create: d_test.QuestionCreate , current_user : m_test.User = Depends(get_current_user) , db: Session = Depends(getDb)):
    c_test.create_question(db=db, question_create=_question_create , user=current_user)

@router.put("/question/update", status_code=status.HTTP_200_OK , description='줄문 1개 수정')
def question_update(_question_update: d_test.QuestionUpdate , db: Session = Depends(getDb), current_user: m_test.User = Depends(get_current_user)):
    db_question = c_test.get_question(db=db , question_id=_question_update.question_id)
    if not db_question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_question.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="수정 권한이 없습니다.")
    c_test.update_question(db=db, db_question=db_question , question_update=_question_update)


@router.delete("/question/delete", status_code=status.HTTP_200_OK , description='질문 1개 삭제')
def question_delete(_question_delete: d_test.QuestionDelete , db: Session = Depends(getDb) , current_user: m_test.User = Depends(get_current_user)):
    db_question = c_test.get_question(db, question_id=_question_delete.question_id)
    if not db_question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_question.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="삭제 권한이 없습니다.")
    c_test.delete_question(db=db, db_question=db_question)

@router.post("/question/vote", status_code=status.HTTP_200_OK , description='질문 추천')
def question_vote( _question_vote: d_test.QuestionVote , db: Session = Depends(getDb) , current_user: m_test.User = Depends(get_current_user) ):
    db_question = c_test.get_question(db=db , question_id=_question_vote.question_id)
    if not db_question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    c_test.vote_question(db=db, db_question=db_question, db_user=current_user)
    


 

@router.post("/answer/create/{question_id}",  description='질문의 답변 1개 생성')
def answer_create(question_id: int , _answer_create: d_test.AnswerCreate , current_user : m_test.User = Depends(get_current_user), db: Session = Depends(getDb) ):
    question = c_test.get_question(db, question_id=question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    _data = c_test.create_answer(db=db , question=question , answer_create=_answer_create , user=current_user)
    return _data

@router.get("/answer/detail/{answer_id}", response_model=d_test.Answer)
def answer_detail(answer_id: int, db: Session = Depends(getDb)):
    answer = c_test.get_answer(db=db, answer_id=answer_id)
    return answer

@router.put("/answer/update", status_code=status.HTTP_200_OK , description='답변 1개 수정')
def answer_update(_answer_update: d_test.AnswerUpdate , db: Session = Depends(getDb) , current_user: m_test.User = Depends(get_current_user)):
    db_answer = c_test.get_answer(db=db, answer_id=_answer_update.answer_id)
    if not db_answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_answer.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="수정 권한이 없습니다.")
    c_test.update_answer(db=db, db_answer=db_answer , answer_update=_answer_update)

@router.delete("/answer/delete", status_code=status.HTTP_200_OK , description='답변 1개 삭제')
def answer_delete(_answer_delete : d_test.AnswerDelete , db: Session = Depends(getDb) , current_user: m_test.User = Depends(get_current_user)):
    db_answer = c_test.get_answer(db=db, answer_id=_answer_delete.answer_id)
    if not db_answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    if current_user.id != db_answer.user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="삭제 권한이 없습니다.")
    c_test.delete_answer(db=db, db_answer=db_answer)

@router.post("/answer/vote", status_code=status.HTTP_200_OK , description='질문 추천')
def answer_vote(_answer_vote: d_test.AnswerVote , db: Session = Depends(getDb) , current_user: m_test.User = Depends(get_current_user)):
    db_answer = c_test.get_answer(db, answer_id=_answer_vote.answer_id)
    if not db_answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST , detail="데이터를 찾을수 없습니다.")
    c_test.vote_answer(db, db_answer=db_answer, db_user=current_user)



@router.post("/user/create", status_code=status.HTTP_200_OK , description='user 1건 생성')
def user_create( _input : d_test.UserCreate, db: Session = Depends(getDb)):    
    user = c_test.get_existing_user(db=db , user_create= _input)
    if user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT , detail="이미 존재하는 사용자입니다.")
    user = c_test.create_user(db=db, user_create=_input)   

@router.post('/user/login', response_model= d_test.Token ,description='로그인' )
def login_for_access_token( form_data : OAuth2PasswordRequestForm = Depends() , db: Session = Depends(getDb)  ):    
    email = form_data.username
    user = c_test.get_user( db=db , email=email )    
    if not user:
        raise HTTPException( status_code=status.HTTP_400_BAD_REQUEST , detail='Check Your Infomation' )
    
    res = util.verifyHash( form_data.password , user.password )  # 입력 패스워드와 등록 패스워드 확인
    access_token = util.create_access_token( data={'user_nm' :  user.user_nm , 'user_id' : user.id , 'email':user.email} , expires_delta=0 )     
    return {
        "access_token"  : access_token,
        "token_type"    : "bearer",
        "user_id"       : user.id ,
        "user_nm"       : user.user_nm,
        "email"         : user.email
    }

@router.post("/user/token", status_code=status.HTTP_200_OK , description='토큰 데이터 검사')
def user_create( token : str):    
    _data = util.token_decoe(token)
    return{
        "현재시간" : datetime.now(),
        "만료시간" : datetime.fromtimestamp(_data['exp'] ) , 
        "기타" : _data
    }   
    
    
    
    













#------------------- dantic 시작
# basemodel을 상속한 요청 모델을 선언하자
# 요청본문으로 json데이터가 들어오면 FastApi가 자동으로  pydantic모델로 변환한다.
# 모델에 정의된 조건을 만족하지 않으면 함수가 실행되지 않고 오류를 반환한다.
# Field()를 이용해 제약조건과 문서 예시를 축할수 있다.
# response_model를 지정하면 return 형식이 response_model 기준으로 출력된다.
#from Dantics import Test
todos : List[str] = []
'''
@router.post('/dantic/todos/' , status_code=201 , response_model=d_test.TodoItem)
def create_todo(item : d_test.TodoItemCreate):    
    todo = {'id':len(todos)+1 , 'title':item.title , 'description':item.description}
    todos.append(todo)
    return todo
'''
#------------------- dantic 종료

#------------ 엔드포인트 구현 - 시작
# pydantic의 basemodel을 이용해 데이터 구조를 선언하고 , 타입검증과 기본값 설정까지 깔끔하게 처리한다.
# 요청 바디에 pydantic  모델을 활용해 json을 자동 파싱하고 오류를 422 상태 코드로 처리한다.
# RESPONSE_MODEL 를 통해 응답 데이터를 명확히 통제하고 민감 정보를 자동 제거 할수 있다.
# SWAGGER 문서화 에 입력/출력 모델이 자동 반영되는 과정을 통해 개발 생산성이 높아진다.


#todoz:List[d_test.TodoItem] = []
#current_id = 0

'''
@router.post('/todoz' , response_model=d_test.TodoItem , status_code=201 , description='할일 생성')
def create_todo(item : d_test.TodoItemCreate):
    global current_id
    current_id += 1
    print('*'*50)
    print(item)
    print( item.dict() )
    print('*'*50)

    todo = d_test.TodoItem(id=current_id , **item.dict() )

    todoz.append( todo )
    return todo

@router.get('/todoz' , response_model=List[d_test.TodoItem] , description='aaaa')
def list_todoz():    
    return todoz

@router.get('/todoz/{id}' , response_model=d_test.TodoItem , description='단일 할일 조회')
def get_todo(id:int):
    for todo in todoz:
        if todo.id == id:
            return todo
    raise HTTPException(status_code=404 , detail='todo not found')

@router.put('/todoz/{id}' , response_model=d_test.TodoItem , description='완료 여부 수정정')
def update_todo(id:int , done:bool):
    print(id , done)
    for todo in todoz:
        if todo.id == id:
            todo.done = done
            return todo
    
    raise HTTPException(status_code=404 , detail='todo not found')

@router.delete('/todoz/{id}',status_code=201,response_model=d_test.TodoItem , description='삭제')
def delete_todo(id:int):
    for idx,todo in enumerate(todoz):
        if todo.id == id:
            print(todo)
            todoz.pop(idx)
            return todo
    
    raise HTTPException(status_code=404 , detail='todo not found')

#------------ 엔드포인트 구현 - 종료

@router.post('/todos2' , response_model=d_test.TodoItem)
def create_todo(item : d_test.TodoItemCreate , db : Session = Depends(getDb) ):
    db_todo = m_test.Todo( title=item.title , description=item.description )
    db.add( db_todo )
    db.commit()
    db.refresh( db_todo )
    return db_todo


@router.get("/ttt2")
async def ttt2():
    template = templates.get_template("Test/index.html")
    context = {'title':'제목되니?'}
    render = template.render(context)
    return HTMLResponse(content=render )

@router.get("/ttt/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    template = templates.get_template("Test/test.html")
    context = {'title':'오호호' , 'id':id}
    rendered = template.render(context)  # context 전달
    return HTMLResponse(content=rendered)
    #context = {'request':request , 'id':id}
    #return templates.TemplateResponse(name="Test/test.html", context=context)

@router.get("/tabs", response_class=HTMLResponse)
async def tabs():
    template = templates.get_template("Test/tabs.html")
    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)

@router.get("/tabs2", response_class=HTMLResponse)
async def tabss():
    template = templates.get_template("Test/tabs2.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)

@router.get("/material", response_class=HTMLResponse)
async def material():
    template = templates.get_template("Test/material.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)    

@router.get("/test2", response_class=HTMLResponse)
async def material( res : Response , req : Request ):
    access_token = req._cookies.get('access_token')
    if access_token == None: # 로그인 토큰 값이 없으면 로그인 창으로 이동한다.
        return RedirectResponse(url='/user')
    
    template = templates.get_template("Test/test2.html")    
    rendered = template.render()  # context 전달
    return HTMLResponse(content=rendered)      
'''
# --------------------------------------------------------------------------------------------------------------- 
