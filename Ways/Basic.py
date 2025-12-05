from fastapi import APIRouter , Query , Path  , Body , Depends , HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import List

router = APIRouter()

@router.get('/' )
def first():
    ''' 첫번째 '''
    return RedirectResponse(url='/sys/')

@router.get('/version')
def GetVersion():
    '''버전을...'''
    return {'version' : '0.1.0235'}



@router.get('/users/{user_id}')
def read_user(user_id:str):
    '''
    Pydantic의 path() 함수를 이용해서 더 정교하게 검증할수 있다. 정규화시킬수 있다.
    '''
    return {'user_id':user_id}

@router.get('/items')
def read_user(skip:int=0 , limit:int=10):
    ''' get  방식으로 skip 값과 limit 값을 받아 처리 할수 있다.'''
    return {'skip':skip , 'limit':limit}

@router.get('/search')
def SearchItem( q : str=Query(min_length=3 , max_length=10) ):
    '''
    Query() 를 이용해서 정규검증을 처리할수 있다.
    1. 경로에 포함되지 않은 인자는 자동으로 쿼리 파라미터로 처리
    2. 기본값을 지정해 선택적 파라미터 구현
    3. query를 통해 유효성 검사
    '''
    return {'query':q}

'''------------------- todos 시작'''
todos : List[str] = []
@router.post('/todos' , status_code=201) # status_code 할시 성공시 반환하는 응답코드를 지정한다.
def PostTodos(item:str):
    todos.append(item)
    return {'msg':'todo 생성' , 'item':item}

@router.get('/todos')
def GetTodos():
    return {'todos':todos}

@router.delete('/todos/{index}')
def DeleteTodos(index:int):
    if 0 < index < len(todos):
        removed = todos.pop(index)
        return {'msg':'todo deleted','item':removed}
    else :
        #return {'msg':'indoex out of range','index':index}
        raise HTTPException( status_code=404 , detail= 'todo index of out range' )
    
'''------------------- todos 종료'''