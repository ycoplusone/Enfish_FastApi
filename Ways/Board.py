from fastapi        import APIRouter , Query , Path  , Body , Depends , HTTPException
from sqlalchemy.orm import Session
from typing         import List
from fastapi.templating import Jinja2Templates

from database   import getDb
from Dantics    import Board as dantics_board
from Cruds      import Board as cruds_board

router = APIRouter()
templates = Jinja2Templates(directory="Templates")
'''
접두사  => url 의 규칙의 단건 item 다중 items 로만 구분하면 crud는 오직 method 로 구분한다.
get     조회        => 전체 items , 단건 item/{id}
post    생성        => 생선 1건씩 처리 한다. item
put     수정(전체)  => 수정 1건씩 처리한다. item
patch   수정(부분)  => 수정 1건씩 처리한다. item
delete  삭제        => 삭제 1건씩 처리한다. item
'''


@router.post(path='/item' , description='기본 게시판 글 생성' , response_model=list[dantics_board.Board]  )
async def post_item( new_board : dantics_board.CreateBoard , db : Session = Depends(getDb) ):    
    return cruds_board.post_item( new_board , db )

@router.get(path='/items' , description='전체 조회' , response_model=list[dantics_board.BoardList] )
async def get_items( db : Session = Depends(getDb) ):
    return cruds_board.get_items( db )

@router.get(path='/item/{user_id}' , description='단일 조회' , response_model=dantics_board.Board)
async def get_item( user_id : int , db : Session = Depends(getDb) ):
    return cruds_board.get_item( user_id , db)

@router.put(path='/item' , description='기본 게시판 글 수정' , response_model=dantics_board.Board)
async def put_item(  board : dantics_board.UpdateBoard , db : Session = Depends(getDb)  ):
    return cruds_board.put_item( board , db ) 

@router.patch(path='/item' , description='기본 게시판 글 수정' , response_model=dantics_board.Board)
async def put_item(  board : dantics_board.BoardPatch , db : Session = Depends(getDb)  ):
    return cruds_board.patch_item( board , db ) 

@router.delete(path='/item' , description='기본 게시판 글 수정' )
async def delete_item(  board : dantics_board.BoardSeq , db : Session = Depends(getDb)  ):
    return cruds_board.delete_item( board , db ) 