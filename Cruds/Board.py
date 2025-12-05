from database   import Session , update
from Models     import Board as models_board
from Dantics    import Board as dantics_board

from datetime import datetime

def post_item( new_board : dantics_board.CreateBoard , db : Session ):
    '''board 생성'''
    board = models_board.board(
        content         = new_board.content 
        , user_id       = new_board.user_id
    )
    db.add(board)
    db.commit()
    item = db.query(models_board.board).filter(
            models_board.board.seq == board.seq
            , models_board.board.use_yn == 'Y'            
            ).all()
    return item

def get_items( db : Session ):
    '''board 전체 조회'''
    items = db.query(models_board.board).filter(models_board.board.use_yn=='Y').all()    
    return items

def get_item( user_id : int ,  db : Session ):
    '''board 1건 조회'''
    item = db.query(models_board.board).filter(models_board.board.seq == user_id ).one_or_none()
    return item

def put_item( board : dantics_board.UpdateBoard , db : Session):
    '''board 1건 수정'''

    row = db.query(models_board.board).filter(
                        models_board.board.seq == board.seq 
                        , models_board.board.use_yn == 'Y').one_or_none()
    if row is None:
        return None
    row.content = board.content
    row.user_id = board.user_id  
    row.create_dt = datetime.now()

    db.commit()
    db.refresh(row)    
    print(row.__dict__)
    return row

def patch_item( board : dantics_board.BoardPatch , db : Session):
    '''board 1건 수정'''

    row = db.query(models_board.board).filter( models_board.board.seq == board.seq ).one_or_none()
    if row is None:
        return None
    row.use_yn = board.use_yn  
    row.create_dt = datetime.now()

    db.commit()
    db.refresh(row)    
    return row


def delete_item( board : dantics_board.BoardSeq , db : Session):
    '''board 1건 삭제'''

    row = db.query(models_board.board).filter(
                        models_board.board.seq == board.seq 
                        , models_board.board.use_yn == 'Y').one_or_none()
    if row is None:
        return {'msg':'자료가 없습니다.'}
    
    db.delete(row)
    db.commit()
    return {'msg':'삭제 되었습니다.'}


    



