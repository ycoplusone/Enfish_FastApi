from fastapi    import HTTPException , status
from database   import Session , update
from utils      import utils
from datetime   import datetime
from sqlalchemy import and_

from .          import test_dantic  as d_test
from ..Models   import ModelTest as m_test

util = utils()

'''
def get_question_list(db: Session , skip: int = 0, limit: int = 10 , keyword : str = ''):
    #question_list = db.query(m_test.Question).order_by(m_test.Question.create_date.desc()).all()
    #return question_list
    _question_list = db.query(m_test.Question).order_by(m_test.Question.id.desc())

    total = _question_list.count()
    question_list = _question_list.offset(skip).limit(limit).all()
    return total, question_list  # (전체 건수, 페이징 적용된 질문 목록)    
'''
def get_question_list(db: Session, skip: int = 0, limit: int = 10, keyword: str = ''):
    question_list = db.query(m_test.Question)
    if keyword:
        search = '%%{}%%'.format(keyword)
        sub_query = db.query(
                        m_test.Answer.question_id, 
                        m_test.Answer.content, 
                        m_test.User.user_nm
                    ).outerjoin(
                        m_test.User, 
                        and_(m_test.Answer.user_id == m_test.User.id)
                    ).subquery()
        question_list = question_list.outerjoin(
                            m_test.User
                        ).outerjoin(
                            sub_query, 
                            and_(sub_query.c.question_id == m_test.Question.id)
                        ).filter(
                            m_test.Question.subject.ilike(search) |        # 질문제목
                            m_test.Question.content.ilike(search) |        # 질문내용
                            m_test.User.user_nm.ilike(search) |           # 질문작성자
                            sub_query.c.content.ilike(search) |     # 답변내용
                            sub_query.c.user_nm.ilike(search)      # 답변작성자
                        )
    total = question_list.distinct().count()
    question_list = question_list.order_by(
                        m_test.Question.create_date.desc()
                    ).offset(skip).limit(limit).distinct().all()
    return total, question_list  # (전체 건수, 페이징 적용된 질문 목록)
    

def get_question(db: Session , question_id: int):
    _question = db.query(m_test.Question).get(question_id)
    return _question

def create_question(db: Session, question_create: d_test.QuestionCreate , user : m_test.User):
    db_question = m_test.Question(
                        subject=question_create.subject,
                        content=question_create.content,
                        create_date=datetime.now() ,
                        user = user
                        )
    db.add(db_question)
    db.commit()

def update_question(db: Session, db_question: m_test.Question , question_update: d_test.QuestionUpdate):
    db_question.subject     = question_update.subject
    db_question.content     = question_update.content
    db_question.modify_date = datetime.now()
    db.add(db_question)
    db.commit()

def delete_question(db: Session, db_question: m_test.Question):
    db.delete(db_question)
    db.commit()    

def vote_question(db: Session, db_question: m_test.Question, db_user: m_test.User): 
    # 질문 추천
    db_question.voter.append(db_user)
    db.commit()



def create_answer(db: Session, question: m_test.Question, answer_create: d_test.AnswerCreate , user : m_test.User):
    db_answer = m_test.Answer(
                        question    = question,
                        content     = answer_create.content,
                        create_date = datetime.now(),
                        user        = user
                        )
    db.add(db_answer)
    db.commit()

def get_answer(db: Session, answer_id: int):
    return db.query(m_test.Answer).get(answer_id)

def update_answer(db: Session, db_answer: m_test.Answer , answer_update: d_test.AnswerUpdate):
    db_answer.content       = answer_update.content
    db_answer.modify_date   = datetime.now()
    db.add(db_answer)
    db.commit()

def delete_answer(db: Session, db_answer: d_test.Answer):
    db.delete(db_answer)
    db.commit()

def vote_answer(db: Session, db_answer: m_test.Answer, db_user: m_test.User):
    # 답변 추천
    db_answer.voter.append(db_user)
    db.commit()    



def create_user(db: Session, user_create: d_test.UserCreate):
    db_user = m_test.User(
                    user_nm = user_create.user_nm, 
                    password = util.getHash(user_create.password) ,
                    email=user_create.email)
    
    db.add(db_user)
    db.commit()

def get_existing_user(db: Session, user_create: d_test.UserCreate):
    return db.query(m_test.User).filter( m_test.User.email == user_create.email ).first()

def get_user(db: Session, email: str):
    return db.query(m_test.User).filter(m_test.User.email == email).first()

