from fastapi            import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , Response
from typing             import List
from starlette          import status
from datetime           import datetime, timedelta
from sqlalchemy.orm     import Session

# 확장 라리브러리
import openai
import base64
import numpy as np
import json
import fitz
import requests
from tqdm import tqdm
#import ollama


# 사용자 라이브러리
from database           import engine , getDb
from utils              import utils
from ..system           import system_route 
#from ..Models           import ModelSystem      as sm
#from .                  import system_dantic    as sd
from .                  import Rag_crud      as rc

router  = APIRouter()
util    = utils()
'''
※ RAG 
    - "LLM Master : 기초부터 심화까지 RAG 쿡북 with Python" 을 기반으로 테스트 함.
        => https://github.com/no-wave/llm-master-rag-techniques/tree/main
    - "Do it!LLM을 활용한 AI에이전트 개발 입문:GPT API+딥시크+라마+랭체인+랭그래프+RAG"
        => 

'''



'''
@router.post("/test" , status_code=status.HTTP_200_OK , description='system_route 에서 로그인 확인 부분 가져와서 테스트')
def system_users_update_role( current_user: List = Depends(system_route.get_current_user) , db: Session = Depends(getDb) ):
    return current_user
'''
'''
@router.post("/test3" , tags=['AI / Rag'],  description='OpenAi 이미지 생성')
def test3( txt : str ):
    _model='gpt-4o-mini'
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.Image.create(
        prompt = txt , 
        n=1, # 이미지 한장
        size="512x512"
    )
    #print(response)
    image = response["data"][0]["url"]
    return {
        "image_url": image
    }    
'''
'''    
@router.post("/test3" , tags=['AI / Rag'] ,  description='OpenAi Vector Embedding')
def test4( txt : str ):
    _model='text-embedding-3-large'
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(
        model = _model ,
        input = '음식은 맛있었고 가격은 저렴했다.',        
    )
    print(response)
'''

@router.post("/test2" , tags=['AI / Rag'],  description='OpenAi 라이브러리 테스트')
def test2( txt : str ):
    _model='gpt-4o-mini'
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.chat.completions.create(
        model       = _model , 
        messages    = [
            {'role':'system' , 'content':'너는 유용하게 사용하는 assistant야.'},
            {'role':'user' , 'content': txt },
        ]
    )
    #print(response)
    #print(response["choices"])
    return {
        'result': response.choices[0].message.content
    }

#----------------- simpleRAG : start
@router.get("/simple_test" , tags=['AI / Rag'] ,  description='심플 RAG 테스트')
def simple_test( ):
    rc.ch1_RunningaQueryOnExtractedChunks()

@router.get("/simple/{id}" , tags=['AI / Rag'],  description='심플 RAG')
def simple( id:int=0 ):
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'    
    
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 첫번째 항목에서 질의 추출
    query = data[id]['question']

    # 텍스트 다시 추출
    text = rc.ch1_extract_text_from_pdf( pdf_path )

    # chunks 생성
    chunks = rc.ch1_chunk_text(text , 1000,200)

    # 임베딩 생성
    embedding = rc.ch1_create_embeddings(chunks)

    # 의미 기반 검색 수행 : 주어진 쿼리에 대해 가장 관련성 높은 텍스트 청크 2개
    top_chunks = rc.ch1_semantic_search(query , chunks , embedding.data , 2 )

    #검색된 상위 문맥(top_chunks)을 기반으로 사용자 프롬프트 구성
    user_prompt = '\n'.join([
        f'Context {i+1}:\n{chunk}\n---------\n'
        for i ,chunk in enumerate(top_chunks)
    ])
    user_prompt = f"{user_prompt}\n Question : {query}"

    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 문맥을 기반으로만ㄴ 답변하는 AI어시턴트 입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )

    # AI응답 생성 
    ai_res = rc.ch1_generate_response(system_prompt , user_prompt)

    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 AI 어시스턴트의 응답을 평가하는 지능형 평가 시스템입니다. "
        "AI 응답이 정답에 매우 근접하면 점수 1점을 부여하고, "
        "정답과 맞지 않거나 불만족스러우면 0점을 부여하세요. "
        "부분적으로 일치하면 0.5점을 부여하세요."
    )

    # 사용자 질의, AI 응답, 정답, 평가 프롬프트를 결합하여 평가용 프롬프트 생성
    evaluation_prompt = (
        f"User Query: {query}\n"
        f"AI Response:\n{ai_res.choices[0].message.content}\n"
        f"True Response: {data[id]['ideal_answer']}\n"
        f"{evaluate_system_prompt}"
    )    

    # 평가 시스템 프롬프트를 사용하여 AI 응답 평가 점수 생성
    evaluation_response = rc.ch1_generate_response(evaluate_system_prompt, evaluation_prompt)

  
    return {
            'score':evaluation_response.choices[0].message.content , 
            'True Response' : data[id]['ideal_answer'] ,
            'answer': query ,
            'question':ai_res.choices[0].message.content ,
            }
#----------------- simpleRAG : end

#----------------- SemanticChunking : start
@router.get("/semanticchunking_test" , tags=['AI / Rag'],  description='세멘틱 청킹 테스트')
def semanticchunking_test( ):
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    
    # 텍스트 다시 추출
    extracted_text = rc.extract_text_from_pdf2( pdf_path )    
    
    # 텍스트를 문장 단위로 분할(기본적인 마침표 기준 분리)
    sentences = extracted_text.split('. ')
    
    # 각문장에 대해 임베딩 벡터 생성
    embeddings = [ rc.get_embedding(sentence) for sentence in sentences ]
    #print( f"총 {len(embeddings)} 개의 문장 임베딩이 생성되었습니다." )

    # 인접한 문장 쌍간 유사도 계산
    similarities = [
        rc.cosine_similarity( embeddings[i] , embeddings[i+1] )
        for i in range(len(embeddings)-1)
    ]

    # 퍼센트일 방법을 사용하여 임계값 90으로 분할점을 계산.
    breakpoints = rc.compute_breakpoints( similarities=similarities , method='percentile' , threshold=90 )    

    # split_into_chunks 함수를 사용하여 청크를 생성.
    text_chunks = rc.split_into_chunks( sentences=sentences , breakpoints=breakpoints )
    # 생성된 청크의 개수 출력
    #print( f'의미청크 개수:{len(text_chunks)}' )
    #print( f'\n 첫 번째 텍스트 청크: \n {text_chunks[0]}' )
    
    # create_embeddings2 함수를 사용하여 청크 임베딩을 생성.
    chunk_embeddings = rc.create_embeddings2(text_chunks=text_chunks)
    
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 검증 데이터에서 첫 번째 쿼리를 추출.
    query = data[0]['question']    

    # 관련성 높은 상위 2개의 텍스트 추출
    top_chunks = rc.semantic_search2( query=query , text_chunks=text_chunks , chunk_embeddings=chunk_embeddings , k=2 )
    #쿼리 출력.
    print(f'쿼리 : {query}')
    #상위 2개 관련성 높은 텍스트 출력
    for i , chunk in enumerate(top_chunks):
        print(f"컨텍스트 {i+1}:\n{chunk}\n{'-'*40}")
    
@router.get("/semanticchunking/{id}" , tags=['AI / Rag'],  description='세멘틱 청킹')
def semanticchunking( id:int=0 ):
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    
    # 텍스트 다시 추출
    extracted_text = rc.extract_text_from_pdf2( pdf_path )    
    
    # 텍스트를 문장 단위로 분할(기본적인 마침표 기준 분리)
    sentences = extracted_text.split('. ')
    
    # 각문장에 대해 임베딩 벡터 생성
    embeddings = [ rc.get_embedding(sentence) for sentence in sentences ]
    #print( f"총 {len(embeddings)} 개의 문장 임베딩이 생성되었습니다." )

    # 인접한 문장 쌍간 유사도 계산
    similarities = [
        rc.cosine_similarity( embeddings[i] , embeddings[i+1] )
        for i in range(len(embeddings)-1)
    ]

    # 퍼센트일 방법을 사용하여 임계값 90으로 분할점을 계산.
    breakpoints = rc.compute_breakpoints( similarities=similarities , method='percentile' , threshold=90 )    

    # split_into_chunks 함수를 사용하여 청크를 생성.
    text_chunks = rc.split_into_chunks( sentences=sentences , breakpoints=breakpoints )
    # 생성된 청크의 개수 출력
    #print( f'의미청크 개수:{len(text_chunks)}' )
    #print( f'\n 첫 번째 텍스트 청크: \n {text_chunks[0]}' )
    
    # create_embeddings2 함수를 사용하여 청크 임베딩을 생성.
    chunk_embeddings = rc.create_embeddings2(text_chunks=text_chunks)
    
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 검증 데이터에서 첫 번째 쿼리를 추출.
    query = data[id]['question']    

    # 관련성 높은 상위 2개의 텍스트 추출
    top_chunks = rc.semantic_search2( query=query , text_chunks=text_chunks , chunk_embeddings=chunk_embeddings , k=2 )

    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 문맥을 기반으로만ㄴ 답변하는 AI어시턴트 입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )

    #검색된 상위 문맥(top_chunks)을 기반으로 사용자 프롬프트 구성
    user_prompt = '\n'.join([
        f'Context {i+1}:\n{chunk}\n---------\n'
        for i ,chunk in enumerate(top_chunks)
    ])
    user_prompt = f"{user_prompt}\n Question : {query}"    

    # AI응답 생성 
    ai_res = rc.generate_response(system_prompt , user_prompt)

    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 AI 어시스턴트의 응답을 평가하는 지능형 평가 시스템입니다. "
        "AI 응답이 정답에 매우 근접하면 점수 1점을 부여하고, "
        "정답과 맞지 않거나 불만족스러우면 0점을 부여하세요. "
        "부분적으로 일치하면 0.5점을 부여하세요."
    )

    # 사용자 질의, AI 응답, 정답, 평가 프롬프트를 결합하여 평가용 프롬프트 생성
    evaluation_prompt = (
        f"User Query: {query}\n"
        f"AI Response:\n{ai_res.choices[0].message.content}\n"
        f"True Response: {data[id]['ideal_answer']}\n"
        f"{evaluate_system_prompt}"
    )    

    # 평가 시스템 프롬프트를 사용하여 AI 응답 평가 점수 생성
    evaluation_response = rc.generate_response(evaluate_system_prompt, evaluation_prompt)    
    
    return {
            'score':evaluation_response.choices[0].message.content , 
            'True Response' : data[id]['ideal_answer'] ,
            'answer': query ,
            'question':ai_res.choices[0].message.content ,
            }    
#----------------- SemanticChunking : end    

#----------------- Chunk Sizes Rag : start
@router.get("/chunksize/{id}" , tags=['AI / Rag'],  description='Chunk size 테스트')
def chunksize(id:int = 0 ):
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    
    # 텍스트 다시 추출
    extracted_text = rc.ch3_extract_text_from_pdf( pdf_path )    

    # 평가할 다양한 청크 크기를 정의합니다.
    chunk_sizes = [128,256,512]

    # 각 청크 크기에 대해 텍스트 청크를 생성하여 dict에 저장.
    text_chunks_dict = { size : rc.chunk_text(extracted_text , size , size // 5) for size in chunk_sizes }

    # 각 청크 크기에 대해 생성된 청크 개수를 출력
    for size , chunks in text_chunks_dict.items():
        print(f"청크 크기 : {size} , 생성된 청크수 : {len(chunks)}")
    
    # 각 청크 크기별 텍스트에 대한 임베딩 생성
    # tqdm을 사용하여 진행상태를 시작화 한다.
    chunk_embeddings_dict = {
        size : rc.ch3_create_embeddings( chunks )
        for size , chunks in tqdm(text_chunks_dict.items() , desc='임베딩생성중')
    }

    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 검증 데이터에서 4 번째 쿼리를 추출.
    query = data[id]['question']

    # 각 청크 크기에 대해 관련성 높은 텍스트 청크를 검색.
    retrieved_chunks_dict = {
        size : rc.ch3_retrieve_relevant_chunks(query , text_chunks=text_chunks_dict[size] , chunk_embeddings=chunk_embeddings_dict[size])
        for size in chunk_sizes
    }

    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 문맥을 기반으로만ㄴ 답변하는 AI어시턴트 입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )    

    # 각 청크 크기별로 AI응답을 생성.
    ai_responses_dict = {
        size : rc.ch3_generate_response( query , system_prompt , retrieved_chunks_dict[size] )
        for size in chunk_sizes
    }

    # 기준 정답을 불러 옵니다.
    ture_answer = data[id]['ideal_answer']

    # 청크 크기 256 , 128에 대해 응답 평가수행.
    faithfulness1 , relevancy1 = rc.ch3_evaluate_response(query , ai_responses_dict[256] , ture_answer)
    faithfulness2 , relevancy2 = rc.ch3_evaluate_response(query , ai_responses_dict[128] , ture_answer)
    return {
            'answer': query ,
            'True Response' : ture_answer ,
            '256 답변' : ai_responses_dict[256] , 
            '256 신뢰성' : faithfulness1 ,
            '256 관련성' : relevancy1 ,
            '128 답변' : ai_responses_dict[128] , 
            '128 신뢰성' : faithfulness2 ,
            '128 관련성' : relevancy2 ,            
        }   
#----------------- Chunk Sizes Rag : end

#----------------- Context-Enriched Rag : start
@router.get("/contextenriched/{id}" , tags=['AI / Rag'],  description='문맥 강화 RAG')
def contextenriched( id:int = 0 ):
    ''' 문맥 강화 Context Enriched RAG'''
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    
    # 텍스트 다시 추출
    extracted_text = rc.ch4_extract_text_from_pdf( pdf_path )    

    # 추출된 텍스트를 1000자 단위, 200자 중첩으로 청크 분할.
    text_chunks = rc.ch4_chunk_text( extracted_text , 1000 , 200 )
    # 생성된 텍스트 청크 개수 확인
    #print( f'청크 개수 : {len(text_chunks)}' )
    #print('-'*200)
    #print( f"첫 번째 청크 \n{text_chunks[0]}" )
    #print('-'*200)
    #print( f"첫 번째 청크 \n{text_chunks[1]}" )

    # 텍스트 청크에 대한 임베딩 생성.
    response = rc.ch4_create_embeddings(text_chunks)

    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 검증 데이터에서 id 번째 쿼리를 추출.
    query = data[id]['question']

    # 문맥 확장 검색을 통해 관련 청크와 그 주변 청크를 검색
    # 매개변수 설명.
    # query : 검색할 질문
    # text_chunks : Pdf 에서 분할한 텍스트 청크
    # response.data : 텍스트 청크들의 임베딩
    # k=1 가장 유사한 청크 하나만 선택
    # context_size = 1 : 해당 청크 앞뒤로 1개씩 문맥 포함.
    top_chunks = rc.ch4_context_enriched_search(query , text_chunks , response.data , k=1,context_size=1)
    '''
    # 쿼리 출력
    print(f'쿼리 :{query}')
    # 검색된 각 청크를 번호와 구분석을 포함하여 출력
    for i, chunk in enumerate(top_chunks):
        print(f"컨텍스트 {i+1}:\n{chunk}\n{'-'*30}")
    '''
    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 컨텍스트에 기반하여 엄격하게 대답하는 AI어시턴트입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )  

    # 검색된 상위 청크들을 기반으로 사용자 프롬프트를 생성.
    user_prompt = "\n".join([
        f"Context {i+1}:\n{chunk}\n-----------------------n"
        for i , chunk in enumerate(top_chunks)
    ])
    user_prompt = f"{user_prompt}\n Question:{query}"

    # AI 응답을 생성.
    ai_response = rc.ch4_generate_response( system_prompt=system_prompt , user_message=user_prompt )

    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 AI 어시스턴트의 응답을 평가하는 지능형 평가 시스템입니다. "
        "AI 응답이 정답에 매우 근접하면 점수 1점을 부여하고, "
        "정답과 맞지 않거나 불만족스러우면 0점을 부여하세요. "
        "부분적으로 일치하면 0.5점을 부여하세요."
    )

    # 사용자 질의, AI 응답, 정답, 평가 프롬프트를 결합하여 평가용 프롬프트 생성
    evaluation_prompt = (
        f"User Query: {query}\n"
        f"AI Response:\n{ai_response.choices[0].message.content}\n"
        f"True Response: {data[id]['ideal_answer']}\n"
        f"{evaluate_system_prompt}"
    )     

    # 평가 시스템을 사용하여 점수를 생성.
    evaluation_response = rc.ch4_generate_response( evaluate_system_prompt , evaluation_prompt )

    # 평가 결과 
    return {
            'score':evaluation_response.choices[0].message.content , 
            'True Response' : data[id]['ideal_answer'] ,
            'answer': query ,
            'question':ai_response.choices[0].message.content ,
            }  
#----------------- Context-Enriched Rag : end
#----------------- Contextual Chunk Headers : start
@router.get("/contextualchunkheaders/{id}" , tags=['AI / Rag'],  description='헤드 추출 RAG')
def contextualchunkheaders( id:int = 0 ):
    ''' 헤드 추출 RAG'''
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    
    # 텍스트 다시 추출
    extracted_text = rc.ch5_extract_text_from_pdf( pdf_path )        

    # 추출된 텍스트를 헤더로 청크.
    # 청크 크기는 1000자, 겹침은 200자 사용
    text_chunks = rc.ch5_chunk_text_with_headers( extracted_text , 1000 , 200 )

    # 생성된 헤더와 함께 샘플 청크 인쇄.
    #print(f"header : {text_chunks[0]['header']}")
    #print(f"content : {text_chunks[0]['text']}")

    # 각 청크에 대한 임베딩 생성
    embeddings = []

    # 진행률 표시줄을 사용.
    for chunk in tqdm(text_chunks , desc='Generating embeddings'):
        # 청크의 텍스트에 대한 임베딩 만들기
        text_embedding = rc.ch5_create_embeddings( chunk['text'] )
        # 청크의 헤더에 임베딩을 생성
        header_embedding = rc.ch5_create_embeddings( chunk['header'] )
        # 청크의 헤더 , 텍스트및 해당 임베딩을 목록에 추가
        embeddings.append(
            { 
                'header' : chunk['header'] ,
                'text' : chunk['text'] ,                
                'embedding' : text_embedding ,
                'header_embedding' : header_embedding,
            }
        )
    
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    query = data[id]['question']

    # 가장 연관성 높은 상위 2개 청크 검색
    top_chunks = rc.ch5_semantic_search( query , embeddings , k=2 )
    '''
    # 쿼리 출력
    print(f'쿼리 :{query}')
    # 검색된 각 청크를 번호와 구분석을 포함하여 출력
    for i, chunk in enumerate(top_chunks):
        print(f"Header : {chunk['header']}")
        print(f"컨텍스트 {i+1}:\n{chunk['text']}\n{'-'*30}")    
    '''
    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 컨텍스트에 기반하여 엄격하게 대답하는 AI어시턴트입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    ) 

    # 상위 문맥 청크들을 기반으로 사용자 프롬프트 생성
    # 각 청크는 딕셔너리 형식이며 header와 text 키를 가짐
    user_prompt = "\n".join(
        [
            f"제목:{chunk['header']}\n내용:\n{chunk['text']}"
            for chunk in top_chunks
        ]
    )

    # 질문을 프롬프트 마지막에 추가
    user_prompt = f"{user_prompt}\n질문:{query}"

    #AI응답 생성
    ai_response = rc.ch5_generate_response(system_prompt , user_prompt)

    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 지능형 평가 시스템입니다. "
        "제공된 문맥을 바탕으로 AI 어시턴트의 응답을 평가하십시오. "
        "- 정답과 매우 유사한 경우 점수는 1점을 부여하십시오. "
        "- 부분적으로 맞은 경우 0.5점을 부여하세요."
        "- 틀린 경우 0점을 부여하십시오."
        "반환값은 오직 점수(0.0.5,1) 중 하나여야 합니다."
    )

    # 검증 데이터에서 정답 추출
    true_answer = data[id]['ideal_answer']

    # 평가용 프롬프트 생성
    evaluation_prompt = (
        f"사용자 질문 : {query}"
        f"AI 응답:{ai_response.choices[0].message.content}"
        f"정답: {true_answer}"
        f"{evaluate_system_prompt}"
    )   

    # 평가 점수 생성(AI 모델에 평가 프롬프트 전달)  
    evaluation_response = rc.ch5_generate_response( evaluate_system_prompt , evaluation_prompt )

    # 평가 결과 
    return {
            '점수'  : evaluation_response.choices[0].message.content , 
            '정답'  : true_answer ,
            '질문'  : query ,
            '응답'  : ai_response.choices[0].message.content ,
            }  
#----------------- Contextual Chunk Headers : end

#----------------- Document augmentation RAG : start
@router.get("/documentaugmentation/{id}", tags=['AI / Rag'],  description='Docment Augmentation RAG')
def documentaugmentation( id:int = 0 ):
    ''' Docment Augmentation RAG'''
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"

    # 문서 처리: 텍스트 추출, 청크 분할, 질문 생성, 벡터 저장소 구축
    text_chunks, vector_store = rc.ch6_process_document(
        pdf_path, 
        chunk_size=1000,       # 각 청크는 1000자
        chunk_overlap=200,     # 청크 간 200자 중첩
        questions_per_chunk=3  # 청크당 질문 3개 생성
    )

    # 벡터 저장소에 저장된 항목 개수 출력
    print(f"벡터 저장소에 저장된 항목 수: {len(vector_store.texts)}개")

    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 쿼리 추출
    query = data[id]['question']    

    # 의미 기반 검색을 통해 관련 있는 콘텐츠를 찾습니다.
    search_results = rc.ch6_semantic_search1(query, vector_store, k=5)

    print("Query:", query)
    print("\nSearch Results:")

    # 검색 결과를 타입에 따라 분류합니다.
    chunk_results = []
    question_results = []

    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_results.append(result)
        else:
            question_results.append(result)

    # 먼저 문서 청크 결과 출력
    print("\nRelevant Document Chunks:")
    for i, result in enumerate(chunk_results):
        print(f"Context {i + 1} (유사도: {result['similarity']:.4f}):")
        print(result["text"][:300] + "...")
        print("-----------------------------")

    # 이어서 관련 질문 결과 출력
    print("\nMatched Questions:")
    for i, result in enumerate(question_results):
        print(f"Question {i + 1} (유사도: {result['similarity']:.4f}):")
        print(result["text"])
        chunk_idx = result["metadata"].get("chunk_index", "N/A")
        print(f"연결된 청크 인덱스: {chunk_idx}")
        print("-----------------------------")

    # 검색 결과로부터 응답 생성을 위한 컨텍스트를 준비합니다.
    context = rc.ch6_prepare_context(search_results)

    # 쿼리와 문맥을 기반으로 AI 응답을 생성합니다.
    response_text = rc.ch6_generate_response(query, context)

    # 쿼리와 응답 출력
    print("\n질문(Query):", query)
    print("\n응답(Response):")
    print(response_text)

    # 검증 데이터에서 기준 정답을 가져옵니다.
    reference_answer = data[id]['ideal_answer']

    # 생성된 응답을 기준 정답과 비교하여 평가합니다.
    evaluation = rc.ch6_evaluate_response(query, response_text, reference_answer)

    # 평가 결과 출력
    print("\nEvaluation:")
    print(evaluation)

    # 텍스트 청크에 대해 임베딩을 생성합니다.
    response = rc.ch6_create_embeddings(text_chunks)

    # 의미 기반 검색을 수행하여 쿼리와 가장 관련성 높은 상위 2개의 텍스트 청크를 찾습니다.
    top_chunks = rc.ch6_semantic_search2(query, text_chunks, response.data, k=2)

    # 쿼리를 출력합니다.
    print("질문(Query):", query)

    # 상위 2개의 관련 텍스트 청크를 출력합니다.
    for i, chunk in enumerate(top_chunks):
        print(f"\n컨텍스트 {i + 1}:\n{chunk}\n{'-' * 37}")    
    
    # AI 어시스턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 컨텍스트에 기반하여 엄격하게 응답하는 AI 어시스턴트입니다. "
        "제공된 문맥에서 직접적으로 답변을 도출할 수 없는 경우, 다음과 같이 응답하세요: "
        "'I do not have enough information to answer that.'"
    )

    # 검색된 상위 청크들을 바탕으로 사용자 프롬프트 생성
    user_prompt = "\n".join([
        f"Context {i + 1}:\n{chunk}\n=====================================\n"
        for i, chunk in enumerate(top_chunks)
    ])
    user_prompt = f"{user_prompt}\nQuestion: {query}"

    # AI 응답 생성
    ai_response = rc.ch6_generate_response(system_prompt, user_prompt)

    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 AI 어시스턴트의 응답을 평가하는 지능형 평가 시스템입니다. "
        "AI 응답이 기준 정답과 매우 유사하면 점수 1을 부여하세요. "
        "응답이 부정확하거나 부적절하면 점수 0을 부여하세요. "
        "부분적으로 일치하면 점수 0.5를 부여하세요."
    )

    # 사용자 쿼리, AI 응답, 기준 정답을 포함한 평가용 프롬프트 생성
    evaluation_prompt = (
        f"User Query: {query}\n"
        f"AI Response:\n{ai_response.choices[0].message.content}\n"
        f"True Response: {data[id]['ideal_answer']}\n"
        f"{evaluate_system_prompt}"
    )

    # 평가 수행
    evaluation_response = rc.ch6_generate_response(evaluate_system_prompt, evaluation_prompt)
    
    # 평가 결과 
    return {
            '점수'  : evaluation_response.choices[0].message.content , 
            '정답'  : data[id]['ideal_answer'] ,
            '질문'  : query ,
            '응답'  : ai_response.choices[0].message.content ,
            }      
#----------------- Document augmentation RAG : end

#----------------- ch7 Query Transfomation RAG : start
@router.get("/querytransformation/{id}" , tags=['AI / Rag'],  description='Query Transformation RAG')
def querytransformation( id:int = 0 ):
    '''Query Transformation RAG'''
    # 예시 쿼리
    original_query = "AI가 업무 자동화와 고용에 미치는 영향은 무엇인가요?"
    

    # 쿼리 변환 적용
    print("Original Query:")
    print(original_query)

    # 1. 쿼리 재작성 (더 구체적으로)
    rewritten_query = rc.ch7_rewrite_query(original_query)
    print("\nRewritten Query:")
    print(rewritten_query)

    # 2. 스텝백 쿼리 생성 (더 일반화된 문맥 요청)
    step_back_query = rc.ch7_generate_step_back_query(original_query)
    print("\nStep-back Query:")
    print(step_back_query)

    # 3. 하위 쿼리 분해 (다양한 측면으로 나눔)
    sub_queries = rc.ch7_decompose_query(original_query, num_subqueries=4)
    print("\nSub-queries:")
    for i, query in enumerate(sub_queries, 1):
        print(f"   {i}. {query}")    

    # 검증 데이터를 JSON 파일에서 불러옵니다.
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)

    # 첫 번째 쿼리와 기준 정답을 추출합니다.
    query = data[id]['question']
    reference_answer = data[id]['ideal_answer']

    # PDF 파일 경로 설정
    pdf_path    = 'D:\\python_workspace\\FastApi\\Area\\Rag\\AI_Understanding.pdf'
    pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"

    # 다양한 쿼리 변환 기법에 대한 RAG 평가 실행
    evaluation_results = rc.ch7_evaluate_transformations(pdf_path, query, reference_answer)

#-----------------ch7 Query Transfomation RAG : end

#-----------------ch8 Reranking RAG : start
@router.get("/reranking/{id}", tags=['AI / Rag'],  description='Reranking RAG')
def reranking( id:int = 0 ):
    ''' Reranking RAG
    시스템에서 검색 품질을 정밀하게 개선하기 위한 핵심 기술로, 초기 검색 결과에 대해 2차 정렬 과정을 수행하여
    최정적으로 생성에 사용할 문서를 선별하는 방식이다. 초기 검색 속도 위주로 진행되지만, 재랭크는 정확도와 의미기반
    정합성을 고려하여 최종 응답의 품질을 극대화 한다.
    ※ reranking 주요 개념및 단계
    1. 초기검색(fist-pass Retrieval)
    - 벡터 임베딩 기반의 기본 유사도 검색을 사용하여 쿼리와 유사한 문서 청크들을 빠르게 검색한다.
    - 이단계는 속도는 빠르지만 정확도는 상대적으로 낮을수 있음.
    2. 문서 채점(scoring)
    - 검색된 각 문서에 대해 쿼리와의 관련성을 정밀하게 평가.
    - 단순 임베딩 유사도 대신, cross-Encoder 또는 LLM 기반 점수 평가 모델을 사용하여 정교한 채점을 수행.
    3. 재정렬(reranking)
    - 점수에 따라 문서들을 내림 차순으로 정렬한다.
    - 이 과정에서 덜 관련된 청크는 하위로 밀려나며, 최종 응답에 포함되지 않을 수 있다.
    4. 선택(Selection)
    - 상위 N개의 청크를 최종 선택하여 LLM에 입력한다.
    - 선택된 문서들은 더 높은 신뢰도 와 일관성을 기반으로 응답 생성.

    '''
    # JSON 파일에서 검증용 데이터 로드
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)

    # 검증 데이터에서 첫 번째 질문 추출
    query = data[id]['question']

    # 검증 데이터에서 해당 질문의 정답(참조 답변) 추출
    reference_answer = data[id]['ideal_answer']

    # 사용할 PDF 파일 경로 정의    
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'
    # PDF 문서를 처리하여 벡터 저장소 생성
    vector_store = rc.ch8_process_document(pdf_path)

    # 테스트용 질의 정의
    query = "AI는 우리의 생활과 업무 방식을 변화시킬 잠재력을 가지고 있을까요?"

    # 다양한 검색 및 재정렬 방식 비교
    print("=== 검색 및 재정렬 방식 비교 ===")

    # 1. 기본 검색 (재정렬 없이)
    print("\n--- [1] STANDARD RETRIEVAL ---")
    standard_results = rc.ch8_rag_with_reranking(query, vector_store, reranking_method="none")
    print(f"\n[질문]\n{query}")
    print(f"\n[응답]\n{standard_results['response']}")

    # 2. LLM 기반 재정렬
    print("\n--- [2] LLM-BASED RERANKING ---")
    llm_results = rc.ch8_rag_with_reranking(query, vector_store, reranking_method="llm")
    print(f"\n[질문]\n{query}")
    print(f"\n[응답]\n{llm_results['response']}")

    # 3. 키워드 기반 재정렬
    print("\n--- [3] KEYWORD-BASED RERANKING ---")
    keyword_results = rc.ch8_rag_with_reranking(query, vector_store, reranking_method="keywords")
    print(f"\n[질문]\n{query}")
    print(f"\n[응답]\n{keyword_results['response']}")    

    # LLM 기반 재정렬 결과가 기본 검색 결과보다 더 나은지 평가
    evaluation = rc.ch8_evaluate_reranking(
        query=query,  # 사용자 질의
        standard_results=standard_results,  # 기본 검색 결과
        reranked_results=llm_results,  # LLM 기반 재정렬 결과
        reference_answer=reference_answer  # 참조 정답 (비교 기준)
    )

    # 평가 결과 출력
    print("\n***평가 결과***")
    print(evaluation)

#-----------------ch8 Reranking RAG : end

#-----------------ch9 Relevant Segment Extraction RAG : start
@router.get("/rserag/{id}" , tags=['AI / Rag'],  description='Relevant Sement Extraction RAG')
def rserag( id:int = 0 ):
    '''Relevant Sement Extraction RAG
    향상된 RAG를 위한 관련 세그먼트 추출은 단순히 유사한 청크들을 개별적으로 검색하는 기존 방식에서 벗어나,
    문서 내에서 의미적으로 연결된 연속적인 텍스트 세그먼트를 식별하고 구성하는 방식으로 RAG시스템의 문맥 품질을
    향상 시키는 전략이다.
    기존 RAG 시스템에서는 벡터 유사도를 기반으로 TOP-k 청크를 독립적으로 검색하지만 이 방식은 문서의 논리적
    흐름이나 문맥적 연속성을 고려하지 않기 때문에 응답이 단절되거나 불완전한 정보에 기반해 생성될 가능성이 높다.
    반면 RSE는 서로 연관된 청크들이 문서 내에서 함께 등장하는 경향을 활용하여 관련된 청크들을
    클러스터 단위로 그룹화 하고 이를 기반으로 더 일관된 문맥을 제공한다.
    ※ RSE의 작동 원리
    1. 관련 청크는 문서 내에서 클러스터링되는 경향이 있다.
    - 의미적으로 연관된 문장은 종종 인접하거나 동일한 섹션에 위치한다.
    - RSE는 이러한 위치 기반 연속성과 의미적 유사성을 동시에 고려하여 관련 세그먼트를 식별한다.
    2. 연속된 텍스트를 재구성하여 더 나은 문맥 제공한다.
    - LLM은 개별 청크보다 일관된 흐름을 가진 연속적인 문단에서 더 우수한 응답을 생성한수 있다.
    ※ 구현단계
    1. 초기청크검색 : 사용자의 쿼리에 대한 top-k 청크를 벡터 검색으로 검색.
    2. 관련 세그먼트를 클러스터링한다.
    - 검색된 청크의 문서내 위치를 활용하여, 인접하거나 유사한 청크끼리 그룹화한다.
    - 클러스터링의 기준 : 위치기준(연속된 청크ID) , 의미기준(내적 유사도 평균이 일정 이상일경우)
    3. 세그먼트 재구성
    - 각 클러스터의 청크들을 하나의 연속된 세그먼트로 합쳐서 새로운 컨텍스트 블록으로 구성한다.
    - 이때 필요시 중복 문장 제거, 접속어 보완 등의 후처리를 적용할수 있다.
    4. 응답평가
    - 생성된 응답의 문맥 일관성, 정보 충실도 , 관련성을 기준으로 평가한다.
    '''
    # 검증용 JSON 파일에서 데이터 로드
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)

    # 첫 번째 테스트 케이스의 질의 추출
    query = data[id]['question']

    # 참조 정답(ideal_answer) 추출
    reference_answer = data[id]['ideal_answer']

    # 사용할 PDF 문서 경로
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'

    # 두 가지 RAG 방식(RSE vs Standard Top-K)에 대한 평가 실행
    results = rc.ch9_evaluate_methods(pdf_path, query, reference_answer)
#-----------------ch9 Relevant Segment Extraction RAG : end

#-----------------ch10 문맥압축 Contextual Compression RAG : start
@router.get("/contextualcompression/{id}" , tags=['AI / Rag'],  description='Contextual Compression RAG')
def contextualcompression( id:int = 0 ):
    '''Contextual Compression RAG
    파이프라인에서 검색된 텍스트 청크를 정제하여, 쿼리와 관련된 정보만을 추출하고 컨텍스트의 정보 밀도를 극대화하는 기법이다.
    이 기술은 특히 LLM의 컨텍스트 창이 제한되어 있는 환경에서 노이즈를 줄이고 응답의 품질을 향상시키는 데 매우 효과적이다.

    ※ 문맥 압출이 필요한 이유
    기본 RAG에서는 벡터 유사도 기반 검색을 통해 전체 청크단위를 검색하므로 , 해당 청크에 일부 유용한 정보가 포함되어 
    있더라도 불필요한 문장이 나 단락이 함께 전달되는 경우가 많다.
    - LLM의 입력 제한을 초과하거나
    - 중요한 문장이 잘려 나가고,
    - 응답이 분산되거나 비논리적으로 생성될 가능성이 있다.
    이를 해결하기 위해 문맥 압출을 적용하여 쿼리 중심의 고밀도 정보만 LLM에 제공하도록 한다.

    ※ 문맥 압출 파이프라인 단계
    1. 초기 단계
    - 사용자 쿼리에 따라 top-k 청크를 벡터 유사도로 검색.
    - 이 단계는 속도 위주로 작동하므로 정밀도는 낮음.
    2. 문장 단위 분할
    - 각 청크를 문장 단위로 분할하여 세부 평가가 가능하도록 준비.
    - 문장 경계 탐지 또는 구두점 기반 분할 적용.
    3. 쿼리-문장 유사도 평가
    - 사용자 쿼리와 각 문장 간의 의미적 유사도를 임베딩 또는 LLM으로 평가한다.
    => Cosine Similarity
    => Sentence Transformers 기반 scoring
    => LLM을 사용한 relevance 판단
    4. 비관련 문장 제거
    - 유사도 점수가 임계값 이하인 문장은 제거.
    - 일정 기준 이상 관련성 높은 문장만 남기고 압축된 텍스트로 재구성.
    5. 압축결과 구성
    - 여러 청크에서 압축된 문장들을 조합하여 새로운 고밀도 문맥 블록을 구성.
    - 문맥 흐름이 깨지지 않도록 자연스러운 문장 연결보완하는 역할을 한다.
    6. 응답생성
    - 최종 압축된 문맥 블록을 LLM에 전달하여 응답 생성.
    - 더 적은 토큰으로 더 정확한 응답이 가능하다.
    '''
    # AI 윤리에 관한 정보를 담고 있는 PDF 문서 경로
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'

    # 문서에서 관련 정보를 추출하기 위한 사용자 질의
    query = "의사 결정에 AI를 사용하는 것과 관련된 윤리적 우려는 무엇인가요?"  

    # (선택사항) 평가에 사용할 참조 정답
    reference_answer = """
    의사 결정에 AI를 사용하면 몇 가지 윤리적 문제가 제기됩니다.
    - 특히 채용, 대출, 법 집행과 같은 중요한 영역에서 AI 모델의 편향성은 불공정하거나 차별적인 결과를 초래할 수 있습니다.
    - AI 기반 의사 결정의 투명성과 설명 가능성이 부족하면 개인이 불공정한 결과에 이의를 제기하기 어렵습니다.
    - AI 시스템이 명시적인 동의 없이 방대한 양의 개인 데이터를 처리하기 때문에 개인정보 보호 위험이 발생합니다.
    - 자동화로 인한 일자리 대체 가능성은 사회적, 경제적 우려를 불러일으킵니다.
    - 또한 AI 의사결정은 소수의 대형 기술 기업에 권력이 집중되어 책임 문제가 발생할 수 있습니다.
    - AI 시스템의 공정성, 책임성, 투명성을 보장하는 것은 윤리적 배포를 위해 필수적입니다.
    """

    # 다양한 압축 기법을 사용하여 평가 실행
    # 압축 방식:
    # - "selective": 중요 정보는 유지하고 덜 중요한 내용은 생략
    # - "summary": 질의에 관련된 내용을 간결하게 요약
    # - "extraction": 관련 문장을 문서에서 그대로 추출
    results = rc.ch10_evaluate_compression(  
        pdf_path=pdf_path,  
        query=query,  
        reference_answer=reference_answer,  
        compression_types=["selective", "summary", "extraction"]  
    )    

    # 다양한 압축방식의 결과를 시각적으로 비교 출력
    rc.ch10_visualize_compression_results(results)

#-----------------ch10 문맥압축 Contextual Compression RAG : end

#-----------------ch11 피드백 루프 Feedback Loop RAG : start
@router.get("/feedbackloop/{id}" , tags=['AI / Rag'],  description='Feedback Loop RAG')
def feedbackloop( id:int = 0 ):
    '''Feedback Loop RAG
    이용자 상호작용을 기반으로 지속적으로 개선되는 학습형 구조를 갖추도록 설계된 메커니즘이다.
    기존 RAG시스템은 정적인 벡터 임베딩과 유사도 검색에만 의존하는 반면, 피드백 루프는 사용자의 
    실제 피드백을 반영하여 검색과 생성 품질을 점진적으로 향상시키는것이 목표이다.
    ※ 피드백 루프의 필요성 :     기존RAG시스템의 한계는 다음과 같다.
    - 검색 결과가 항상 동일한 유사도 기준에 따라 반환되어 문맥 변화나 사용자 의도 적응이 어렵다.
    - 잘못된 정보가 반복될 가능성이 있으며 검증된 응답이 데이터베이스에 반영되지 않는다.
    - 사용자의 만족도나 응답 효과성에 대한 학습이 없다.
    ※ 피드백 루프의 주요기능
    1. 효과가 있었던 것과 없었던 것을 기억
    - 사용자가 높은 평가를 준 Q&A쌍은 양질의 지식 자산으로 기록된다.
    - 부정확한 응답은 다시 학습하거나 제외 할수 있는 학습 대상으로 분류한다.
    2. 문서 연관성 점수 조정
    - 문서나 청크에 대해 반복적으로 높은 평가를 받은 경우, 해당 청크의 연관성 점수를 동적으로 상향조정.
    - 반대로 유사도는 높았지만 자주 부정적 피드백을 받은 청크는 패너틸를 부여하여 검색 우선순위에서 낮춘다.
    3. 성공적인 Q&A쌍을 지식 창고에 통합
    - 특정 쿼리에 대해 우수한 응답이 생성되었다면, 해당 Q&A쌍을 별도의 Retrieval 대상 지식창고로 저장할수 있다.
    - 이후 유사 질문이 들어오면, 벡터 검색보다 빠르게 해당 사례를 참조하여 응답품질을 향상시킨다.
    4. 각 상호작용을 통해 더욱 스마트해짐
    - 피드백은 단순 통계가 아니라 랭킹학습, 임베딩 리트레이닝, 규칙 학습등에 활용
    - 반복적인 상호작용속에서 사용자 맞춤화도 가능하다.
    
    '''
    # AI 문서 경로 설정
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'
    
    # 테스트 질의 정의
    test_queries = [
        "신경망이란 무엇이며 어떻게 작동하나요?",
        ### 테스트 속도를 위해 일부 질의는 주석 처리됨 ###
        
        # "Describe the process and applications of reinforcement learning.",
        # "What are the main applications of natural language processing in today's technology?",
        # "Explain the impact of overfitting in machine learning models and how it can be mitigated."
    ]

    # 정답/참조 응답 정의
    # (평가 및 synthetic feedback 생성을 위한 기준)
    reference_answers = [
        "신경망은 인간의 뇌가 작동하는 방식을 모방한 프로세스를 통해 데이터 집합의 기본 관계를 인식하려는 일련의 알고리즘입니다. 신경망은 여러 층의 노드로 구성되며 각 노드는 뉴런을 나타냅니다. 신경망은 예상 결과와 비교한 출력의 오차에 따라 노드 간 연결의 가중치를 조정하여 작동합니다.",
        ### 테스트 속도를 위해 일부 정답 응답은 주석 처리됨 ###
        
        # "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It involves exploration, exploitation, and learning from the consequences of actions. Applications include robotics, game playing, and autonomous vehicles.",
        # "The main applications of natural language processing in today's technology include machine translation, sentiment analysis, chatbots, information retrieval, text summarization, and speech recognition. NLP enables machines to understand and generate human language, facilitating human-computer interaction.",
        # "Overfitting in machine learning models occurs when a model learns the training data too well, capturing noise and outliers. This results in poor generalization to new data, as the model performs well on training data but poorly on unseen data. Mitigation techniques include cross-validation, regularization, pruning, and using more training data."
    ]
    # 평가 워크플로우 실행
    # - RAG 성능을 피드백 전후로 비교
    # - synthetic feedback 포함
    evaluation_results = rc.ch11_evaluate_feedback_loop(
        pdf_path=pdf_path,
        test_queries=test_queries,
        reference_answers=reference_answers
    )    
    # 대화형 예제 실행
    print("\n\n***INTERACTIVE EXAMPLE***")
    print("Enter your query about AI:")
    user_query = input()

    # 누적 피드백 로드
    all_feedback = rc.ch11_load_feedback_data()
    

    # 전체 워크플로 실행
    result = rc.ch11_full_rag_workflow(
        pdf_path=pdf_path,
        query=user_query,
        feedback_data=all_feedback,
        fine_tune=True
    )    

    # 피드백 적용 전후 성능 비교 결과 추출
    comparisons = evaluation_results['comparison']

    # 분석 결과 출력
    # 각 질의에 대해 피드백이 응답에 미친 영향 확인
    print("\n피드백 효과 분석 결과\n")

    for i, comparison in enumerate(comparisons):
        print(f"질의 {i+1}: {comparison['query']}")
        print("피드백 반영 효과 분석:")
        print(comparison['analysis'])
        print("\n" + "-"*50 + "\n")

    # 응답 길이를 기반으로 충실도 간접 평가
    # 라운드별 응답 길이 수집
    round_responses = [evaluation_results[f'round{round_num}_results'] for round_num in range(1, len(evaluation_results) - 1)]
    response_lengths = [[len(r["response"]) for r in round] for round in round_responses]

    print("\n응답 길이 비교")
    avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]

    for round_num, avg_len in enumerate(avg_lengths, start=1):
        print(f"라운드 {round_num}: 평균 응답 길이 = {avg_len:.1f}자")

    # 라운드 간 응답 길이 변화율 출력
    if len(avg_lengths) > 1:
        changes = [
            (avg_lengths[i] - avg_lengths[i - 1]) / avg_lengths[i - 1] * 100
            for i in range(1, len(avg_lengths))
        ]
        for round_num, change in enumerate(changes, start=2):
            print(f"라운드 {round_num - 1}에서 라운드 {round_num}로의 길이 변화율: {change:.1f}%")

#-----------------ch11 피드백 루프 Feedback Loop RAG : end

#-----------------ch12 적응형검색 adaptive Retrieval RAG : start
@router.get("/adaptiveretrieval/{id}" , tags=['AI / Rag'],  description='adaptive Retrieval RAG')
def adaptiveretrieval( id:int = 0 ):
    ''' 적응형 검색 adaptive Retrieval RAG
    사용자 쿼리의 성격에 따라 가장 적합한 검색 전략을 동적으로 선택함으로써, 다양한 유형의 질문에 대해 보다
    정확하고 상황에 맞는 응답을 생성할수 있도록 진화된 RAG 프레임워크이다.
    기존RAG 시스템은 모든 질문에 동일한 벡터 유사도 기반 검색 방식을 적용하여 응답을 생성하지만,
    사실 기반 질문(fact-based)과 의견 기반 질문(opinion-based) 또는 문맥강화질문(contextual reasoning)
    은 그 성격이 다르기 때문에 동일한 전략으로는 만족스러운 응답을 제공하기 어렵다.
    ※ 핵심 개념 : adaptive Retrieval 작동방식
    1. 쿼리 유형 분류 : 입력된 질문을 자연어 처리 모델 또는 프롬프트 기반 규칙을 통해 다음과 같은 유형으로 분류.
    - 사실기반(factual) : 명확한 정답이 있는 질문
    - 분석기반(analytical) : 비교, 요약, 해석이 필요한 질문
    - 의견기반(opinion) : 주관적 견해나 사례가 필요한 질문
    - 문맥기반(contextual) : 배경 설명이나 상황에 따른 해석이 필요한 질문
    2. 검색 전략 선택 : 분류된 쿼리 유형에 따라 적절한 검색 전략 선택.
    - 사실기반 : 빠른 Top-k유사도 검색 , 정답 포함 청크 중심 응답.
    - 분석기반 : 멀티 청크 통합 검색 , 다수 문서 비교 또는 요약.
    - 의견기반 : 유사 질문의 기반 Q&A검색, 사용자 피드백 기반선택.
    - 문맥기반 : 문맥 강화 검색
    3. 전문 검색 기법 실행 : 전략에 따른 다음 기법들을 조합해서 실행한다.
    - Crosee-Encoder기반 재랭크 : 세밀한 관련성 점수 평가.
    - Contextual Chunking : 문맥 유지를 위한 청크 설계.
    - Step-back Prompting : 배경 검색 유도
    - Feedback Loop : 이전 상호작용 기반 정보 반영
    - RSE : 연속된 의미 단위로 세그먼트 구성.
    - Copmression : 응답품질을 위한 문맥 압축.
    4. 맞춤형 응답생성 : 검색 결과를 기반으로 LLM은 질문 유형에 최적화 된 프롬프트를 사용해 응답을 생성한다.
    - 분석 질문의 경우 -> "두 시스템의 장점을 나열하고 차이점을 비교해주세요."
    - 문맥 질문의 경우 -> "해당 정책의 적용 조건과 예외 조항을 함께 설명해주세요."
    ※ Adaptive Retrieval의 장점
    - 질문 적합성 향상 : 각 질문 유형에 맞는 검색 전략을 적용함을써 정답율및 만족도 향상.
    - 응답 품질 고도화 : 다양한 문맥과 응답 유형에 최적화된 텍스트 생성가능.
    - 유연한 검색 구조 : 단일 전략에 의존하지 않고 쿼리에 따라 동적 구성가능.
    - 도메인 적응성 강화 : 법율 , 의료 , 정책 , 상담 등 다양한 도메인에 맞춤화 용이.
    ※ 결론
    Adaptive Retrieval은 RAG 시스템이 질문마다 다른 요구를 이해하고, 전략적으로 대응할수 있도록 설계된 
    고도화된 접근 방식이다. 이를 통해 RAG는 단순한 검색 기반 생성 시스템을 넘어, 지능적인 질의 대응 플랫폼으로
    발전할수 있으며, 특히 복잡하고 다양성이 높은 실무 환경에서 뛰어난 성능을 발휘할수 있다.
    '''
    # 지식 소스 문서의 경로 설정
    # 이 PDF 파일은 RAG 시스템이 활용할 정보를 포함하고 있음
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'

    # 다양한 질의 유형을 포함하는 테스트 질의 정의
    # 적응형 검색(adaptive retrieval)이 질의 의도를 어떻게 처리하는지 시연
    test_queries = [
        "설명 가능한 AI(XAI)란 무엇인가요?", # 사실 기반 질의 - 정의/구체 정보 요청
        # "How do AI ethics and governance frameworks address potential societal impacts?", # 분석형 질의 - 포괄적 분석 필요
        # "Is AI development moving too fast for proper regulation?", # 의견형 질의 - 다양한 관점 요청
        # "How might explainable AI help in healthcare decisions?", # 문맥형 질의 - 사용자 상황 고려 필요
    ]

    # 보다 철저한 평가를 위한 참조 정답 정의
    # 응답 품질을 객관적으로 평가하는 데 사용 가능
    reference_answers = [
        "설명 가능한 AI(XAI)는 의사 결정 방식에 대한 명확한 설명을 제공함으로써 AI 시스템을 투명하고 이해하기 쉽게 만드는 것을 목표로 합니다. 이는 사용자가 AI 기술을 신뢰하고 효과적으로 관리하는 데 도움이 됩니다.",
        # "AI 윤리 및 거버넌스 프레임워크는 AI 시스템이 책임감 있게 개발되고 사용될 수 있도록 가이드라인과 원칙을 수립하여 잠재적인 사회적 영향을 해결합니다. 이러한 프레임워크는 공정성, 책임성, 투명성, 인권 보호에 중점을 두어 위험을 완화하고 유익한 결과를 촉진합니다.",
        # "AI 개발이 너무 빠르게 진행되어 적절한 규제가 필요한지에 대한 의견은 다양합니다. 일부에서는 빠른 발전이 규제 노력보다 빨라 잠재적인 위험과 윤리적 문제를 야기한다고 주장합니다. 다른 사람들은 새로운 도전 과제를 해결하기 위해 규제가 함께 진화하면서 혁신이 현재의 속도로 계속되어야 한다고 생각합니다.",
        # "설명 가능한 AI는 AI 기반 추천에 대한 투명하고 이해하기 쉬운 인사이트를 제공함으로써 의료진의 의사결정에 큰 도움을 줄 수 있습니다. 이러한 투명성은 의료 전문가가 AI 시스템을 신뢰하고, 정보에 입각한 결정을 내리고, AI 제안의 근거를 이해함으로써 환자 치료 결과를 개선하는 데 도움이 됩니다."
    ]    
    # 적응형 검색과 표준 검색을 비교하는 평가 실행
    # 각 질의를 두 방법으로 처리하고 결과를 비교함
    evaluation_results = rc.ch12_evaluate_adaptive_vs_standard(
        pdf_path=pdf_path,                  # 지식 추출을 위한 소스 문서
        test_queries=test_queries,          # 평가할 테스트 질의 목록
        reference_answers=reference_answers  # 비교를 위한 참조 정답 (옵션)
    )

    # 결과는 표준 검색과 적응형 검색의 성능을 질의 유형별로 비교하여
    # 적응형 전략이 더 나은 결과를 제공하는 경우를 강조함
    print(evaluation_results["comparison"])    
#-----------------ch12 적응형검색 adaptive Retrieval RAG : end

#-----------------ch13 Self-RAG : start
@router.get("/selfrag/{id}" , tags=['AI / Rag'],  description='Self-RAG')
def selfrag( id:int = 0 ):
    '''Self-RAG(Self Retrieval Augmented Generation)
    전통적인 RAG의 한계를 극복하고자 설계된 고급 자율형 RAG 시스템으로, 검색과 생성을 단순히 순차적으로 실행하지 않고,
    검색이 언제 필요하고 어떤 정보를 사용할지를 스스로 판단하며 동적으로 수행하는 구조를 갖는다.
    이를 통해 응답의 품질, 안정성, 효율성이 동시에 향상된다.
    기존의 RAG는 쿼리가 입력되면 항상 외부 문서를 검색하고, 그결과를 기반으로 응답을 생성한다.
    하지만 모든 쿼리가 외부 검색을 필요로 하는것은 아니며, 검색된 정보가 항상 도움이 되는것도 아니다.
    Self-RAG는 이러한 문제를 인식하고 각 단계마다 판단과 검증 절차를 추가하여 더 정밀하게 정보를 활용한다.
    ※ Self-RAG의 주요 구성 요소
    1. 검색 결정(Search Decision)
    - 쿼리를 입력받은 후, 검색이 필요한지 여부를 LLM이 먼저 판단하다.
    - 예 : "오늘은 무슨 요일인가요" -> 검색 불필요 / "GDPR" 의 핵심원칙 -> 검색필요.
    - 구현 예시 : LLM에게 아래와 같은 질문을 하여 판단 "이 질문에 답변하기 위한 외부 정보를 검색해야 하나요?"
    2. 문서검색(Document Retrieval)
    - 검색이 필요하다고 판단된 경우에만 외부 지식 저장소(VectorDB)에 관련문서 검색.
    - 검색 효율성과 정확성을 위해 RSE,Contextual Compression, Adaptive Retrieval 등과 결함.
    3. 관련성 평가(Relevance Assessment)
    - 검색된 각 문서나 청크에 대해, 쿼리와의 의미적 관련성을 다시 평가하여 가중치를 부여하거나 필터링한다.
    - 점수화된 관련성은 이후 생성에 반영될수 있다.
    - 예시방법 : LLM을 사용한 pair-wise비교 "이문장은 질문에 얼마나 관련있을까"Cross-Encoder 기반 관련성 점수 활용.
    4. 응답 생성(Answer Generation)
    - 관련성 높은 컨텍스트만 선택ㅎ여 LLM이 응답을 생성한다.
    - 불필요한 문서가 제거된 상태에서 고밀도의 문맥 기반 응답이 생성되므로 품질이 높다.
    5. 지원 평가(Support Assessment)
    - 생성된 응답이 제공된 컨텍스트에 실제로 기반하고 있는지를 검증한다.
    - 이는 hallucination 방지에 매우 효과적이다.
    - 예시 : "이 응답은 아래 문서에 기반하고 있습니가?" , "응답의 각 주장에 대해 출처가 문서에 존재합니가?"
    - 최종 응답이 사용자의 질문 의도에 부합하고, 실제로 유용한지를 평가한다.
    - 필요시, 보강이나 수정이 이뤄질 수 있다.
    - 평가 기준예시 : "질문의 맥락을 충실히 반영했는가" , "정보가 명확하고 실용적인가","사용자 입력에 충분한 설명인가."
    ※ Self-RAG의 장점
    - 불필요한 검색방지 : 검색 리소스를 절약하고 처리 시간 단축.
    - 정보 기반성 강화 : 응답이 실제 문서에 근거하고 있는지를 평가함.
    - 유연한 처리 흐름 : 상황에 따라 검색 및 생성 전략이 달라짐.
    - hallucination 감소 : 지원 문서 기반 판단을 통해 사실 오류 방지.
    - 지속적 개선과 평가 가능 : 평가 피드백 기반 품질 개선 가능.
    ※ 결론
    Self-RAG는 기존 RAG의 단순 검색-생성 흐름을 넘어서, 각 단계를 능동적으로 판단하고 조정함으로써 더욱 신뢰도 높고
    유연한 응답을 생성하는 차세대 RAG 프레임워크이다. 정보가 넘쳐나는 시대에, 단순히 검색하는것에서 벗어나     
    "검색을 언제,왜 어떻게 할 것인가"를 스스로 판단하는 AI시스템으로의 진화를 의미한다.
    '''
    # AI 정보 문서 경로
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'

    # Self-RAG의 적응형 검색 전략을 테스트하기 위한 다양한 유형의 테스트 질의 정의
    test_queries = [
        "AI 개발에서 주요한 윤리적 문제는 무엇인가요?",  # 문서 기반 정보 질의
        # "설명 가능한 AI는 어떻게 신뢰를 향상시키나요?",  # 문서 기반 정보 질의
        # "인공지능에 관한 짧은 시를 써주세요",             # 창의적 질의 (검색 불필요)
        # "초지능 AI는 인간의 소외를 초래할까요?"           # 가설적 질의 (부분 검색 필요)
    ]

    # 보다 객관적인 평가를 위한 참조 정답
    reference_answers = [
        "AI 개발에서 주요한 윤리적 문제는 편향과 공정성, 프라이버시, 투명성, 책임성, 안전성, 악용 가능성 등입니다.",
        # "설명 가능한 AI는 의사결정 과정을 사용자에게 이해 가능하게 제공하여 공정성 검증, 편향 탐지, 시스템 신뢰 형성에 기여합니다.",
        # "양질의 인공지능 시는 AI의 가능성과 한계, 인간과의 관계, 미래 사회, 인식과 지능에 대한 철학적 탐구 등을 창의적으로 표현해야 합니다.",
        # "초지능 AI의 인간 소외에 대한 관점은 다양합니다. 일부는 경제적 대체나 통제 상실을 우려하며, 다른 일부는 보완적 역할과 인간 중심 설계로 여전히 인간이 중요하다고 봅니다. 대부분의 전문가는 안전하고 책임 있는 AI 설계가 핵심이라고 강조합니다."
    ]

    # Self-RAG과 전통 RAG 접근법을 비교 평가 실행
    evaluation_results = rc.ch13_evaluate_rag_approaches(
        pdf_path=pdf_path,                  # AI 정보를 담고 있는 문서 경로
        test_queries=test_queries,          # AI 관련 테스트 질의 리스트
        reference_answers=reference_answers # 평가용 기준 정답
    )

    # 최종 종합 비교 분석 결과 출력
    print("\n***전체 비교 분석 결과***\n")
    print(evaluation_results["overall_analysis"])    
#-----------------ch13 Self-RAG : end

#-----------------ch14 명제 청킹 : Proposition Chunking RAG : start
@router.get("/propositionchunking/{id}" , tags=['AI / Rag'],  description='Proposition Chunking RAG')
def propositionchunking( id:int = 0 ):
    '''Proposition Chunking RAG
    RAG 시스템에서 검색 정밀도를 극대화하기 위한 고급 청크 분할 기법으로, 문서를 단순히 일정 길이 또는 토큰 수로 자르는 대신,
    논리적 단위이자 의미적으로 완결된 개별 사실(명제) 단위로 분할하는 방법이다.
    기존ㄴ의 청킹 방식은 문단, 문장, 고정 길이 토큰 단위로 텍스트를 자르는 경우가 많았지만, 이는 종종 의미 단절 또는 불필요한
    정보 혼입을 초래해 검색 결과의 정확도를 떨어뜨리는 문제가 있었다.
    명제 청킹(Proposition Chunking)은 문서 내 정보를 원자적이고 독립적인 의미 단위로 나눔으로써,
    세분화된 정확한 검색과 정밀한 응답 생성이 가능해진다.
    ※ Proposition Chunking의 핵심 개념
    - 의미 단위 분할 : 문장을 단순히 문장부호로 나누는 것이 아니라, 각 문장내 개별 사실이냐 주장 단위로 분해.
    - 원자적 단위 생성 : 각 청크는 **하나의 명확한 정보 단위(명제)**만 포함 -> 검색 및 평가에 유리.
    - 중복 및 불완전 제안 필터링 : 의미가 불분명하거나 모호한 문장은 제거하거나 수정하여 품질 높은 검색 단위만 유지.
    ※ Proposition Chunking 단계별 절차
    1. 텍스트 입력 및 전처리
    PDF, HTML, 텍스트 파일 등에서 추출한 문서를 불용어 정리, 문단 분리 등 기본 전처리를 수행한다.
    2. 문장 분할
    텍스트를 문장 단위로 1차 분할 -> 예:구두점(.!?)기반 또는 언어 무델 기반 문장 추출한다.
    3. 명제 단위 분해
    각 문장을 내부의 주장(proposition) 단위로 분해한다.
    ex)
        - 원문 : "회사는 이용자의 개인정보를 수집하며, 동의 없이 제 3자에게 제공하지 않습니다."
        - 분해 결과
            => [1] 회사는 이용자의 개인정본를 수집한다.
            => [2] 회사는 개인정보를 동의 없이 제3자에게 제공하지 않는다.
        - 방법
            => 규칙 기반 분해(접속사 분리, 주어-동사 구조 파악)
            => LLM활용 : 다음 문장을 의미 단위의 명제들로 분해해 주세요.
    4. 의미 무결성 검사 및 필터링
    - 각 명제가 독립적으로 의미를 지니는지, 문맥 없이도 해석 가능한지 검토한다.
    - 불완전한 문장은 제거하거나 정제한다.
    5. 임베딩및 저장
    분해된 명제 단위 청크를 임베딩 모델로 벡터화하여 벡터 저장소(vectorDB)에 저장한다.
    6. 질의 매칭 및 응답 생성
    - 사용자 쿼리에 대해 의미적으로 가장 관련성 높은 명제 청크 단위로 정밀 검색을 수행한다.
    - 선택된 명제들은 LLM입력으로 사용되어 짧지만 핵심적인 정보를 기반으로 응답 생성이 가능하다.
    ※ Proposition Chunking의 장점
    - 정밀도 향상 : 검색 단위가 작고 명확하여 불필요한 문맥 혼입 최소화
    - 매칭 효율 증대 : 의미 유사도 기준의 더 세밀한 검색 결과 제공
    - Hallucination 감소 : 불완전하거나 다의적인 정보 제거로 응답 왜곡 방지
    - 멀티 쿼리 대응 : 복합 쿼리에 대해 명제 단위로 응답 조합 가능
    ※ 사용예시
    - 문장 : "회사는 이용자의 정보를 수집하고, 통계 처리를 위해 내부적으로 분석합니다."
    - 명제 청킹 결과
        => [1] 회사는 이용자의 정보를 수집한다.
        => [2] 회사는 수집한 정보를 통계 처리를 위해 내부적으로 분석한다.
    - 사용자가 "개인정보 분석 목적이 뭐야?"라고 질문할 경우 -> [2]만정확히 매칭됨.
    ※ 결론
    Proposition Chunking은 기존 RAG의 청크 분할 방식 보다 한층 더 정교한 접근으로, 정보를 의미 중심으로 
    세분화하여 검색의 정밀도와 응답의 신뢰도를 동시에 높인다. 특히 규제 문서, 계약서, 법률문헌, 정책 보고서와
    같이 정보 단위가 명확한 문서에서 탁월한 성능을 발휘하며, 고품질 RAG시스템의 필수 요소로 주목받고 있다.
    
    ※ PS SWI 너무 오래 걸리며 중간에 오류가 발생했다. 추후 점검하자 1회 수행시 약 30분 정도 소요된다.
    '''
    # 처리할 AI 정보 문서의 경로
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'

    # AI의 다양한 측면을 평가하기 위한 테스트 쿼리 정의 (현재는 1개 사용)
    test_queries = [
        "AI 개발의 주요 윤리적 문제는 무엇인가요?",
    ]

    # 명제 기반 vs 청크 기반 응답의 정확도 비교를 위한 기준 정답 (Reference Answers)
    reference_answers = [
        "AI 개발의 주요 윤리적 문제에는 편견과 공정성, 개인정보 보호, 투명성, 책임성, 안전, 오용 또는 유해한 애플리케이션의 가능성 등이 있습니다.",
    ]

    # 평가 실행 (엔드 투 엔드: 문서 처리 → 명제 생성 → 벡터 저장소 → 검색 → 응답 생성 → 평가)
    evaluation_results = rc.ch14_run_proposition_chunking_evaluation(
        pdf_path=pdf_path,
        test_queries=test_queries,
        reference_answers=reference_answers
    )

    # 전체 분석 결과 출력
    print("\n\n***Overall Analysis***")
    print(evaluation_results["overall_analysis"])    
#-----------------ch14 명제 청킹 : Proposition Chunking RAG : end

#----------------- ollama 를 이용한 온프라미스 환경에서의 RAG : start
@router.post("/openai/" , tags=['AI / Rag / simple'],  description='심플 RAG')
def simple( query:str='' ):
    print('openai','-'*10,'start')
    print(query)
    print('openai','-'*10,'end')
    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'
    #pdf_url     = "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf"
    

    # 텍스트 다시 추출
    text = rc.ch1_extract_text_from_pdf( pdf_path )

    # chunks 생성
    chunks = rc.ch1_chunk_text(text , 1000,200)

    # 임베딩 생성
    embedding = rc.ch1_create_embeddings(chunks)

    # 의미 기반 검색 수행 : 주어진 쿼리에 대해 가장 관련성 높은 텍스트 청크 2개
    top_chunks = rc.ch1_semantic_search(query , chunks , embedding.data , 2 )

    #검색된 상위 문맥(top_chunks)을 기반으로 사용자 프롬프트 구성
    user_prompt = '\n'.join([
        f'Context {i+1}:\n{chunk}\n---------\n'
        for i ,chunk in enumerate(top_chunks)
    ])
    user_prompt = f"{user_prompt}\n Question : {query}"

    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 문맥을 기반으로만 답변하는 AI어시턴트 입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )

    # AI응답 생성 
    ai_res = rc.ch1_generate_response(system_prompt , user_prompt)
  
    return {
            'question': query ,
            'answer':ai_res.choices[0].message.content ,
            }



@router.post("/ollamarag/" , tags=['AI / Rag / simple'],  description='ollama 를 이용한 온프라미스 환경에서의 RAG')
def ollamarag(query:str=''):
    ''' 
    ※ 온프라미스 LLM 기반 Simple RAG 구성 개요
    본 테스트는 외부 LLM 서비스(ChatGPT 등)에 전혀 의존하지 않고,
    온프라미스 환경에서 LLM 기반 RAG 시스템이 독립적으로 동작 가능한지 여부를 검증하는 것을 목적으로 한다.

    ※ 테스트 목적 및 정의
    - 외부 API 호출 없이 로컬/사내 환경에서 완결된 RAG 파이프라인 구성 가능 여부 확인
    - 생성 모델, 임베딩, 검색, 응답까지의 전체 흐름이 온프라미스 환경에서 안정적으로 동작하는지 검증
    - 노트북 단일 환경 기준에서 성능과 응답 품질의 현실적인 균형점 탐색

    ※ 적용 기술 스택
    1. Ollama (온프라미스 LLM 서버)
    - 언어 모델로 Gemma2:2b 사용
    - 모델 선택 근거
        => 8b 모델: 노트북 환경에서 실행 불가 또는 응답 지연 과도
        => 4b 모델: 실행은 가능하나 출력 속도가 느림
        => 1b 모델: 응답 속도는 빠르나 출력 품질이 현저히 낮음
    - Gemma2:2b는 성능·속도·하드웨어 제약을 종합적으로 고려했을 때 가장 현실적인 선택
    - 외부 네트워크 연결 없이 완전 온프라미스 구동 가능

    2. FastAPI (백엔드)
    - Python 기반의 경량 웹 백엔드 프레임워크
    - 역할
        => 사용자 질의 수신
        => 임베딩 및 문서 검색 처리
        => Ollama LLM과 연동하여 답변 생성
        => 프론트엔드에 API 형태로 결과 제공
        => 단순 RAG 구조 구현 및 테스트에 적합

    3. Svelte (프론트엔드)
    - 경량 프론트엔드 프레임워크
    - 목적
        => 사용자 질의 입력
        => RAG 응답 결과 시각화
        => 테스트 목적상 복잡한 UI보다는 빠른 개발과 가벼운 실행 환경에 초점    

    ※ 정리 요약
    - 본 구성은 온프라미스 LLM 기반 RAG의 최소 실행 가능 구조 검증을 목표로 한다.
    - Gemma2:2b 모델을 중심으로, 노트북 단일 환경에서도 실행 가능한 현실적인 구성을 채택하였다.
    - 본 테스트 결과를 통해 향후
        => 모델 확장(4b 이상)
        => FAISS 적용
        => 운영 환경 분리
        => 여부를 판단할 수 있는 기초 데이터를 확보한다.

    ※ RAG 처리 흐름
        [사용자 질문]
                |            
        ================= Retrieval =================
                |
        [질문 임베딩]
                |
        [문서 청킹,임베딩 검색]
                |
        [Top-K 컨텍스트 선택]
                |
        ================ Generation =================
                |
        [프롬프트 생성]
                |
        [LLM 답변 생성]
                |
        [응답 반환]    
    
    ※ RAG 처리 기법
    - Simple RAG : 질의에 대해서 단순히 문서를 청크하여 처리.
    - Semantic Chunking : 문서의 청크할시 청크의 간의 유사도를 기준으로 분류 후 처리.
    - Context Enriched : 청크 문서 탐색시 해당 청크의 문서 앞뒤 자료를 첨부해서 처리.
    - Contextual Chunk Headers : 청크문서 생성시 청크문서의 문단의 제목등을 첨부하여 처리.
    - Document Augmentation : 청크문서 검색시 기존에 유사도에 의존하는것을 강화하여 질문을 먼저 LLM을통해서 질문을 강화 하여 검색한다.
    '''

    # 항목에서 질의 추출
    query_embedding = rc.ollama_embedding( query )

    pdf_path    = r'D:\python_workspace\FastApi\Area\Rag\AI_Understanding.pdf'
    
    # 텍스트 다시 추출
    text = rc.ollama_extract_text_from_pdf( pdf_path )
    #print(text)

    # chunks 생성
    chunks = rc.ollama_chunk_text(text , 500,200)
    
    # 임베딩 생성
    embedding = rc.ollama_create_embeddings(chunks)    


    # 의미 기반 검색 수행 : 주어진 쿼리에 대해 가장 관련성 높은 텍스트 청크 2개
    top_chunks = rc.ollama_semantic_search(query_embedding , chunks , embedding , 5 )
    user_prompt = '\n'.join([
        f'Context {i+1}:\n{chunk}\n---------\n'
        for i ,chunk in enumerate(top_chunks)
    ])    
    user_prompt = f"{user_prompt}\n Question : {query}"

    # ai 어시턴트를 위한 시스템 프롬프트 정의
    system_prompt = (
        "당신은 주어진 문맥을 기반으로만ㄴ 답변하는 AI어시턴트 입니다."
        "제공된 문맥에서 직접적으로 답을 도출할 수 없는 경우에는 다음과 같이 답하십시오 : "
        "'I do not have enough information to answer that'"
    )    

    # AI응답 생성 
    ai_res = rc.ollama_generate_response(system_prompt , user_prompt)    
    '''
    # 평가 시스템을 위한 시스템 프롬프트 정의
    evaluate_system_prompt = (
        "당신은 AI 어시스턴트의 응답을 평가하는 지능형 평가 시스템입니다. "
        "AI 응답이 정답에 매우 근접하면 점수 1점을 부여하고, "
        "정답과 맞지 않거나 불만족스러우면 0점을 부여하세요. "
        "부분적으로 일치하면 0.5점을 부여하세요."
    )

    # 사용자 질의, AI 응답, 정답, 평가 프롬프트를 결합하여 평가용 프롬프트 생성
    evaluation_prompt = (
        f"User Query: {query}\n"
        f"AI Response:\n{ai_res['message']['content']}\n"
        f"True Response: {true_answer}\n"
        f"{evaluate_system_prompt}"
    ) 
    '''

    # 평가 시스템 프롬프트를 사용하여 AI 응답 평가 점수 생성
    #evaluation_response = rc.ollama_generate_response(evaluate_system_prompt, evaluation_prompt)
    return {
            'question': query ,
            'answer':ai_res['message']['content'] ,
        } 

#----------------- ollama 를 이용한 온프라미스 환경에서의 RAG : end
