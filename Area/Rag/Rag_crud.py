from utils      import utils

# 확장 라리브러리
import openai
import base64
import numpy as np
import json
import fitz
import requests
import re
import os
from tqdm import tqdm
from datetime import datetime


util = utils()

class SimpleVectorStore:
    """
    NumPy를 사용한 간단한 벡터 저장소 구현 클래스입니다.
    """
    def __init__(self):
        """
        벡터 저장소를 초기화합니다.
        """
        self.vectors    = []    # 임베딩 벡터들을 저장
        self.texts      = []    # 원본 텍스트들을 저장
        self.metadata   = []    # 텍스트에 대한 메타데이터 저장
    
    def add_item(self, text, embedding, metadata=None):
        """
        벡터 저장소에 항목을 추가합니다.

        Args:
            text (str): 원본 텍스트.
            embedding (List[float]): 임베딩 벡터.
            metadata (dict, optional): 추가 메타데이터 (기본값: None).
        """
        self.vectors.append(np.array(embedding))             # 벡터 추가
        self.texts.append(text)                              # 텍스트 추가
        self.metadata.append(metadata or {})                 # 메타데이터 추가
    
    def similarity_search(self, query_embedding, k=5):
        """
        쿼리 임베딩과 가장 유사한 항목을 검색합니다.

        Args:
            query_embedding (List[float]): 쿼리 벡터.
            k (int): 반환할 결과 수 (기본값: 5).

        Returns:
            List[Dict]: 상위 k개의 유사 항목. 텍스트, 메타데이터, 유사도 포함.
        """
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        similarities = []

        # 각 벡터와의 코사인 유사도 계산
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # 유사도 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        
        return results


#----------------- ch1 simpleRAG : begin
def ch1_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text

def ch1_chunk_text(text , n , overlap):
    '''
    주어진 텍스트를 n 자 단위로, 지정된 overlap만큼 겹치도록 분할합니다.
    
    :param text: 분할할 원본 텍스트
    :param n: 각 청크의 문자수
    :param overlap: 청크간 겹치는 문자수.
    '''
    chunks = [] # 청크를 저장할 빈리스트 초기화
    for i in range(0 , len(text) , n - overlap):
        chunks.append(text[i:i + n ])
    return chunks

def ch1_create_embeddings( text , model='text-embedding-3-small'):
    ''' 주어진 텍스트에 대한 지정된 모델을 사용하여 임베딩을 생성하는 함수
    text : 임베딩을 생성할 입력 텍스트
    model : 사용할 임베딩 모델(기본값 BAAI/bge0-en-icl)
    return 
        dict : openAI API로 부터 받은 임베딩 응답 결과.
    '''
    # 지정된 모델을 사용하여 텍스트 임베딩 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(
        model = model ,
        input = text ,
    )
    return res

def ch1_cosine_similarity(vec1 , vec2):
    ''' 코사인 유사도를 구현하여 사용자 쿼리에 가장 관련성 높은 텍스트 청크를 찾는다.
    vec1 : 첫번째 벡터
    vec2 : 두번째 벡터
    return :
        float :  두벡터 간의 코사인 유사도(값의 범위 -1 ~ 1)
    '''
    return np.dot(vec1 , vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) )

def ch1_semantic_search( query , text_chunks , embeddings , k=5 ):
    '''
    주어진 쿼리와 임베딩을 사용하여 텍스트 청크에서 의미 기반 검색을 수행합니다.
    
    :param query(str): 의미 검색에 사용할 쿼리 텍스트
    :param text_chunks(List[str]): 검색 대상이 되는 텍스트 청크 리스트
    :param embeddings(List[dict]): 각 청크에 대한 임베딩 객체 리스트
    :param k: 상위 k개의 관련 텍스트 청크를 반환
    return :
        List[str] : 쿼리와 가장 관련 있는 텍스트 청크 상위 k개
    '''
    # 쿼리에 대한 임베딩 생성
    query_embedding     = ch1_create_embeddings(query).data[0].embedding
    similarity_scores   = [] # 유사도 점수를 저장할 리스트 초기화

    #각 텍스트 청크의 임베딩과 쿼리 임베딩 간의 코사인 유사도 계산.
    for i, chunk_embedding in enumerate(embeddings):

        similarity_score = ch1_cosine_similarity(
                np.array(query_embedding) , 
                np.array(chunk_embedding.embedding) ,
                )
        similarity_scores.append( (i , similarity_score) ) # 인덱스와 유사도 함께 저장


    # 유사도 점수를 기준으로 내림차순 정렬
    similarity_scores.sort(key=lambda x : x[1] , reverse=True)

    # 상위 k개의 청크 인덱스를 추출
    top_indices = [index for index , _ in similarity_scores[:k]]

    # 상위 k개의 관련 텍스트 청크 반환
    return [text_chunks[index] for index in top_indices]
        
def ch1_RunningaQueryOnExtractedChunks():
    '''
    위 사항을 검증 한다.
    '''
    path = r"D:\python_workspace\FastApi\Area\Rag\validation.json"
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    
    # 첫번째 항목에서 질의 추출
    query = data[0]['question']

    # 텍스트 다시 추출
    text = ch1_extract_text_from_pdf( "https://raw.githubusercontent.com/no-wave/llm-master-rag-techniques/main/dataset/AI_Understanding.pdf" )

    # chunks 생성
    chunks = ch1_chunk_text(text , 1000,200)

    # 임베딩 생성
    embedding = ch1_create_embeddings(chunks)

    # 의미 기반 검색 수행 : 주어진 쿼리에 대해 가장 관련성 높은 텍스트 청크 2개
    top_chunks = ch1_semantic_search(query , chunks , embedding.data , 2 )

    #질의 출력
    print('질의 query',query)

    # 관련성 높은 문맥 청크 2개 출력
    for i ,chunk in enumerate(top_chunks):
        print(f'문맥{i +1} \n {chunk} -----' )
    

def ch1_generate_response(system_prompt , user_message , model='gpt-4o-mini'):
    '''
    시스템 프롬프트와 사용자 메시지를 기반으로 AI모델의 응답을 생성합니다.
    
    :param system_prompt(str) : AI의 응답 방식을 지정하는 시스템 메세지
    :param user_message(str): 사용자 질의 또는 메시지
    :param model(str): 사용할 언어 모델 이름
    return :
        dict 생성된 AI응답을 포함한 API응답 객체    
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.chat.completions.create(
        model = model , 
        temperature = 0 ,
        messages = [
            {'role':'system' , 'content':system_prompt} ,
            {'role':'user' , 'content':user_message} ,
        ]
     )
    return res

#----------------- ch1 simpleRAG : end
#----------------- SemanticChunking : start
def extract_text_from_pdf2(pdf_path):
    '''
    pdf 파일에서 텍스트를 추출하는 함수.       
    :pdf_path:  pdf http url 경로
    '''      
    mypdf       = fitz.open(pdf_path)
    all_text    = ''
    for page in mypdf:
        all_text += page.get_text('text')+' '
    
    return all_text.strip() #텍스트의 앞뒤 공백을 제거한 텍스트 반환.

def get_embedding(text , model='text-embedding-3-small'):
    '''
    OpenAi 클라이언트를 사용하여 입력된 텍스트의 임베딩을 생성합니다.
    
    :param text(str): 임베딩을 생성할 입력 텍스트
    :param model(str): 사용할 임베딩 모델이름(기본값 : BAAI/bge-en-icl)
    returns:
        np.ndarray:생성된 임베딩 벡터
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(
        model = model ,
        input = text ,
    )
    return np.array( res.data[0].embedding )

def compute_breakpoints(similarities , method='percentile' , threshold=90):
    '''
    유사도 하락에 기반하여 청킹 분할점을 계산.
    
    :param similarities(List[float]): 문장간의 유사도 점수 리스트
    :param method(str): percentile , standard_deviation , interquartile
    :param threshold(float) : 임계값(퍼센트일경우 퍼센터, 표준편차일경우는 표준편차 배수)
    returns:
        List[int] : 청킹 분할이 발생해야 하는 인덱스 목록
    '''
    # 선택된 방법에 따라 임계값을 결정합니다.
    if method == 'percentile':
        # 유사도 점수의 X번째 퍼센트을 계산합니다.
        threshold_value = np.percentile( similarities , threshold )
    elif method =='standard_deviation':
        # 유사도 점수의 평균과 표준편차를 계산합니다.
        mean    = np.mean(similarities)
        std_dev = np.std(similarities)
        # 평균에서 x 표준편차를 뺀 값으로 임계값을 설정.
        threshold_value = mean - (threshold * std_dev)
    elif method =='interquartile':
        #첫 번째와 세번째 사분위(Q1 및 Q3)를 계산한다.
        q1 , q3 = np.percentile(similarities[25,75])
        # IQR 규칙을 이요해 이상치 기준 임계값을 설정.
        threshold_value = q1 - 1.5 *(q3-q1)
    else:
        # 유효하지 않은 방법이 제공된 경우 에러 발생.
        raise ValueError('유효하지 않습니다. percentile , standard_deviation , interquartile 만 가능')
    
    # 유사도가 임계값보다 낮은 인덱스를 식별.
    rt = [i for i , sim in enumerate(similarities) if sim < threshold_value]

    return rt

def split_into_chunks( sentences , breakpoints ):
    '''
    문장들을 의미 단위로 분할.
    
    :param sentences(List[str]): 문장리스트
    :param breakpoints(List[int]): 청킹(분할) 발생해야 하는 인덱스 목록.
    returns:
        List[str] : 텍스트 청크 리스트
    '''
    chunks  = [] # 청크를 저장할 빈 리스트 초기화.
    start   = 0 # 시작 인덱스 초기화.
    
    # 각 분할점을 순회하여 청크를 생성.
    for bp in breakpoints:
        # 시작 인덱스부터 현재 분할점까지의 문장들을 연결하여 청크를 생성, 청크리스트 추가.
        chunks.append('. '.join(sentences[start:bp+1])+'.'  )
        start = bp+1 # 시작 인덱스를 업데이트하여 다음 문장 부터 청크를 생성.

    # 남은 문장들을 마지막 청크로 추가.
    chunks.append( '. '.join( sentences[start:] ) )
    return chunks # 청크 리스트를 반환.

def create_embeddings2(text_chunks):
    '''
    각 텍스트 청크에 대한 임베딩을 생성합니다.
    
    :param text_chunks(List[str]): 텍스트 청크 리스트
    return :
        List[np.ndarray] : 임베딩 벡터 리스트.
    '''
    # get_embedding 함수를 사용하여 각 텍스트 청크에 대한 임베딩을 생성.
    rt = [ get_embedding(chunk) for chunk in text_chunks ]
    return rt

def semantic_search2(query , text_chunks , chunk_embeddings , k=5):
    '''
    쿼리에 가장 관련성 높은 텍스트 청크들을 찾습니다.
    
    :param query(str): 검색 쿼리
    :param text_chunks(List[str]): 텍스트 청크 리스트
    :param chunk_embeddings(List[np.ndarray]): 청크 임베딩 리스트
    :param k(int): 반환할 결과의 수.
    returns:
        List[str] : 상위 k개의 관련성 높은 텍스트 청크 리스트.
    '''
    # 쿼리에 대한 임베딩을 생성.
    query_embedding = get_embedding(query)

    # 쿼리 임베딩과 각 청크 임베딩 간의 코사인 유사도를 계산.
    similarities = [ ch1_cosine_similarity(query_embedding , emb)  for emb in chunk_embeddings]

    # 유사도가 높은 순으로 상위 k개의 인덱스를 가져옴.
    top_indices = np.argsort( similarities )[-k:][::-1]

    # 상위k개의 관련성 높은 텍스트 청크들을 반환.
    rt = [text_chunks[i] for i in top_indices]
    return rt
#----------------- SemanticChunking : end

#----------------- Chunk Sizes Rag : start
def ch3_extract_text_from_pdf(pdf_path):
    '''
    pdf 파일에서 텍스트를 추출하는 함수.       
    :pdf_path:  pdf http url 경로
    '''      
    mypdf       = fitz.open(pdf_path)
    all_text    = ''
    for page in mypdf:
        all_text += page.get_text('text')+' '
    
    return all_text.strip() #텍스트의 앞뒤 공백을 제거한 텍스트 반환.

def ch3_create_embeddings( text , model='text-embedding-3-small'):
    ''' 주어진 텍스트에 대한 지정된 모델을 사용하여 임베딩을 생성하는 함수
    text : 임베딩을 생성할 입력 텍스트
    model : 사용할 임베딩 모델(기본값 BAAI/bge0-en-icl)
    return 
        List[np.ndarray] : 생성된 임베딩 벡터
    '''
    # 지정된 모델을 사용하여 텍스트 임베딩 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(        model = model ,         input = text ,    )
    rt = [ np.array(embedding.embedding) for embedding in res.data]
    return rt

def ch3_cosine_similarity(vec1 , vec2):
    ''' 두 벡터 간의 코사인 유사도 계산
    vec1 : 첫번째 벡터
    vec2 : 두번째 벡터
    return :
        float :  두벡터 간의 코사인 유사도(값의 범위 -1 ~ 1)
    '''
    return np.dot(vec1 , vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) )

def ch3_retrieve_relevant_chunks(query , text_chunks , chunk_embeddings , k=5):
    '''
    가장 관련성 높은 사우이 k개의 텍스트 청크 검색.
    
    :param query(str): 사용자 쿼리
    :param text_chunks(List[str]): 텍스트 청크 리스트
    :param chunk_embeddings(List[np.ndarray]): 텍스트 청크들의 임베딩
    :param k(int): 반환할 상위 청크 개수
    returns :
        List[str] : 관련성 높은 텍스트 청크 리스트
    '''
    # 쿼리에 대한 임베딩을 생성합니다.
    query_embedding = ch3_create_embeddings([query])[0]

    # 쿼리 임베딩과 각 청크 임베딩 간의 코사인 유사도를 계산.
    similarities = [ ch3_cosine_similarity(query_embedding , emb) for emb in chunk_embeddings ]

    # 유사도가 높은 순서대로 상위 K개의 인덱스를 가져온다.
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 상위 k개의 텍스트 청크를 반환
    rt = [text_chunks[i] for i in top_indices]
    return rt

def ch3_generate_response(query , system_prompt , retrieved_chunks , model='gpt-4o-mini'):
    '''
    검색된 청크를 기반으로 ai응답을 생성
    
    :param query(str): 사용자쿼리
    :param system_prompt(str) : AI의 응답 방식을 지정하는 시스템 메세지
    :param retrieved_chunks(List[str]): 검색된 텍스트 청크 리스트 
    :param model(str): 설명
        return : str ai가 생성된 응답 문자열.
    '''
    # 검색된 청크들을 하나의 컨텍스트 문자열로 결합.
    context = '\n'.join( [f"컨텍스트 {i+1}:\n{chunk}" for i ,chunk in enumerate(retrieved_chunks)] )

    # 컨텍스트와 쿼리를 조합하여 사용자 프롬프트를 생성.
    user_prompt = f"{context}\n\n질문:{query}"

    # 지정된 모델을 사용하여 AI응답 생성.
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.chat.completions.create(
        model = model , 
        temperature = 0 ,
        messages = [
            {'role':'system' , 'content':system_prompt} ,
            {'role':'user' , 'content':user_prompt} ,
        ]
    )
    # 응답의 내용반환.
    return res.choices[0].message.content
    
def ch3_evaluate_response(question, response, true_answer):
    """
    AI가 생성한 응답의 품질을 신뢰성과 관련성 기준으로 평가합니다.

    Args:
        question (str): 사용자 질문.
        response (str): AI가 생성한 응답.
        true_answer (str): 정답 (기준 정답).

    Returns:
        Tuple[float, float]: (신뢰성 점수, 관련성 점수) 튜플.
                             각 점수는 1.0 (완전 일치), 0.5 (부분 일치), 0.0 (불일치) 중 하나입니다.
    """

    # 평가 점수 시스템에 사용할 상수를 정의합니다.
    SCORE_FULL = 1.0     # 완전 일치 또는 매우 만족스러운 응답
    SCORE_PARTIAL = 0.5  # 부분 일치 또는 다소 만족스러운 응답
    SCORE_NONE = 0.0     # 불일치 또는 만족스럽지 않은 응답

    # 신뢰성 평가를 위한 엄격한 프롬프트 템플릿을 정의합니다.
    FAITHFULNESS_PROMPT_TEMPLATE = """
    AI 응답이 정답에 비해 얼마나 신뢰성 있게 일치하는지를 평가하세요.
    사용자 질문: {question}
    AI 응답: {response}
    정답: {true_answer}

    신뢰성(Faithfulness)은 AI의 응답이 정답의 사실과 얼마나 일치하며, 환각(hallucination) 없이 사실에 기반하고 있는지를 평가합니다.

    지침:
    - 다음 점수 값만 사용하여 엄격하게 평가하십시오:
        * {full} = 완전히 신뢰 가능함, 정답과 모순 없음
        * {partial} = 부분적으로 신뢰 가능함, 약간의 모순 존재
        * {none} = 신뢰할 수 없음, 명백한 모순 또는 환각 포함
    - 설명이나 부가 텍스트 없이 숫자 점수({full}, {partial}, 또는 {none})만 반환하세요.
    """
    # 관련성 평가를 위한 엄격한 프롬프트 템플릿을 정의합니다.
    RELEVANCY_PROMPT_TEMPLATE = """
    AI 응답이 사용자 질문과 얼마나 관련성이 있는지를 평가하세요.
    사용자 질문: {question}
    AI 응답: {response}

    관련성(Relevancy)은 응답이 사용자 질문에 얼마나 잘 대응하는지를 측정합니다.

    지침:
    - 다음 점수 값만 사용하여 엄격하게 평가하십시오:
        * {full} = 완전히 관련 있음, 질문에 직접적으로 응답함
        * {partial} = 부분적으로 관련 있음, 질문의 일부만 응답함
        * {none} = 관련 없음, 질문을 제대로 다루지 못함
    - 설명이나 부가 텍스트 없이 숫자 점수({full}, {partial}, 또는 {none})만 반환하세요.
    """    


    # 평가 프롬프트를 구성합니다.
    faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        true_answer=true_answer,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )

    relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )

    # 신뢰성 평가 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    faithfulness_response = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": faithfulness_prompt}
        ]
    )

    # 관련성 평가 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    relevancy_response = _client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an objective evaluator. Return ONLY the numerical score."},
            {"role": "user", "content": relevancy_prompt}
        ]
    )

    # 점수를 파싱하고 오류 발생 시 0.0으로 대체합니다.
    try:
        faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
    except ValueError:
        print("Warning: Could not parse faithfulness score, defaulting to 0")
        faithfulness_score = 0.0

    try:
        relevancy_score = float(relevancy_response.choices[0].message.content.strip())
    except ValueError:
        print("Warning: Could not parse relevancy score, defaulting to 0")
        relevancy_score = 0.0

    return faithfulness_score, relevancy_score
#----------------- Chunk Sizes Rag : end

#----------------- Context-Enriched Rag : start
def ch4_extract_text_from_pdf(pdf_path):
    '''
    pdf 파일에서 텍스트를 추출하는 함수.       
    :pdf_path:  pdf http url 경로
    '''      
    mypdf       = fitz.open(pdf_path)
    all_text    = ''
    # pdf 의 각 페이지를 순회 하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page        = mypdf[page_num] # 페이지 객체를 가져옴.
        text        = page.get_text('text') # 해당 페이지에서 텍스트 추출
        all_text    += text # 추출된 텍스트 누적.
    return all_text

def ch4_chunk_text(text , n , overlap):
    '''
    주어진 텍스트를 n 자 단위로, 지정된 overlap만큼 겹치도록 분할합니다.
    
    :param text: 분할할 원본 텍스트
    :param n: 각 청크의 문자수
    :param overlap: 청크간 겹치는 문자수.
    '''
    chunks = [] # 청크를 저장할 빈리스트 초기화
    for i in range(0 , len(text) , n - overlap):
        chunks.append(text[i:i + n ])
    return chunks

def ch4_create_embeddings( text , model='text-embedding-3-small'):
    ''' 주어진 텍스트에 대한 지정된 모델을 사용하여 임베딩을 생성하는 함수
    text : 임베딩을 생성할 입력 텍스트
    model : 사용할 임베딩 모델(기본값 BAAI/bge0-en-icl)
    return 
        List[np.ndarray] : 생성된 임베딩 벡터
    '''
    # 지정된 모델을 사용하여 텍스트 임베딩 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(        model = model ,         input = text ,    )
    #rt = [ np.array(embedding.embedding) for embedding in res.data]
    return res

def ch4_cosine_similarity(vec1 , vec2):
    ''' 두 벡터 간의 코사인 유사도 계산
    vec1 : 첫번째 벡터
    vec2 : 두번째 벡터
    return :
        float :  두벡터 간의 코사인 유사도(값의 범위 -1 ~ 1)
    '''
    return np.dot(vec1 , vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) )

def ch4_context_enriched_search(query , text_chunks , embeddings , k=1 , context_size=1):
    '''
    가장 관련성 높은 청크와 그 주변 청크들을 함께 검색합니다.    
    :param query(str): 검색 쿼리
    :param text_chunks(List[str]: 텍스트 청크 리스트
    :param embeddings(List[dict]): 텍스트 청크에 대한 임베딩 리스트
    :param k(int): 검색할 관련 청크 개수(현재는 상위 1개만)
    :param context_size(int): 관련 청크 주변에 포함할 청크 수.
    returns : List[str] 관련성 높은 청크및 문맥 정보를 포함한 텍스트 청크 리스트
    '''
    # 쿼리 임베딩 벡터로 변환.
    query_embedding     = ch4_create_embeddings(query).data[0].embedding
    similarity_scores   = []

    # 각 청크 임베딩과 쿼리 임베딩 간의 유사도 점수 계계.
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = ch4_cosine_similarity(
            np.array( query_embedding ) , 
            np.array( chunk_embedding.embedding )
        )
        similarity_scores.append((i , similarity_score)) #(인텍스 , 유사도)
    
    # 유사도 점수를 기준으로 내림차순 정렬
    similarity_scores.sort(key=lambda x: x[1] , reverse=True)

    # 가장 관련성 높은 청크의 인덱스를 가져옴.
    top_index = similarity_scores[0][0]
    
    #문맥 포함 범위를 정의
    start   = max(0,top_index - context_size)
    end     = min(len(text_chunks) , top_index+context_size+1)

    # 해당 범위의 청크들을 반환.
    rt = [text_chunks[i] for i in range(start , end)]
    return rt

def ch4_generate_response(system_prompt:str , user_message:str , model:str='gpt-4o-mini'):
    '''
    시스템 프롬프트와 사용자 메시지를 기반으로 AI모델의 응답을 생성.
    
    :param system_prompt(str): 설명
    :param user_message(str): 설명
    :param model(str): 설명
    returns(str) : AI모델의 응답 객체.
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.chat.completions.create(
        model = model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response
#----------------- Context-Enriched Rag : end

#----------------- Contextual Chunk Headers : start
def ch5_extract_text_from_pdf(pdf_path):
    '''
    pdf 파일에서 텍스트를 추출하는 함수.       
    :pdf_path:  pdf http url 경로
    '''      
    mypdf       = fitz.open(pdf_path)
    all_text    = ''
    # pdf 의 각 페이지를 순회 하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page        = mypdf[page_num] # 페이지 객체를 가져옴.
        text        = page.get_text('text') # 해당 페이지에서 텍스트 추출
        all_text    += text # 추출된 텍스트 누적.
    return all_text

def ch5_generate_chunk_header(chunk , model='gpt-4o-mini'):
    '''
    LLM을 사용하여 지정된 텍스트 청크의 제목/헤더를 생성합니다.
    
    :param chunk(str):헤더로 요약할 텍스트 청크 
    :param model(str): 헤더 생성하는데 사용할 모델.   
    return (str)   : 생성된 헤더/제목.    
    '''
    # AI의 동작을 안내하는 시스템 프롬프트 정의하기
    system_prompt = 'Generate a concise and informative title for the given text'

    # 시스템 프롬프트 및 텍스트 청크를 기반으로 AI모델에서 응답을 생성.
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.chat.completions.create(
        model = model,
        temperature=0,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}            
        ])
    # 생성된 헤더/제목을 반환하고 선행/후행 strip 함수로 공백을 제거.
    return res.choices[0].message.content.strip()
    
def ch5_chunk_text_with_headers(text , n , overlap):
    '''
    청크 텍스트를 더 작은 세그멘트로 나누고 헤더를 생성.
    
    :param text(str): 청크할 전체 텍스트
    :param n(int): 문자 단위의 청크 크기
    :param overlap(int): 청크 사이의 겹치는 문자수
    return (List[dict]) : header 와 text 키를 가진 딕셔너리 리스트.
    '''
    chunks = [] # 청크를 저장하기 위해 공백 리스트 지정

    # 지정된 청크 크기와 겹침으로 텍스트를 반복.
    for i in range(0 , len(text) , n - overlap):
        chunk = text[i:i+n] # 텍스트 청크 추출
        header = ch5_generate_chunk_header( chunk ) # LLM을 사용하여 청크의 헤더를 생성.
        chunks.append( {'header':header , 'text':chunk} ) #목록에 헤더와 청크추가.
    
    return chunks # 헤드가 있는 청크 목록 반환.

def ch5_create_embeddings(text , model='text-embedding-3-small'):
    '''
    주어진 텍스트에 대한 임베딩을 생성
    
    :param text(str): 임베드할 입력 텍스트
    :param model(str): 사용한 임베딩 모델.
    returns (dict) : 입력텍스트에 대한 임베딩이 포함된 응답.
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(        model = model ,         input = text ,    )
    #rt = [ np.array(embedding.embedding) for embedding in res.data]
    return res.data[0].embedding

def ch5_cosine_similarity(vec1 , vec2):
    ''' 두 벡터 간의 코사인 유사도 계산
    vec1 : 첫번째 벡터
    vec2 : 두번째 벡터
    return :
        float :  두벡터 간의 코사인 유사도(값의 범위 -1 ~ 1)
    '''
    return np.dot(vec1 , vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) )

def ch5_semantic_search( query , chunks , k=5 ):
    '''
    쿼리를 기반으로 가장 관련성 높은 청크를 검색.
    
    :param query(str): 의미 검색에 사용할 쿼리 텍스트
    :param chunks(List[str]): 검색 대상이 되는 텍스트 청크 리스트
    :param k: 상위 k개의 관련 텍스트 청크를 반환
    return :
        List[dict] : 쿼리와 가장 관련 있는 텍스트 청크 상위 k개
    '''
    # 쿼리에 대한 임베딩 생성
    query_embedding     = ch5_create_embeddings(query)
    similarities        = [] # 유사도 점수를 저장할 리스트 초기화

    #각 텍스트 청크의 임베딩과 쿼리 임베딩 간의 코사인 유사도 계산.
    for chunk in chunks:
        # 각 청크를 반복하여 유사도 점수를 계산.
        sim_text = ch5_cosine_similarity(np.array(query_embedding) , np.array(chunk['embedding']))
        #쿼리 임베딩과 청크 헤더 임베딩 간의 코사인 유사성 계산.
        sim_header = ch5_cosine_similarity(np.array(query_embedding) , np.array(chunk['header_embedding']) )
        # 평균 유사도 점수 계산
        avg_similarity = (sim_text + sim_header) / 2
        # 청크와 평균 유사도 점수를 목록에 추가.
        similarities.append( (chunk , avg_similarity) )
    #유사도 점수를 기준으로 청크를 내림차순 정렬
    similarities.sort(key=lambda x:x[1] , reverse=True)
    # 가장 관련성 높은 상위 k개 청크 반환.
    rt = [x[0] for x in similarities[:k]]
    return rt

def ch5_generate_response(system_prompt:str , user_message:str , model:str='gpt-4o-mini'):
    '''
    시스템 프롬프트와 사용자 메시지를 기반으로 AI모델의 응답을 생성.
    
    :param system_prompt(str): AI의 행동방식을 정의
    :param user_message(str): 사용자의 질문 또는 입력
    :param model(str): 모델
    returns(str) : AI모델의 응답 객체.
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.chat.completions.create(
        model = model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

#----------------- Contextual Chunk Headers : end

#----------------- Document augmentation RAG : start
def ch6_extract_text_from_pdf(pdf_path):
    '''
    pdf 파일에서 텍스트를 추출하는 함수.       
    :pdf_path:  pdf http url 경로
    '''      
    mypdf       = fitz.open(pdf_path)
    all_text    = ''
    # pdf 의 각 페이지를 순회 하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page        = mypdf[page_num] # 페이지 객체를 가져옴.
        text        = page.get_text('text') # 해당 페이지에서 텍스트 추출
        all_text    += text # 추출된 텍스트 누적.
    return all_text

def ch6_chunk_text(text , n , overlap):
    '''
    주어진 텍스트를 n 자 단위로, 지정된 overlap만큼 겹치도록 분할합니다.
    
    :param text(str)    : 분할할 원본 텍스트
    :param n(int)       : 각 청크의 문자수
    :param overlap(int) : 청크간 겹치는 문자수.
    returns(List[str])  : 분할된 텍스트 청크 리스트
    '''
    chunks = [] # 청크를 저장할 빈리스트 초기화
    for i in range(0 , len(text) , n - overlap):
        chunks.append(text[i:i + n ])
    return chunks # 청크 리스트 반환.

def ch6_generate_questions(text_chunk , num_questions=5 , model='gpt-4o-mini'):
    '''
    주어진 텍스트 청크로부터 관련 질문들을 생성합니다.
    
    :param chunk(str):헤더로 요약할 텍스트 청크 
    :param model(str): 헤더 생성하는데 사용할 모델.   
    return (str)   : 생성된 헤더/제목.    
    '''
    # AI의 동작을 안내하는 시스템 프롬프트 정의하기
    system_prompt = (
        f"당신은 텍스트로부터 관련 질문을 생성하는 전무가입니다."
        f"제공된 텍스트를 바탕으로 그 내용에만 근거한 간결한 질문들을 생성하세요."
        f"핵심 정보와 개념에 초점을 맞추세요."
    )

    # 사용자 프롬프트 : 텍스트와 함께 질문생성 요청.
    user_prompt = (
        f"다음 텍스트를 기반으로, 해당 텍스트만으로 답할 수 있는 서로 다른 질문 {num_questions}개를 생성하세요."
        f"{text_chunk}"
        f"응답은 번호가 매겨진 질문 리스트 형식으로만 작성하고, 그외 부가 설명는 하지마세요."
        )

    # 시스템 프롬프트 및 텍스트 청크를 기반으로 AI모델에서 응답을 생성.
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.chat.completions.create(
        model = model,
        temperature=0.7,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}            
        ])
    
    # 응답에서 질문문자열 추출
    questions_text = res.choices[0].message.content.strip()
    questions = []

    # 줄 단위로 질문을 추출하고 정리
    for line in questions_text.split('\n'):
        #번호 제거 및 양쪽 공백 제거
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append( cleaned_line)
    
    return questions
    
def ch6_create_embeddings(text , model='text-embedding-3-small'):
    '''
    주어진 텍스트에 대한 임베딩을 생성
    
    :param text(str): 임베드할 입력 텍스트
    :param model(str): 사용한 임베딩 모델.
    returns (dict) : 입력텍스트에 대한 임베딩이 포함된 응답.
    '''
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.embeddings.create(        model = model ,         input = text ,    )    
    return res

def ch6_process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    문서를 처리하고, 각 청크에 대해 질문을 생성하여 벡터 저장소에 추가합니다.

    Args:
        pdf_path (str): PDF 파일 경로.
        chunk_size (int): 각 청크의 문자 수.
        chunk_overlap (int): 청크 간 중첩 문자 수.
        questions_per_chunk (int): 청크당 생성할 질문 수.

    Returns:
        Tuple[List[str], SimpleVectorStore]: 생성된 텍스트 청크 리스트와 벡터 저장소 객체.
    """
    print("PDF에서 텍스트 추출 중...")
    extracted_text = ch6_extract_text_from_pdf(pdf_path)
    
    print("텍스트 청크 분할 중...")
    text_chunks = ch6_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"총 {len(text_chunks)}개의 텍스트 청크가 생성되었습니다.")
    
    vector_store = SimpleVectorStore()
    
    print("각 청크에 대해 임베딩 및 질문 생성 중...")
    for i, chunk in enumerate( tqdm(text_chunks, desc="청크 처리 중")):
        # 청크 임베딩 생성
        chunk_embedding_response = ch6_create_embeddings(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding
        
        # 청크를 벡터 저장소에 추가
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )
        
        # 해당 청크 기반 질문 생성
        questions = ch6_generate_questions(chunk, num_questions=questions_per_chunk)
        
        # 각 질문에 대한 임베딩 생성 후 저장소에 추가
        for j, question in enumerate(questions):
            question_embedding_response = ch6_create_embeddings(question)
            question_embedding = question_embedding_response.data[0].embedding
            
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={
                    "type": "question",
                    "chunk_index": i,
                    "original_chunk": chunk
                }
            )
    
    return text_chunks, vector_store

def ch6_semantic_search1(query, vector_store, k=5):
    """
    쿼리와 벡터 저장소를 이용한 의미 기반 검색을 수행합니다.

    Args:
        query (str): 사용자 검색 쿼리.
        vector_store (SimpleVectorStore): 검색 대상 벡터 저장소.
        k (int): 반환할 결과 개수.

    Returns:
        List[Dict]: 관련성 높은 상위 k개의 항목 (텍스트, 메타데이터, 유사도 포함).
    """
    # 쿼리 임베딩 생성
    query_embedding_response = ch6_create_embeddings(query)
    query_embedding = query_embedding_response.data[0].embedding    

    
    # 벡터 저장소에서 유사한 항목 검색
    results = vector_store.similarity_search(query_embedding, k=k)
    
    return results

def ch6_prepare_context(search_results):
    """
    응답 생성을 위한 통합 컨텍스트를 구성합니다.

    Args:
        search_results (List[Dict]): 의미 기반 검색 결과.

    Returns:
        str: 결합된 전체 컨텍스트 문자열.
    """
    # 이미 포함된 청크 인덱스를 추적하기 위한 집합
    chunk_indices = set()
    context_chunks = []
    
    # 우선적으로 직접적으로 매칭된 문서 청크를 추가
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_idx = result["metadata"]["index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(f"Chunk {chunk_idx}:\n{result['text']}")
    
    # 질문이 참조하는 원본 청크도 추가 (중복 제외)
    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                original_chunk = result["metadata"]["original_chunk"]
                question_text = result["text"]
                context_chunks.append(
                    f"Chunk {chunk_idx} (참조 질문: '{question_text}'):\n{original_chunk}"
                )
    
    # 모든 청크를 하나의 문자열로 결합
    full_context = "\n\n".join(context_chunks)
    return full_context

def ch6_generate_response(query, context, model="gpt-4o-mini"):
    """
    쿼리와 컨텍스트를 기반으로 AI 응답을 생성합니다.

    Args:
        query (str): 사용자의 질문.
        context (str): 벡터 저장소에서 검색된 문맥 정보.
        model (str): 사용할 언어 모델 이름.

    Returns:
        str: 생성된 응답 텍스트.
    """
    # 시스템 프롬프트: 반드시 주어진 문맥에 기반해 응답하도록 설정
    system_prompt = (
        "당신은 주어진 컨텍스트에 기반하여 엄격하게 응답하는 AI 어시스턴트입니다. "
        "제공된 문맥에서 직접적으로 답변을 도출할 수 없는 경우, 다음과 같이 응답하세요: "
        "'I do not have enough information to answer that.'"
    )
    
    # 사용자 프롬프트: 질문과 함께 문맥을 제공
    user_prompt = f"""
        문맥:
        {context}

        질문: {query}

        위에 제공된 문맥에만 근거하여 질문에 답하세요. 간결하고 정확하게 응답해 주세요.
    """
    
    # AI 모델을 호출하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')    
    response = _client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # 생성된 응답만 반환
    return response.choices[0].message.content

def ch6_evaluate_response(query, response, reference_answer, model="gpt-4o-mini"):
    """
    AI가 생성한 응답을 기준 정답과 비교하여 평가합니다.

    Args:
        query (str): 사용자 질문.
        response (str): AI가 생성한 응답.
        reference_answer (str): 기준이 되는 정답.
        model (str): 평가에 사용할 언어 모델.

    Returns:
        str: 평가 결과 및 점수에 대한 설명.
    """
    # 평가 시스템용 시스템 프롬프트 정의
    evaluate_system_prompt = """당신은 AI 응답을 평가하는 지능형 평가 시스템입니다.
    
    AI 어시스턴트의 응답을 기준 정답과 비교하여 다음 기준으로 평가하세요:
    1. 사실성(Factual correctness) - 정확한 정보를 담고 있는가?
    2. 완전성(Completeness) - 기준 정답의 핵심 내용을 충분히 포함하고 있는가?
    3. 관련성(Relevance) - 질문에 직접적으로 응답하고 있는가?

    아래 기준에 따라 0 ~ 1 사이의 점수를 부여하세요:
    - 1.0: 내용과 의미가 완벽하게 일치함
    - 0.8: 아주 좋음, 약간의 누락 또는 차이 있음
    - 0.6: 좋음, 주요 내용을 담고 있으나 일부 세부 정보 부족
    - 0.4: 부분적인 응답, 중요한 내용이 빠짐
    - 0.2: 관련된 정보가 거의 없음
    - 0.0: 틀리거나 전혀 관련 없는 응답

    점수와 함께 평가 사유를 제시하세요.
    """
    
    # 평가 요청용 프롬프트 구성
    evaluation_prompt = f"""
    사용자 질문:
    {query}

    AI 응답:
    {response}

    기준 정답:
    {reference_answer}

    위 기준에 따라 AI 응답을 평가해 주세요.
    """
    
    # 평가 모델 호출
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')    
    eval_response = _client.chat.completions.create(    
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": evaluate_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    
    # 평가 결과 반환
    return eval_response.choices[0].message.content

def ch6_cosine_similarity(vec1, vec2):
    """
    두 벡터 간의 코사인 유사도를 계산합니다.

    Args:
        vec1 (np.ndarray): 첫 번째 벡터.
        vec2 (np.ndarray): 두 번째 벡터.

    Returns:
        float: 두 벡터 간의 코사인 유사도 값.
    """
    # 두 벡터의 내적을 계산하고, 벡터의 크기(norm) 곱으로 나누어 코사인 유사도를 구함
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def ch6_semantic_search2(query, text_chunks, embeddings, k=5):
    """
    주어진 쿼리와 임베딩을 기반으로 텍스트 청크에서 의미 기반 검색을 수행합니다.

    Args:
        query (str): 의미 검색에 사용할 쿼리.
        text_chunks (List[str]): 검색 대상이 되는 텍스트 청크 리스트.
        embeddings (List[dict]): 각 텍스트 청크에 대한 임베딩 리스트.
        k (int): 반환할 관련 청크의 수 (기본값: 5).

    Returns:
        List[str]: 쿼리와 가장 관련성 높은 상위 k개의 텍스트 청크 리스트.
    """
    # 쿼리에 대한 임베딩 생성
    query_embedding = ch6_create_embeddings(query).data[0].embedding
    similarity_scores = []  # 유사도 점수를 저장할 리스트 초기화

    # 각 텍스트 청크 임베딩과 쿼리 임베딩 간의 유사도 계산
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = ch6_cosine_similarity(
            np.array(query_embedding),
            np.array(chunk_embedding.embedding)
        )
        similarity_scores.append((i, similarity_score))  # (인덱스, 유사도) 저장

    # 유사도 기준 내림차순 정렬
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개의 인덱스를 추출
    top_indices = [index for index, _ in similarity_scores[:k]]

    # 해당 인덱스의 텍스트 청크들을 반환
    return [text_chunks[index] for index in top_indices]

def ch6_generate_response(system_prompt, user_message, model="gpt-4o-mini"):
    """
    시스템 프롬프트와 사용자 메시지를 기반으로 AI 모델의 응답을 생성합니다.

    Args:
        system_prompt (str): AI의 응답 방식을 정의하는 시스템 프롬프트.
        user_message (str): 사용자 메시지 또는 질문.
        model (str): 사용할 언어 모델.

    Returns:
        dict: AI 모델의 응답 객체.
    """
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')    
    response = _client.chat.completions.create(        
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

#----------------- Document augmentation RAG : end

#----------------- ch7 Query Transfomation RAG : start
class ch7_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 클래스입니다.
    """
    def __init__(self):
        """
        벡터 저장소 초기화.
        """
        self.vectors = []    # 임베딩 벡터를 저장할 리스트
        self.texts = []      # 원본 텍스트를 저장할 리스트
        self.metadata = []   # 각 텍스트에 대한 메타데이터를 저장할 리스트
    
    def add_item(self, text, embedding, metadata=None):
        """
        벡터 저장소에 항목을 추가합니다.

        Args:
            text (str): 원본 텍스트.
            embedding (List[float]): 임베딩 벡터.
            metadata (dict, optional): 추가 메타데이터 (기본값: None).
        """
        self.vectors.append(np.array(embedding))     # 임베딩 벡터를 NumPy 배열로 변환하여 저장
        self.texts.append(text)                      # 원본 텍스트 저장
        self.metadata.append(metadata or {})         # 메타데이터 저장 (없으면 빈 딕셔너리)

    def similarity_search(self, query_embedding, k=5):
        """
        쿼리 임베딩과 가장 유사한 항목들을 검색합니다.

        Args:
            query_embedding (List[float]): 쿼리 임베딩 벡터.
            k (int): 반환할 결과 수 (기본값: 5).

        Returns:
            List[Dict]: 가장 유사한 상위 k개 항목 (텍스트, 메타데이터, 유사도 포함).
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없으면 빈 리스트 반환

        # 쿼리 벡터를 NumPy 배열로 변환
        query_vector = np.array(query_embedding)

        # 코사인 유사도를 계산하여 (인덱스, 유사도) 튜플 저장
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))

        # 유사도를 기준으로 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개의 결과 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results

def ch7_rewrite_query(original_query , model='gpt-4o-mini'):
    '''
    검색 정확도를 높이기 위해 쿼리를 더 구체적이고 명확하게 재작성.
    
    :param original_query(str): 쿼리 원문
    :param model(str): 재작성에 사용할 언어 모델.
    return (str) : 재작된 구체적인 쿼리
    '''
    # AI 어시스턴트의 동작을 안내하는 시스템 프롬프트 정의
    system_prompt = (
        "당신은 검색 쿼리를 개선하는 데 특화된 AI 어시스턴트입니다. "
        "사용자의 원본 쿼리를 더 구체적이고 상세하게 다시 작성하여, "
        "정확한 정보 검색이 가능하도록 돕는 것이 목적입니다."
    )

    # 사용자 프롬프트: 개선이 필요한 원본 쿼리를 포함
    user_prompt = f"""
    다음 쿼리를 더 구체적이고 상세하게 다시 작성하세요. 
    관련된 키워드나 개념을 포함하여 보다 정확한 검색이 가능하도록 만드세요.

    원본 쿼리: {original_query}

    재작성된 쿼리:
    """
    # 지정된 모델을 사용하여 쿼리 재작성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(        
        model=model,
        temperature=0.0,  # 결과의 일관성을 위한 낮은 온도 설정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 결과에서 텍스트를 정제하여 반환
    return response.choices[0].message.content.strip()    

def ch7_generate_step_back_query(original_query, model="gpt-4o-mini"):
    """
    더 넓은 문맥을 검색할 수 있도록 일반화된 '스텝백(step-back)' 쿼리를 생성합니다.

    Args:
        original_query (str): 원래의 사용자 질문.
        model (str): 스텝백 쿼리 생성을 위한 언어 모델.

    Returns:
        str: 일반화된 스텝백 쿼리.
    """
    # AI 어시스턴트의 동작을 안내하는 시스템 프롬프트
    system_prompt = (
        "당신은 검색 전략에 특화된 AI 어시스턴트입니다. "
        "사용자의 구체적인 질문을 더 일반적이고 포괄적인 질문으로 바꿔, "
        "배경 지식이나 문맥을 넓게 검색할 수 있도록 도와주는 것이 목표입니다."
    )

    # 사용자 프롬프트: 일반화할 원본 쿼리를 포함
    user_prompt = f"""
    다음 질문을 더 넓고 일반적인 형태로 바꾸어,
    관련된 배경 지식이나 문맥 정보를 검색할 수 있도록 하세요.

    원본 쿼리: {original_query}

    스텝백 쿼리:
    """

    # 언어 모델을 사용하여 스텝백 쿼리 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(        
        model=model,
        temperature=0.1,  # 약간의 다양성을 위한 온도 설정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 결과 반환 (양 끝 공백 제거)
    return response.choices[0].message.content.strip()

def ch7_decompose_query(original_query, num_subqueries=4, model="gpt-4o-mini"):
    """
    복잡한 쿼리를 더 단순한 하위 쿼리로 분해합니다.

    Args:
        original_query (str): 복잡한 원본 질문.
        num_subqueries (int): 생성할 하위 질문 수.
        model (str): 쿼리 분해에 사용할 언어 모델.

    Returns:
        List[str]: 단순한 하위 질문 리스트.
    """
    # 시스템 프롬프트: 복잡한 질문을 분해하는 역할
    system_prompt = (
        "당신은 복잡한 질문을 분해하는 데 특화된 AI 어시스턴트입니다. "
        "주어진 질문을 더 단순한 하위 질문들로 나누고, "
        "이 하위 질문들이 함께 원래 질문에 대한 답변을 구성할 수 있도록 하세요."
    )

    # 사용자 프롬프트 정의
    user_prompt = f"""
    다음 복잡한 질문을 {num_subqueries}개의 더 단순한 하위 질문으로 나누세요.
    각 하위 질문은 원래 질문의 서로 다른 측면에 초점을 맞추어야 합니다.

    원본 질문: {original_query}

    다음 형식으로 {num_subqueries}개의 하위 질문을 생성하세요:
    1. [첫 번째 하위 질문]
    2. [두 번째 하위 질문]
    ...
    """

    # 모델을 호출하여 하위 질문 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(        
        model=model,
        temperature=0.2,  # 약간의 다양성을 허용
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 결과에서 질문만 추출
    content = response.choices[0].message.content.strip()
    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # 번호 제거 및 공백 정리
            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)

    return sub_queries

def ch7_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 지정된 모델을 사용하여 임베딩 벡터를 생성합니다.

    Args:
        text (str or List[str]): 임베딩을 생성할 입력 텍스트 또는 텍스트 리스트.
        model (str): 사용할 임베딩 모델 이름.

    Returns:
        List[float] or List[List[float]]: 단일 텍스트의 경우 임베딩 벡터 하나,
                                          여러 텍스트의 경우 임베딩 벡터 리스트.
    """
    # 입력이 문자열인 경우 리스트로 변환하여 처리
    input_text = text if isinstance(text, list) else [text]
    
    # 지정된 모델로 임베딩 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(       
        model = model ,         
        input = input_text ,    
    )    
    
    # 입력이 문자열이면 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding
    
    # 리스트 입력일 경우 전체 임베딩 벡터 리스트 반환
    return [item.embedding for item in response.data]

def ch7_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일의 경로.

    Returns:
        str: 추출된 전체 텍스트 문자열.
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 추출된 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]                 # 페이지 객체 가져오기
        text = page.get_text("text")           # 페이지에서 텍스트 추출
        all_text += text                       # 누적하여 전체 텍스트 구성

    return all_text  # 최종 텍스트 반환

def ch7_chunk_text(text, n=1000, overlap=200):
    """
    주어진 텍스트를 n자 단위로 중첩(overlap)을 포함하여 분할합니다.

    Args:
        text (str): 분할할 원본 텍스트.
        n (int): 각 청크의 문자 수 (기본값: 1000).
        overlap (int): 청크 간 중첩되는 문자 수 (기본값: 200).

    Returns:
        List[str]: 분할된 텍스트 청크 리스트.
    """
    chunks = []  # 청크를 저장할 리스트 초기화

    # (n - overlap)씩 이동하며 텍스트를 분할
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])  # 현재 위치부터 n자까지 잘라 청크로 추가

    return chunks  # 생성된 청크 리스트 반환

def ch7_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    RAG(Retrieval-Augmented Generation)을 위한 문서 전처리 작업을 수행합니다.

    Args:
        pdf_path (str): PDF 파일 경로.
        chunk_size (int): 각 텍스트 청크의 문자 수.
        chunk_overlap (int): 청크 간 중첩되는 문자 수.

    Returns:
        SimpleVectorStore: 청크와 해당 임베딩이 저장된 벡터 저장소 객체.
    """
    print("PDF에서 텍스트 추출 중...")
    extracted_text = ch7_extract_text_from_pdf(pdf_path)

    print("텍스트를 청크 단위로 분할 중...")
    chunks = ch7_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"총 {len(chunks)}개의 텍스트 청크가 생성되었습니다.")

    print("청크에 대한 임베딩 생성 중...")
    # 효율성을 위해 모든 청크에 대한 임베딩을 한 번에 생성
    chunk_embeddings = ch7_create_embeddings(chunks)

    # 벡터 저장소 생성
    store = ch7_SimpleVectorStore()

    # 각 청크와 임베딩을 저장소에 추가
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"벡터 저장소에 {len(chunks)}개의 청크가 추가되었습니다.")
    return store

def ch7_transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    변환된 쿼리를 사용하여 벡터 저장소에서 검색을 수행합니다.

    Args:
        query (str): 원본 사용자 쿼리.
        vector_store (SimpleVectorStore): 검색 대상 벡터 저장소.
        transformation_type (str): 쿼리 변환 방식 ('rewrite', 'step_back', 'decompose').
        top_k (int): 반환할 상위 결과 수.

    Returns:
        List[Dict]: 검색된 결과 리스트.
    """
    print(f"쿼리 변환 방식: {transformation_type}")
    print(f"원본 쿼리: {query}")

    results = []

    if transformation_type == "rewrite":
        # 쿼리 재작성
        transformed_query = ch7_rewrite_query(query)
        print(f"재작성된 쿼리: {transformed_query}")

        # 임베딩 생성 및 검색 수행
        query_embedding = ch7_create_embeddings(transformed_query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "step_back":
        # 스텝백 쿼리 생성
        transformed_query = ch7_generate_step_back_query(query)
        print(f"스텝백 쿼리: {transformed_query}")

        # 임베딩 생성 및 검색 수행
        query_embedding = ch7_create_embeddings(transformed_query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "decompose":
        # 복잡한 쿼리를 하위 쿼리로 분해
        sub_queries = ch7_decompose_query(query)
        print("하위 쿼리로 분해:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")

        # 하위 쿼리 각각에 대한 임베딩 생성 및 검색 수행
        sub_query_embeddings = ch7_create_embeddings(sub_queries)
        all_results = []

        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # 각 하위 쿼리당 적은 수 반환
            all_results.extend(sub_results)

        # 중복 제거 (동일한 텍스트가 여러 번 등장할 경우, 가장 높은 유사도 결과만 유지)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result

        # 유사도 기준 내림차순 정렬 후 top_k 개 추출
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]

    else:
        # 변환 없이 일반 쿼리로 검색
        query_embedding = ch7_create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    return results

def ch7_generate_response(query, context, model="gpt-4o-mini"):
    """
    쿼리와 검색된 문맥을 기반으로 응답을 생성합니다.

    Args:
        query (str): 사용자 질문.
        context (str): 검색된 문맥 정보.
        model (str): 응답 생성을 위한 언어 모델 이름.

    Returns:
        str: 생성된 응답 문자열.
    """
    # AI 어시스턴트의 동작을 안내하는 시스템 프롬프트 정의
    system_prompt = (
        "당신은 도움이 되는 AI 어시스턴트입니다. 사용자 질문에 대해 "
        "오직 제공된 문맥(Context)만을 기반으로 답변하세요. "
        "만약 문맥에서 답을 찾을 수 없다면, 정보가 부족하다고 솔직하게 말하세요."
    )

    # 사용자 프롬프트 구성: 문맥과 질문 포함
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        위 문맥에만 근거하여 포괄적이고 명확한 답변을 작성해주세요.
    """

    # 모델을 호출하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(            
        model=model,
        temperature=0,  # 일관된 출력 생성을 위한 낮은 온도
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 응답 텍스트 반환
    return response.choices[0].message.content.strip()

def ch7_rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    쿼리 변환을 포함한 RAG 파이프라인 전체를 실행합니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 사용자 질문.
        transformation_type (str): 쿼리 변환 방식 (None, 'rewrite', 'step_back', 'decompose').

    Returns:
        Dict: 쿼리, 변환 방식, 검색된 문맥, 생성된 응답을 포함한 결과 딕셔너리.
    """
    # PDF 문서를 처리하여 벡터 저장소 생성
    vector_store = ch7_process_document(pdf_path)

    # 쿼리 변환 적용 및 검색
    if transformation_type:
        # 변환된 쿼리를 사용한 검색
        results = ch7_transformed_search(query, vector_store, transformation_type)
    else:
        # 변환 없이 일반 쿼리로 검색
        query_embedding = ch7_create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)

    # 검색 결과에서 문맥 추출
    context = "\n\n".join([
        f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)
    ])

    # 문맥을 기반으로 응답 생성
    response = ch7_generate_response(query, context)

    # 결과 딕셔너리 반환
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }

def ch7_compare_responses(results, reference_answer, model="gpt-4o-mini"):
    """
    다양한 쿼리 변환 기법을 통해 생성된 응답들을 기준 정답과 비교하여 평가합니다.

    Args:
        results (Dict): 각 쿼리 변환 기법에 대한 결과 딕셔너리 (original, rewrite, step_back, decompose).
        reference_answer (str): 비교 대상이 되는 기준 정답.
        model (str): 평가에 사용할 언어 모델 이름.
    """
    # 평가 시스템용 시스템 프롬프트 정의
    system_prompt = (
        "당신은 RAG 시스템 평가에 특화된 전문가입니다. "
        "다양한 쿼리 변환 기법을 통해 생성된 응답을 기준 정답과 비교하여, "
        "어떤 기법이 가장 정확하고 관련성 있으며 완전한 응답을 생성했는지를 평가하세요."
    )

    # 평가용 텍스트 구성
    comparison_text = f"기준 정답:\n{reference_answer}\n\n"
    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} 쿼리 응답:\n{result['response']}\n\n"

    # 사용자 프롬프트 구성
    user_prompt = f"""
    {comparison_text}

    각 쿼리 방식 (original, rewrite, step_back, decompose)에 대해 다음을 수행하세요:
    1. 정확성, 완전성, 관련성을 기준으로 1~10 점수 부여
    2. 각 기법의 장점과 단점 기술

    마지막으로 전체 기법을 가장 잘한 순서대로 순위를 매기고,
    어떤 기법이 전반적으로 가장 효과적이었는지 그 이유를 설명하세요.
    """

    # 평가 모델 호출
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 평가 결과 출력
    print("\n***EVALUATION RESULTS***")
    print(response.choices[0].message.content)
    print("-------------------------")

def ch7_evaluate_transformations(pdf_path, query, reference_answer=None):
    """
    동일한 쿼리에 대해 다양한 쿼리 변환 기법의 성능을 평가합니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 평가할 원본 쿼리.
        reference_answer (str, optional): 기준 정답. 제공되면 비교 평가를 수행합니다.

    Returns:
        Dict: 각 기법별 RAG 결과를 포함한 딕셔너리.
    """
    # 평가할 쿼리 변환 기법 리스트 (None = 원본 쿼리 사용)
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}

    # 각 기법에 대해 RAG 파이프라인 실행
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        print(f"\n***{type_name.upper()} 쿼리로 RAG 실행 중***")

        # 변환 기법에 따라 문서 처리 및 응답 생성
        result = ch7_rag_with_query_transformation(pdf_path, query, transformation_type)
        results[type_name] = result

        # 생성된 응답 출력
        print(f"응답 ({type_name} 쿼리):")
        print(result["response"])
        print("=" * 40)

    # 기준 정답이 주어진 경우, 모든 응답을 비교 평가
    if reference_answer:
        ch7_compare_responses(results, reference_answer)

    return results
#-----------------ch7 Query Transfomation RAG : end

#-----------------ch8 Reranking RAG : start
class ch8_SimpleVectorStore:
    """
    NumPy를 사용한 간단한 벡터 스토어 구현.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        """
        벡터 스토어에 항목을 추가합니다.
        Args:
        text (str): 원본 텍스트입니다.
        embedding (List[float]): 임베딩 벡터입니다.
        metadata (dict, optional): 추가 메타데이터.
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def similarity_search(self, query_embedding, k=5):
        """
        쿼리 임베딩과 가장 유사한 항목을 찾습니다.

        Args:
        query_embedding  (List[float]): 쿼리 임베딩 벡터.
        k (int): 반환할 결과의 개수.

        Returns:
        List[Dict]: 텍스트와 메타데이터가 가장 유사한 상위 k개 항목입니다.
        """
        if not self.vectors:
            return []
        
        # 쿼리 임베딩을 numpy 배열로 변환하기
        query_vector = np.array(query_embedding)
        
        # 코사인 유사도를 사용하여 유사도 계산하기
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        # 유사도 기준 정렬(내림차순)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k 결과 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        
        return results

def ch8_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일의 경로입니다.

    Returns:
        str: PDF에서 추출한 텍스트.
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 추출된 텍스트를 저장할 빈 문자열 초기화

    # PDF의 각 페이지를 반복하며 텍스트를 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]                  # 페이지 객체 가져오기
        text = page.get_text("text")            # 텍스트 추출
        all_text += text                        # 텍스트 누적

    return all_text  # 전체 텍스트 반환

def ch8_chunk_text(text, n, overlap):
    """
    주어진 텍스트를 겹치는 n개의 문자 세그먼트로 청크합니다.

    Args:
    text (str): 청크할 텍스트입니다.
    n (int): 각 청크의 문자 수입니다.
    overlap (int): 청크 간에 겹치는 문자 수입니다.

    Returns:
    List[str]: 텍스트 청크의 목록입니다.
    """
    chunks = []  # 청크를 저장할 빈 목록을 초기화합니다.

    # 단계 크기 (n - 겹침)로 텍스트를 반복합니다.
    for i in range(0, len(text), n - overlap):
        # 청크 목록에 인덱스 i에서 i + n까지의 텍스트 청크를 추가합니다.
        chunks.append(text[i:i + n])

    return chunks  # 텍스트 청크 목록을 반환합니다.

def ch8_create_embeddings(text, model="text-embedding-3-small"):
    """
    지정된 OpenAI 모델을 사용하여 지정된 텍스트에 대한 임베딩을 생성합니다.

    Args:
    text (str): 임베딩을 생성할 입력 텍스트입니다.
    모델 (str): 임베딩을 만드는 데 사용할 모델입니다.

    Returns:
    List[float]: 임베딩 벡터입니다.
    """
    # 문자열 입력을 목록으로 변환하여 문자열 입력과 목록 입력을 모두 처리합니다.
    input_text = text if isinstance(text, list) else [text]

    # 지정된 모델을 사용하여 입력 텍스트에 대한 임베딩을 생성합니다.
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(           
        model=model,
        input=input_text
    )

    # 입력이 문자열인 경우, 첫 번째 임베딩만 반환합니다.
    if isinstance(text, str):
        return response.data[0].embedding

    # 그렇지 않으면 모든 임베딩을 벡터 목록으로 반환합니다.
    return [item.embedding for item in response.data]

def ch8_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    RAG용 문서를 처리합니다.

    Args:
        pdf_path (str): PDF 파일의 경로입니다.
        chunk_size (int): 각 청크의 크기(문자 단위).
        chunk_overlap (int): 청크 간 중첩되는 문자 수.

    Returns:
        SimpleVectorStore: 문서 청크와 해당 임베딩이 포함된 벡터 저장소.
    """
    print("PDF에서 텍스트를 추출합니다...")
    extracted_text = ch8_extract_text_from_pdf(pdf_path)

    print("텍스트를 청크 단위로 분할합니다...")
    chunks = ch8_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"{len(chunks)}개의 텍스트 청크가 생성되었습니다.")

    print("각 청크에 대한 임베딩을 생성합니다...")
    # 효율성을 위해 모든 청크에 대해 한 번에 임베딩 생성
    chunk_embeddings = ch8_create_embeddings(chunks)

    # 벡터 저장소 생성
    store = ch8_SimpleVectorStore()

    # 벡터 저장소에 청크 추가
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"총 {len(chunks)}개의 청크가 벡터 저장소에 추가되었습니다.")
    return store

def ch8_rerank_with_llm(query, results, top_n=3, model="gpt-4o-mini"):
    """
    LLM을 활용하여 검색 결과를 관련성 기준으로 재정렬합니다.

    Args:
        query (str): 사용자의 검색 질의.
        results (List[Dict]): 초기 검색 결과 목록.
        top_n (int): 재정렬 후 상위에 올 결과 수.
        model (str): 관련성 평가에 사용할 LLM 모델 이름.

    Returns:
        List[Dict]: 관련성 기준으로 재정렬된 상위 결과 리스트.
    """
    print(f"{len(results)}개의 문서를 LLM을 이용해 재정렬합니다...")

    scored_results = []  # 관련성 점수를 포함한 결과 저장용 리스트

    # 시스템 프롬프트 정의: 관련성 평가 기준 안내
    system_prompt = """
    너는 검색어에 대한 문서 관련성을 평가하는 전문가입니다.
    주어진 쿼리에 얼마나 잘 답변하는지를 기준으로 문서를 0~10 점수로 평가하세요.

    평가기준:
    - 0~2점: 전혀 관련 없음
    - 3~5점: 일부 관련 있으나 직접적인 답변은 아님
    - 6~8점: 관련 있으며 부분적으로 답변함
    - 9~10점: 매우 관련 있으며 직접적인 답변을 포함함

    단일 숫자(0~10)만 응답하세요. 그 외 텍스트는 포함하지 마세요."""

    # 각 검색 결과에 대해 LLM을 사용한 관련성 평가 수행
    for i, result in enumerate(results):
        if i % 5 == 0:
            print(f"{i+1}/{len(results)} 문서 평가 중...")

        user_prompt = (
            f"Query: {query}"
            f"Document:"
            f"{result['text']}"
            f"이 문서가 쿼리에 얼마나 관련 있는지 0~10 사이의 점수로 평가하세요:"
        )

        # 모델 호출
        _client = openai
        _client.api_key = util.getEnv('openai_api_key')          
        response = _client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # 점수 추출
        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'\b(10|[0-9])\b', score_text)

        if score_match:
            score = float(score_match.group(1))
        else:
            print(f"점수 추출 실패: '{score_text}' → similarity 점수 사용")
            score = result["similarity"] * 10

        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })

    # 관련성 점수를 기준으로 정렬
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 상위 top_n개만 반환
    return reranked_results[:top_n]

def ch8_rerank_with_keywords(query, results, top_n=3):
    """
    키워드 매칭 및 위치 기반의 간단한 재정렬 방식
    
    Args:
        query (str): 사용자의 검색 질의
        results (List[Dict]): 초기 검색 결과 목록
        top_n (int): 재정렬 후 반환할 결과 개수
        
    Returns:
        List[Dict]: 재정렬된 결과 목록
    """
    # 질의에서 중요 키워드를 추출 (길이가 3자 초과하는 단어만 선택)
    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []  # 점수를 매긴 결과를 저장할 리스트

    for result in results:
        document_text = result["text"].lower()  # 문서 내용을 소문자로 변환

        # 기본 점수는 벡터 유사도에서 시작 (0.5 가중치)
        base_score = result["similarity"] * 0.5

        keyword_score = 0  # 키워드 관련 점수 초기화
        for keyword in keywords:
            if keyword in document_text:
                # 키워드가 포함되어 있으면 0.1점 추가
                keyword_score += 0.1

                # 키워드가 문서 초반(1/4 지점 이내)에 있으면 추가로 0.1점
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.1

                # 키워드 등장 빈도에 따라 점수 추가 (최대 0.2점까지)
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)

        # 최종 점수는 기본 점수 + 키워드 점수
        final_score = base_score + keyword_score

        # 점수 포함 결과를 리스트에 추가
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })

    # 관련성 점수를 기준으로 내림차순 정렬
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 상위 top_n 개 결과만 반환
    return reranked_results[:top_n]

def ch8_generate_response(query, context, model="gpt-4o-mini"):
    """
    주어진 질의(query)와 문맥(context)을 바탕으로 응답을 생성합니다.
    
    Args:
        query (str): 사용자의 질문
        context (str): 검색된 문맥 정보
        model (str): 응답 생성을 위해 사용할 LLM 모델 이름
        
    Returns:
        str: 생성된 응답 문자열
    """
    # AI의 응답 방식에 대한 지침을 담은 시스템 프롬프트 정의
    system_prompt = (
        "당신은 유용한 AI 비서입니다. "
        "제공된 컨텍스트에 따라서만 사용자의 질문에 답변하세요. "
        "문맥에서 답을 찾을 수 없는 경우 정보가 충분하지 않다고 말하세요."
    )

    # 사용자 프롬프트 생성: 문맥 + 질문 조합
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        위의 문맥에만 근거하여 포괄적인 답변을 제공해 주세요.
    """

    # 지정된 모델을 사용하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 생성된 응답 내용을 반환
    return response.choices[0].message.content

def ch8_rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3, model="gpt-4o-mini"):
    """
    재정렬 기능을 포함한 RAG 파이프라인 전체 흐름
    
    Args:
        query (str): 사용자의 질의
        vector_store (SimpleVectorStore): 벡터 검색이 가능한 저장소
        reranking_method (str): 재정렬 방식 ('llm' 또는 'keywords')
        top_n (int): 재정렬 후 사용할 상위 문서 개수
        model (str): 응답 생성을 위한 LLM 모델 이름
        
    Returns:
        Dict: 질의, 문맥, 응답 및 중간 결과가 포함된 딕셔너리
    """
    # 질의 임베딩 생성
    query_embedding = ch8_create_embeddings(query)

    # 초기 검색 수행 (재정렬을 위해 충분히 많이 검색, 예: 10개)
    initial_results = vector_store.similarity_search(query_embedding, k=10)

    # 재정렬 수행
    if reranking_method == "llm":
        # LLM을 활용한 관련성 기반 재정렬
        reranked_results = ch8_rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        # 키워드 기반 간단한 재정렬
        reranked_results = ch8_rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        # 재정렬 없이 초기 검색 결과 상위 n개 사용
        reranked_results = initial_results[:top_n]

    # 재정렬된 결과에서 문맥(context) 구성
    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    # 문맥을 기반으로 응답 생성
    response = ch8_generate_response(query, context, model)

    # 최종 결과 반환 (디버깅이나 로그용으로 중간 값도 포함)
    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }

def ch8_evaluate_reranking(query, standard_results, reranked_results, reference_answer=None):
    """
    기본 검색 결과와 재정렬된 결과를 비교 평가합니다.
    
    Args:
        query (str): 사용자 질의
        standard_results (Dict): 기본 검색 방식의 결과
        reranked_results (Dict): 재정렬된 검색 방식의 결과
        reference_answer (str, optional): 기준이 되는 정답 (선택사항)
        
    Returns:
        str: 평가 결과 텍스트
    """
    # AI 평가자에게 역할을 부여하는 시스템 프롬프트 정의
    system_prompt = """귀하는 RAG 시스템의 전문 평가자입니다.
    두 가지 다른 검색 방법에서 검색된 컨텍스트와 응답을 비교하세요.
    어떤 것이 더 나은 컨텍스트와 더 정확하고 포괄적인 답변을 제공하는지 평가하세요."""

    # 비교를 위한 텍스트 구성 (문맥은 1000자까지만 표시)
    comparison_text = f"""Query: {query}

    Standard Retrieval Context:
    {standard_results['context'][:1000]}... [truncated]

    Standard Retrieval Answer:
    {standard_results['response']}

    Reranked Retrieval Context:
    {reranked_results['context'][:1000]}... [truncated]

    Reranked Retrieval Answer:
    {reranked_results['response']}"""

    # 참조 정답(reference answer)이 있다면 비교 텍스트에 포함
    if reference_answer:
        comparison_text += f"""
        
        Reference Answer:
        {reference_answer}"""

        # 사용자 프롬프트 구성: 어떤 방식이 더 나은지 평가 요청
        user_prompt = f"""
        {comparison_text}

        제공된 검색 방법을 평가해 주세요:
        1. 더 관련성 높은 컨텍스트
        2. 더 정확한 답변
        3. 보다 포괄적인 답변
        4. 전반적인 성능 향상

        구체적인 예시와 함께 자세한 분석을 제공하세요.
        """

    # AI 모델을 사용하여 평가 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 평가 결과 반환
    return response.choices[0].message.content
#-----------------ch8 Reranking RAG : end

#-----------------ch9 Relevant Segment Extraction RAG : start
class ch9_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 구현체입니다.
    """
    def __init__(self, dimension=1536):
        """
        벡터 저장소 초기화
        
        Args:
            dimension (int): 임베딩 벡터의 차원 수
        """
        self.dimension = dimension
        self.vectors = []     # 임베딩 벡터 리스트
        self.documents = []   # 문서(청크) 리스트
        self.metadata = []    # 각 문서의 메타데이터 리스트

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        문서와 벡터를 벡터 저장소에 추가
        
        Args:
            documents (List[str]): 문서 청크 리스트
            vectors (List[List[float]], optional): 각 문서의 임베딩 벡터 리스트
            metadata (List[Dict], optional): 각 문서에 대한 메타데이터 리스트
        """
        if vectors is None:
            vectors = [None] * len(documents)

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        """
        가장 유사한 문서를 검색 (코사인 유사도 기준)
        
        Args:
            query_vector (List[float]): 질의에 대한 임베딩 벡터
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            List[Dict]: 문서, 유사도 점수, 메타데이터가 포함된 결과 리스트
        """
        if not self.vectors or not self.documents:
            return []

        # 질의 벡터를 NumPy 배열로 변환
        query_array = np.array(query_vector)

        # 각 벡터와의 코사인 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))

        # 유사도 기준으로 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 top_k 결과 추출
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })

        return results
def ch9_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
    pdf_path (str): PDF 파일의 경로입니다.

    Returns:
    str: PDF에서 추출한 텍스트.
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = "" # 추출된 텍스트를 저장할 빈 문자열을 초기화합니다.

    # PDF의 각 페이지를 반복합니다.
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num] # 페이지 가져오기
        text = page.get_text("text") # 페이지에서 텍스트 추출
        all_text += text # 추출한 텍스트를 all_text 문자열에 추가합니다.

    return all_text # 추출된 텍스트를 반환합니다.

def ch9_chunk_text(text, chunk_size=800, overlap=0):
    """
    텍스트를 겹침(overlap) 없이 일정한 크기로 분할합니다.
    RSE(Retrieval Segment Evaluation)에서는 겹치지 않는 청크가 일반적으로 필요합니다.
    
    Args:
        text (str): 분할할 원본 텍스트
        chunk_size (int): 각 청크의 문자 길이 (기본값: 800자)
        overlap (int): 청크 간 겹치는 문자 수 (기본값: 0)
        
    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    chunks = []

    # 문자 기준으로 일정 간격마다 슬라이싱하며 청크 생성
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # 빈 청크가 아닌 경우에만 추가
            chunks.append(chunk)

    return chunks

def ch9_create_embeddings(texts, model="text-embedding-3-small"):
    """
    텍스트 리스트에 대해 임베딩 벡터를 생성합니다.
    
    Args:
        texts (List[str]): 임베딩할 텍스트들의 리스트
        model (str): 사용할 임베딩 모델 이름
        
    Returns:
        List[List[float]]: 임베딩 벡터 리스트
    """
    if not texts:
        return []  # 텍스트가 없으면 빈 리스트 반환

    # 긴 리스트일 경우 배치 단위로 처리
    batch_size = 100  # API 제한에 따라 조정 가능
    all_embeddings = []  # 전체 임베딩을 저장할 리스트 초기화

    # 텍스트 리스트를 배치 단위로 나눠 임베딩 처리
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 현재 배치 추출

        # 지정된 모델을 사용하여 배치 임베딩 생성
        _client = openai
        _client.api_key = util.getEnv('openai_api_key')
        response = _client.embeddings.create(          
            input=batch,
            model=model
        )

        # 응답에서 임베딩 벡터 추출
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 전체 리스트에 추가

    return all_embeddings  # 모든 임베딩 결과 반환

def ch9_process_document(pdf_path, chunk_size=800):
    """
    RSE(Retrieval Segment Evaluation) 또는 RAG에 사용할 수 있도록 문서를 처리합니다.

    Args:
        pdf_path (str): PDF 문서의 경로.
        chunk_size (int): 각 청크의 문자 길이.

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]:
            - 분할된 텍스트 청크 리스트.
            - 벡터 저장소 객체(SimpleVectorStore).
            - 문서 정보 (청크와 소스 경로 포함).
    """
    print("문서에서 텍스트를 추출 중...")
    text = ch9_extract_text_from_pdf(pdf_path)

    print("텍스트를 중첩 없이 청크 단위로 분할 중...")
    chunks = ch9_chunk_text(text, chunk_size=chunk_size, overlap=0)
    print(f"{len(chunks)}개의 청크가 생성되었습니다.")

    print("청크 임베딩을 생성 중...")
    chunk_embeddings = ch9_create_embeddings(chunks)

    # 벡터 저장소 생성
    vector_store = ch9_SimpleVectorStore()

    # 각 청크에 대한 메타데이터 생성
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]

    # 청크와 임베딩을 저장소에 추가
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    # 문서 정보 저장 (재구성에 사용할 수 있음)
    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }

    return chunks, vector_store, doc_info

def ch9_calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    질의에 대한 관련성과 위치 정보를 바탕으로 각 청크의 가치를 계산합니다.
    
    Args:
        query (str): 사용자 질의
        chunks (List[str]): 문서 청크 리스트
        vector_store (SimpleVectorStore): 벡터 저장소 (청크 포함)
        irrelevant_chunk_penalty (float): 관련 없는 청크에 부여할 감점 값
        
    Returns:
        List[float]: 각 청크에 대한 가치 점수 리스트
    """
    # 질의 임베딩 생성
    query_embedding = ch9_create_embeddings([query])[0]

    # 벡터 저장소에서 모든 청크에 대한 유사도 검색 (최대 청크 수만큼)
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)

    # 검색 결과로부터 청크 인덱스별 관련성 점수 매핑 생성
    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}

    # 각 청크에 대해 가치 점수 계산 (관련성 - 감점)
    chunk_values = []
    for i in range(num_chunks):
        # 해당 청크의 관련성 점수가 없으면 기본값 0 사용
        score = relevance_scores.get(i, 0.0)
        # 감점 적용: 관련 없는 청크는 음수 값이 될 수 있음
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)

    return chunk_values

def ch9_find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    최대 합 부분 배열 알고리즘(변형)을 사용하여 가장 가치 있는 연속 청크 구간(세그먼트)을 탐색합니다.

    Args:
        chunk_values (List[float]): 각 청크에 대한 점수 또는 가치 리스트.
        max_segment_length (int): 하나의 세그먼트가 가질 수 있는 최대 길이.
        total_max_length (int): 전체 포함 가능한 청크 수의 최대 합.
        min_segment_value (float): 세그먼트로 인정되기 위한 최소 점수 합계.

    Returns:
        Tuple[List[Tuple[int, int]], List[float]]: 
            - 세그먼트 리스트 (각각 시작 인덱스, 종료 인덱스 형태).
            - 각 세그먼트의 총 점수 리스트.
    """
    print("최적의 연속 세그먼트를 찾는 중...")

    best_segments = []         # 선택된 세그먼트 저장
    segment_scores = []        # 각 세그먼트의 총 점수
    total_included_chunks = 0  # 전체 포함된 청크 수 누적

    # 전체 길이 제한에 도달할 때까지 반복 탐색
    while total_included_chunks < total_max_length:
        best_score = min_segment_value
        best_segment = None

        for start in range(len(chunk_values)):
            # 이미 포함된 세그먼트와 겹치는 경우 제외
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue

            # 가능한 세그먼트 길이 내에서 탐색
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length

                # 종료 지점이 기존 세그먼트와 겹치면 제외
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue

                segment_value = sum(chunk_values[start:end])

                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)

        # 가장 가치 있는 세그먼트를 추가
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"세그먼트 {best_segment} 발견 (점수: {best_score:.4f})")
        else:
            break  # 더 이상 유효한 세그먼트 없음

    # 시작 인덱스 기준 정렬
    best_segments = sorted(best_segments, key=lambda x: x[0])

    return best_segments, segment_scores

def ch9_reconstruct_segments(chunks, best_segments):
    """
    선택된 청크 인덱스를 바탕으로 텍스트 세그먼트를 재조립합니다.
    
    Args:
        chunks (List[str]): 전체 문서를 분할한 청크 리스트
        best_segments (List[Tuple[int, int]]): (시작, 끝) 인덱스로 구성된 세그먼트 리스트
        
    Returns:
        List[str]: 재조립된 텍스트 세그먼트 리스트 (딕셔너리 형태 포함)
    """
    reconstructed_segments = []  # 재조립된 세그먼트를 저장할 리스트

    for start, end in best_segments:
        # 해당 범위의 청크들을 연결하여 하나의 세그먼트 텍스트로 생성
        segment_text = " ".join(chunks[start:end])
        
        # 세그먼트 텍스트와 인덱스 범위를 함께 저장
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })

    # 재조립된 세그먼트 리스트 반환
    return reconstructed_segments

def ch9_format_segments_for_context(segments):
    """
    LLM 입력용 문맥(context) 형식으로 세그먼트를 구성합니다.
    
    Args:
        segments (List[Dict]): 세그먼트 딕셔너리들의 리스트
        
    Returns:
        str: 포맷팅된 문맥 문자열
    """
    context = []  # 포맷팅된 문맥 문자열 조각들을 담을 리스트 초기화

    for i, segment in enumerate(segments):
        # 각 세그먼트에 대한 헤더 생성 (세그먼트 번호 및 청크 범위 표시)
        segment_header = f"SEGMENT {i+1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1]-1}):"
        context.append(segment_header)          # 헤더 추가
        context.append(segment['text'])         # 세그먼트 텍스트 추가
        context.append("-" * 80)                # 가독성을 위한 구분선 추가

    # 리스트의 모든 요소를 두 줄 간격으로 이어붙여 최종 문맥 문자열 생성
    return "\n\n".join(context)

def ch9_generate_response(query, context, model="gpt-4o-mini"):
    """
    주어진 질의와 문맥을 바탕으로 LLM 응답을 생성합니다.

    Args:
        query (str): 사용자 질의.
        context (str): 관련 세그먼트로 구성된 문맥 텍스트.
        model (str): 사용할 LLM 모델 이름.

    Returns:
        str: 생성된 응답 텍스트.
    """
    print("관련 세그먼트를 문맥으로 활용하여 응답을 생성합니다...")

    # 시스템 프롬프트: LLM의 역할 정의
    system_prompt = """당신은 제공된 문맥을 기반으로 질문에 답하는 유용한 AI 어시스턴트입니다.
    문맥은 쿼리와 관련된 문서 세그먼트로 구성되어 있으며,
    당신은 그 정보를 활용해 정확하고 포괄적인 답변을 생성해야 합니다.
    만약 문맥에 해당 질문에 대한 직접적인 정보가 없다면, 그 사실을 명확히 언급하세요."""

    # 사용자 프롬프트: 문맥 + 질의
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    위 문맥에 따라 가능한 한 구체적이고 유익한 답변을 작성해 주세요.
    """

    # LLM 호출하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

def ch9_rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    Relevant Segment Extraction(RSE)을 포함한 RAG 파이프라인 전체 실행 함수입니다.

    Args:
        pdf_path (str): 처리할 PDF 문서의 경로.
        query (str): 사용자 질의.
        chunk_size (int): 분할할 청크의 문자 수.
        irrelevant_chunk_penalty (float): 관련 없는 청크에 부과할 감점 값.

    Returns:
        Dict: 다음 정보를 포함한 결과 딕셔너리:
            - query: 사용자 질의.
            - segments: 선택된 세그먼트 텍스트.
            - response: LLM이 생성한 응답.
    """
    print("\n***RAG with Relevant Segment Extraction 시작***")
    print(f"사용자 질문: {query}")

    # 1. 문서 전처리 (PDF → 텍스트 추출 → 청크 분할 → 임베딩 생성)
    chunks, vector_store, doc_info = ch9_process_document(pdf_path, chunk_size)

    # 2. 각 청크에 대해 쿼리 기반 관련성 점수 계산
    print("\n관련성 점수 계산 중...")
    chunk_values = ch9_calculate_chunk_values(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        irrelevant_chunk_penalty=irrelevant_chunk_penalty
    )

    # 3. 가장 높은 점수를 가진 연속 세그먼트 선택
    best_segments, scores = ch9_find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )

    # 4. 선택된 세그먼트를 기반으로 문맥 구성
    print("\n세그먼트 재구성 중...")
    segments = ch9_reconstruct_segments(chunks, best_segments)

    # 5. LLM 입력용 문맥 문자열 포맷팅
    context = ch9_format_segments_for_context(segments)

    # 6. 문맥 + 질의를 기반으로 응답 생성
    response = ch9_generate_response(query, context)

    # 7. 결과 정리
    result = {
        "query": query,
        "segments": segments,
        "response": response
    }

    print("\n***최종 응답 결과***")
    print(response)

    return result

def ch9_standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    상위 k개의 청크를 검색하여 문맥으로 사용하는 표준 RAG 방식입니다.
    
    Args:
        pdf_path (str): PDF 문서 경로
        query (str): 사용자 질의
        k (int): 검색할 상위 관련 청크 개수
        chunk_size (int): 청크 크기 (문자 단위)
        
    Returns:
        Dict: 질의, 검색된 청크, 생성된 응답이 포함된 결과
    """
    print("\n=== STARTING STANDARD TOP-K RETRIEVAL ===")
    print(f"Query: {query}")

    # 문서 전처리: 텍스트 추출, 청크 분할, 임베딩 생성
    chunks, vector_store, doc_info = ch9_process_document(pdf_path, chunk_size)

    # 질의 임베딩 생성 후, 상위 k개 청크 검색
    print("Creating query embedding and retrieving chunks...")
    query_embedding = ch9_create_embeddings([query])[0]
    results = vector_store.search(query_embedding, top_k=k)

    # 검색된 청크 텍스트만 추출
    retrieved_chunks = [result["document"] for result in results]

    # 검색된 청크들을 문맥 문자열로 포맷팅
    context = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk}" 
        for i, chunk in enumerate(retrieved_chunks)
    ])

    # 문맥을 기반으로 LLM 응답 생성
    response = ch9_generate_response(query, context)

    # 결과 정리
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n=== FINAL RESPONSE ===")
    print(response)

    return result

def ch9_evaluate_methods(pdf_path, query, reference_answer=None):
    """
    RSE 방식과 표준 Top-K 검색 방식을 비교 평가합니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 사용자 질문.
        reference_answer (str, optional): 기준 정답 (있을 경우 응답 평가 수행).

    Returns:
        Dict: RSE 방식과 표준 방식의 결과를 포함한 딕셔너리.
    """
    print("\n========= 평가 시작 =========\n")

    # 1. Relevant Segment Extraction 기반 RAG 실행
    rse_result = ch9_rag_with_rse(pdf_path, query)

    # 2. 표준 Top-K 검색 기반 RAG 실행
    standard_result = ch9_standard_top_k_retrieval(pdf_path, query)

    # 3. 기준 정답이 주어졌을 경우, 두 응답을 비교 평가
    if reference_answer:
        print("\n=== 응답 비교 평가 ===")

        evaluation_prompt = f"""
    Query: {query}

    [Reference Answer]
    {reference_answer}

    [Standard Retrieval Response]
    {standard_result['response']}

    [Relevant Segment Extraction Response]
    {rse_result['response']}

    위 두 응답을 기준 정답과 비교하여 다음을 판단하세요:
    1. 더 정확하고 포괄적인 응답은 무엇인가요?
    2. 사용자 질문을 더 잘 해결한 응답은 어떤 것인가요?
    3. 불필요하거나 관련 없는 정보를 덜 포함한 응답은 무엇인가요?

    각 항목에 대해 근거를 명확히 설명한 후, 어느 방식이 더 우수했는지 종합적으로 평가하세요.
    """

    print("기준 정답과 비교하여 응답 평가 중...")
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    evaluation = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 RAG 응답을 공정하게 평가하는 시스템입니다."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    print("\n=== 평가 결과 ===")
    print(evaluation.choices[0].message.content)

    return {
        "rse_result": rse_result,
        "standard_result": standard_result
    }
#-----------------ch9 Relevant Segment Extraction RAG : end

#-----------------ch10 문맥압축 Contextual Compression RAG : start
class ch10_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 구현체입니다.
    """
    def __init__(self):
        """
        벡터 저장소 초기화
        """
        self.vectors = []     # 임베딩 벡터 리스트
        self.texts = []       # 원본 텍스트 리스트
        self.metadata = []    # 각 텍스트의 메타데이터 리스트

    def add_item(self, text, embedding, metadata=None):
        """
        단일 텍스트 항목을 벡터 저장소에 추가합니다.

        Args:
            text (str): 원본 텍스트
            embedding (List[float]): 텍스트의 임베딩 벡터
            metadata (dict, optional): 추가 메타데이터 (기본값: 빈 딕셔너리)
        """
        self.vectors.append(np.array(embedding))  # 임베딩을 NumPy 배열로 변환하여 저장
        self.texts.append(text)                   # 원본 텍스트 저장
        self.metadata.append(metadata or {})      # 메타데이터 저장 (None일 경우 빈 딕셔너리)

    def similarity_search(self, query_embedding, k=5):
        """
        질의 임베딩과 가장 유사한 텍스트를 검색합니다.

        Args:
            query_embedding (List[float]): 질의 임베딩 벡터
            k (int): 반환할 상위 결과 개수

        Returns:
            List[Dict]: 유사한 항목 리스트 (텍스트, 메타데이터, 유사도 포함)
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없으면 빈 리스트 반환

        # 질의 벡터를 NumPy 배열로 변환
        query_vector = np.array(query_embedding)

        # 각 벡터와의 코사인 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))  # 인덱스와 유사도 점수 저장

        # 유사도 기준 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 항목 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results
    
def ch10_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text

def ch10_chunk_text(text, n=1000, overlap=200):
    """
    주어진 텍스트를 n자 단위로 분할하되, 각 청크 간에 overlap만큼 겹치게 합니다.

    Args:
        text (str): 분할할 텍스트
        n (int): 각 청크의 문자 수 (기본값: 1000)
        overlap (int): 청크 간 겹치는 문자 수 (기본값: 200)

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    chunks = []  # 청크들을 저장할 리스트 초기화

    # n - overlap 만큼 이동하면서 청크 생성
    for i in range(0, len(text), n - overlap):
        # i에서 i + n까지의 텍스트 조각을 청크로 추가
        chunks.append(text[i:i + n])

    return chunks  # 생성된 청크 리스트 반환

def ch10_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 임베딩을 생성합니다.

    Args:
        text (str 또는 List[str]): 임베딩을 생성할 입력 텍스트(또는 텍스트 리스트)
        model (str): 사용할 임베딩 모델 이름

    Returns:
        List[float] 또는 List[List[float]]: 생성된 임베딩 벡터 또는 벡터 리스트
    """
    # 입력이 문자열 하나일 수도 있고, 문자열 리스트일 수도 있으므로 리스트 형태로 통일
    input_text = text if isinstance(text, list) else [text]

    # 지정된 모델을 사용하여 임베딩 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(       
        model=model,
        input=input_text
    )

    # 입력이 단일 문자열이었을 경우, 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding

    # 여러 문자열일 경우, 모든 임베딩 리스트 반환
    return [item.embedding for item in response.data]

def ch10_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    RAG 처리를 위한 문서 전처리 함수입니다.

    Args:
        pdf_path (str): PDF 파일 경로.
        chunk_size (int): 각 청크의 문자 수.
        chunk_overlap (int): 청크 간 겹치는 문자 수.

    Returns:
        SimpleVectorStore: 청크 및 임베딩이 저장된 벡터 저장소 객체.
    """
    print("PDF에서 텍스트를 추출합니다...")
    extracted_text = ch10_extract_text_from_pdf(pdf_path)

    print("텍스트를 청크 단위로 분할합니다...")
    chunks = ch10_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"{len(chunks)}개의 청크가 생성되었습니다.")

    print("청크 임베딩을 생성합니다...")
    chunk_embeddings = ch10_create_embeddings(chunks)

    print("벡터 저장소를 초기화합니다...")
    store = ch10_SimpleVectorStore()

    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"{len(chunks)}개의 청크가 벡터 저장소에 추가되었습니다.")
    return store

def ch10_compress_chunk(chunk, query, compression_type="selective", model="gpt-4o-mini"):
    """
    주어진 텍스트 청크에서 질의에 관련된 정보만 추출하여 압축합니다.

    Args:
        chunk (str): 압축 대상 텍스트 청크
        query (str): 사용자 질의
        compression_type (str): 압축 방식 ("selective", "summary", "extraction")
        model (str): 사용할 LLM 모델 이름

    Returns:
        Tuple[str, float]: 압축된 청크 문자열과 압축률(%) 
    """
    
    # 압축 방식에 따라 시스템 프롬프트 설정
    if compression_type == "selective":
        # 관련 문장 또는 문단만 **선택적 추출**
        system_prompt = """귀하는 정보 필터링 전문가입니다.
        귀하의 임무는 문서 청크를 분석하여 사용자의 쿼리와 직접적으로 관련된 문장이나 단락만
        추출하는 것입니다. 관련 없는 콘텐츠는 모두 제거하세요.

        출력물은 다음과 같아야 합니다:
        1. 쿼리에 대한 답변에 도움이 되는 텍스트만 포함해야 합니다.
        2. 관련 문장의 정확한 표현을 유지하세요(의역하지 마세요).
        3. 텍스트의 원래 순서 유지
        4. 중복되는 것처럼 보이더라도 모든 관련 콘텐츠를 포함하세요.
        5. 쿼리와 관련이 없는 모든 텍스트 제외

        추가 설명 없이 일반 텍스트로 응답 형식을 지정하세요."""
    
    elif compression_type == "summary":
        # 관련 내용을 **요약** 형태로 제공
        system_prompt = """귀하는 요약의 전문가입니다.
        여러분의 임무는 제공된 청크의 간결한 요약을 작성하여 사용자의 쿼리와 관련된 정보에만 초점을 맞추는 것입니다.
        정보에만 초점을 맞춘 간결한 요약을 작성하는 것입니다.

        출력물은 다음과 같아야 합니다:
        1. 쿼리 관련 정보에 대해 간략하지만 포괄적으로 작성해야 합니다.
        2. 쿼리와 관련된 정보에만 집중해야 합니다.
        3. 관련 없는 세부 정보는 생략
        4. 중립적이고 사실적인 어조로 작성합니다.

        추가 설명 없이 일반 텍스트로 응답 형식을 지정합니다."""
    
    else:  # "extraction"
        # 질의 관련 **문장만 원문 그대로 추출**
        system_prompt = """귀하는 정보 추출 전문가입니다.
        귀하의 임무는 문서 청크에서 사용자의 쿼리에 대한 답변과 관련된 정보가 포함된 정확한 문장만
        정확한 문장만 추출하는 것입니다.

        출력은 다음과 같아야 합니다:
        1. 원본 텍스트에서 관련 문장의 직접 인용문만 포함해야 합니다.
        2. 원본 문구를 그대로 유지합니다(텍스트를 수정하지 않습니다).
        3. 쿼리와 직접 관련된 문장만 포함하세요.
        4. 추출된 문장을 개행으로 구분합니다.
        5. 주석이나 추가 텍스트를 추가하지 마세요.

        추가 설명 없이 일반 텍스트로 응답 형식을 지정합니다."""

    # 사용자 프롬프트 구성 (질의 + 문서 청크)
    user_prompt = f"""
        Query: {query}

        Document Chunk:
        {chunk}

        이 쿼리에 대한 답변과 관련된 콘텐츠만 추출합니다.
    """

    # LLM을 호출하여 압축된 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 생성된 압축된 텍스트 추출
    compressed_chunk = response.choices[0].message.content.strip()

    # 압축률 계산 (압축 전과 후 길이 차이로 % 계산)
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100

    return compressed_chunk, compression_ratio

def ch10_batch_compress_chunks(chunks, query, compression_type="selective", model="gpt-4o-mini"):
    """
    여러 개의 텍스트 청크를 개별적으로 압축하여 반환합니다.

    Args:
        chunks (List[str]): 압축 대상이 되는 텍스트 청크 리스트.
        query (str): 사용자 질의.
        compression_type (str): 압축 방식 ("selective", "summary", "extraction").
        model (str): 사용할 LLM 모델 이름.

    Returns:
        List[Tuple[str, float]]: (압축된 텍스트, 개별 압축률)로 구성된 리스트.
    """
    print(f"{len(chunks)}개의 청크 압축을 시작합니다...")
    results = []
    total_original_length = 0
    total_compressed_length = 0

    for i, chunk in enumerate(chunks):
        print(f"[{i + 1}/{len(chunks)}] 청크 압축 중...")

        # 개별 청크 압축 수행
        compressed_chunk, compression_ratio = ch10_compress_chunk(
            chunk, query, compression_type, model
        )
        results.append((compressed_chunk, compression_ratio))

        total_original_length += len(chunk)
        total_compressed_length += len(compressed_chunk)

    # 전체 압축률 출력
    overall_ratio = (
        (total_original_length - total_compressed_length) / total_original_length * 100
        if total_original_length > 0 else 0.0
    )
    print(f"전체 압축률: {overall_ratio:.2f}%")

    return results

def ch10_generate_response(query, context, model="gpt-4o-mini"):
    """
    질의(query)와 문맥(context)을 바탕으로 LLM 응답을 생성합니다.
    
    Args:
        query (str): 사용자 질의
        context (str): 압축된 청크에서 추출한 문맥 텍스트
        model (str): 사용할 LLM 모델 이름
        
    Returns:
        str: 생성된 응답 문자열
    """
    # 시스템 프롬프트 정의: LLM의 역할과 응답 조건 지정
    system_prompt = """당신은 유용한 AI 비서입니다. 제공된 문맥에만 근거하여 사용자의 질문에 답변하세요.
    문맥에서 답을 찾을 수 없는 경우 정보가 충분하지 않다고 말합니다."""

    # 사용자 프롬프트 구성: 문맥 + 질문
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        위의 문맥에만 근거하여 포괄적인 답변을 제공하세요.
    """

    # LLM 호출을 통해 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 응답 일관성을 위한 설정
    )

    # 생성된 응답 텍스트 반환
    return response.choices[0].message.content

def ch10_rag_with_compression(pdf_path, query, k=10, compression_type="selective", model="gpt-4o-mini"):
    """
    압축 기반 문맥 생성이 포함된 RAG 파이프라인을 실행합니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 사용자 질의.
        k (int): 초기 검색할 청크 수.
        compression_type (str): 압축 방식 ("selective", "summary", "extraction").
        model (str): 사용할 LLM 모델 이름.

    Returns:
        dict: 쿼리, 압축된 문맥, 응답, 압축률 등의 결과 딕셔너리.
    """
    print("\n=== 문맥 압축 기반 RAG 실행 ===")
    print(f"질문: {query}")
    print(f"압축 방식: {compression_type}")

    # 1. 문서 전처리 → 벡터 저장소 생성
    vector_store = ch10_process_document(pdf_path)

    # 2. 쿼리 임베딩 생성
    query_embedding = ch10_create_embeddings(query)

    # 3. 유사도 기반 상위 k개 청크 검색
    print(f"상위 {k}개의 관련 청크 검색 중...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    # 4. 검색된 청크에 대해 압축 수행
    compressed_results = ch10_batch_compress_chunks(retrieved_chunks, query, compression_type, model)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    # 5. 압축된 청크가 모두 비어 있는 경우 예외 처리
    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:
        print("⚠️ 모든 청크가 빈 문자열로 압축되었습니다. 원본 청크를 사용합니다.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)

    # 6. 압축된 청크들을 문맥으로 구성
    context = "\n\n---\n\n".join(compressed_chunks)

    # 7. 문맥을 기반으로 응답 생성
    print("압축된 문맥을 기반으로 응답 생성 중...")
    response = ch10_generate_response(query, context, model)

    # 8. 결과 반환
    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }

    print("\n=== 최종 응답 ===")
    print(response)

    return result

def ch10_standard_rag(pdf_path, query, k=10, model="gpt-4o-mini"):
    """
    LLM 기반 압축 없이 수행하는 표준 RAG 파이프라인입니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 사용자 질의.
        k (int): 검색할 상위 유사 청크 개수.
        model (str): 응답 생성을 위한 LLM 모델 이름.

    Returns:
        dict: 다음 정보를 포함한 결과 딕셔너리:
            - query: 사용자 질의
            - chunks: 검색된 청크 목록
            - response: LLM이 생성한 응답
    """
    print("\n***표준 RAG 실행***")
    print(f"질문: {query}")

    # 1. 문서 처리 및 벡터 저장소 생성
    vector_store = ch10_process_document(pdf_path)

    # 2. 쿼리 임베딩 생성
    query_embedding = ch10_create_embeddings(query)

    # 3. 상위 k개의 관련 청크 검색
    print(f"상위 {k}개의 청크 검색 중...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    # 4. 검색된 청크를 문맥 문자열로 조합
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 5. 문맥 기반 LLM 응답 생성
    print("응답 생성 중...")
    response = ch10_generate_response(query, context, model)

    # 6. 결과 딕셔너리 구성 및 반환
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n***최종 응답***")
    print(response)

    return result

def ch10_evaluate_responses(query, responses, reference_answer):
    """
    여러 RAG 응답을 기준 정답과 비교하여 평가합니다.

    Args:
        query (str): 사용자 질의.
        responses (Dict[str, str]): 각 방식별 응답 딕셔너리. (예: {"standard": ..., "compressed": ...})
        reference_answer (str): 기준 정답.

    Returns:
        str: 평가 결과 텍스트.
    """
    # 시스템 프롬프트: 평가자 역할 정의
    system_prompt = """당신은 다양한 RAG 응답을 평가하는 공정한 평가자입니다.
    각 응답을 기준 정답과 비교하여 정확성, 포괄성, 관련성, 간결성을 기준으로 평가하고
    가장 우수한 응답부터 순위를 매기세요."""

    # 사용자 프롬프트 구성
    user_prompt = f"""
    Query: {query}

    [Reference Answer]
    {reference_answer}

    """

    # 각 방식별 응답 추가
    for method, response in responses.items():
        user_prompt += f"[{method.capitalize()} Response]\n{response}\n\n"

    # 평가 기준 안내 추가
    user_prompt += """
    각 응답을 다음 기준에 따라 평가하세요:
    1. 정확성 (기준 정답과의 사실 일치 여부)
    2. 포괄성 (질문에 대해 얼마나 완전하게 답했는지)
    3. 간결성 (불필요한 정보 없이 핵심만 전달했는지)
    4. 전반적인 품질

    각 응답에 대한 분석을 제공한 후, 가장 우수한 응답부터 순위를 정하고 그 이유를 설명하세요.
    """

    # 평가 요청 전송
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    evaluation_response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return evaluation_response.choices[0].message.content

def ch10_evaluate_compression(pdf_path, query, reference_answer=None, compression_types=["selective", "summary", "extraction"]):
    """
    다양한 문맥 압축 방식과 standard RAG를 비교 평가합니다.

    Args:
        pdf_path (str): PDF 문서 경로.
        query (str): 사용자 질의.
        reference_answer (str): 기준 정답 (있을 경우 평가 수행).
        compression_types (List[str]): 평가 대상 압축 방식 목록.

    Returns:
        dict: 다음을 포함한 평가 결과 딕셔너리:
            - query: 질의
            - responses: 각 방식의 응답
            - evaluation: LLM 기반 평가 결과 텍스트
            - metrics: 압축률 및 문맥 길이 비교
            - standard_result: 압축 미사용 결과
            - compression_results: 압축 방식별 결과
    """
    print("\n*** 문맥 압축 방식 평가 시작 ***")
    print(f"질문: {query}")

    # 1. standard RAG 실행 (압축 없음)
    standard_result = ch10_standard_rag(pdf_path, query)

    # 2. 압축 방식별 RAG 결과 저장
    compression_results = {}
    for comp_type in compression_types:
        print(f"\n[{comp_type.upper()} 압축 방식 평가 중...]")
        compression_results[comp_type] = ch10_rag_with_compression(
            pdf_path=pdf_path,
            query=query,
            compression_type=comp_type
        )

    # 3. 방식별 응답 수집
    responses = {
        "standard": standard_result["response"]
    }
    for comp_type in compression_types:
        responses[comp_type] = compression_results[comp_type]["response"]

    # 4. 평가 수행 (참조 정답이 있는 경우)
    if reference_answer:
        evaluation = ch10_evaluate_responses(query, responses, reference_answer)
        print("\n*** 평가 결과 ***")
        print(evaluation)
    else:
        evaluation = "기준 정답이 제공되지 않아 자동 평가를 생략했습니다."

    # 5. 압축 방식별 메트릭 계산
    metrics = {}
    for comp_type in compression_types:
        avg_ratio = sum(compression_results[comp_type]["compression_ratios"]) / len(compression_results[comp_type]["compression_ratios"])
        metrics[comp_type] = {
            "avg_compression_ratio": f"{avg_ratio:.2f}%",
            "total_context_length": len("\n\n".join(compression_results[comp_type]["compressed_chunks"])),
            "original_context_length": len("\n\n".join(standard_result["chunks"]))
        }

    # 6. 결과 정리 및 반환
    return {
        "query": query,
        "responses": responses,
        "evaluation": evaluation,
        "metrics": metrics,
        "standard_result": standard_result,
        "compression_results": compression_results
    }

def ch10_visualize_compression_results(evaluation_results):
    """
    다양한 압축 방식의 결과를 시각적으로 비교 출력합니다.

    Args:
        evaluation_results (Dict): evaluate_compression 함수의 실행 결과.
    """
    # 질의 및 standard 방식의 청크 가져오기
    query = evaluation_results["query"]
    standard_chunks = evaluation_results["standard_result"]["chunks"]

    print(f"\n질문:\n{query}")
    print("\n" + "="*80 + "\n")

    # 비교용으로 standard 방식의 첫 번째 청크 선택
    original_chunk = standard_chunks[0]

    # 압축 방식별 비교 시각화
    for comp_type in evaluation_results["compression_results"].keys():
        compressed_chunks = evaluation_results["compression_results"][comp_type]["compressed_chunks"]
        compression_ratios = evaluation_results["compression_results"][comp_type]["compression_ratios"]

        compressed_chunk = compressed_chunks[0]
        compression_ratio = compression_ratios[0]

        print(f"\n=== {comp_type.upper()} COMPRESSION EXAMPLE ===\n")

        # 원본 청크 출력
        print("원본 청크:")
        print("-" * 40)
        if len(original_chunk) > 800:
            print(original_chunk[:800] + "... [중략]")
        else:
            print(original_chunk)
        print("-" * 40)
        print(f"문자 수: {len(original_chunk)}\n")

        # 압축 청크 출력
        print("압축된 청크:")
        print("-" * 40)
        print(compressed_chunk)
        print("-" * 40)
        print(f"문자 수: {len(compressed_chunk)}")
        print(f"해당 청크 압축률: {compression_ratio:.2f}%\n")

        # 전체 평균 압축률 및 길이 감소 정보
        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        print(f"전체 평균 압축률: {avg_ratio:.2f}%")
        print(f"총 문맥 길이 감소율: {evaluation_results['metrics'][comp_type]['avg_compression_ratio']}")
        print("-" * 80)

    # 압축 방식별 요약 통계 표 출력
    print("\n*** 압축 방식 요약 통계 ***\n")
    print(f"{'방식':<15} {'평균 압축률':<20} {'압축 후 길이':<18} {'원본 길이':<15}")
    print("-" * 70)
    for comp_type, metrics in evaluation_results["metrics"].items():
        print(f"{comp_type:<15} {metrics['avg_compression_ratio']:<20} {metrics['total_context_length']:<18} {metrics['original_context_length']:<15}")
#-----------------ch10 문맥압축 Contextual Compression RAG : end

#-----------------ch11 피드백 루프 Feedback Loop RAG : start
class ch11_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 구현체입니다.
    """
    def __init__(self):
        """
        벡터 저장소 초기화
        """
        self.vectors = []     # 임베딩 벡터 리스트
        self.texts = []       # 원본 텍스트 리스트
        self.metadata = []    # 각 텍스트의 메타데이터 리스트

    def add_item(self, text, embedding, metadata=None):
        """
        단일 텍스트 항목을 벡터 저장소에 추가합니다.

        Args:
            text (str): 원본 텍스트
            embedding (List[float]): 텍스트의 임베딩 벡터
            metadata (dict, optional): 추가 메타데이터 (기본값: 빈 딕셔너리)
        """
        self.vectors.append(np.array(embedding))  # 임베딩을 NumPy 배열로 변환하여 저장
        self.texts.append(text)                   # 원본 텍스트 저장
        self.metadata.append(metadata or {})      # 메타데이터 저장 (None일 경우 빈 딕셔너리)

    def similarity_search(self, query_embedding, k=5):
        """
        질의 임베딩과 가장 유사한 텍스트를 검색합니다.

        Args:
            query_embedding (List[float]): 질의 임베딩 벡터
            k (int): 반환할 상위 결과 개수

        Returns:
            List[Dict]: 유사한 항목 리스트 (텍스트, 메타데이터, 유사도 포함)
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없으면 빈 리스트 반환

        # 질의 벡터를 NumPy 배열로 변환
        query_vector = np.array(query_embedding)

        # 각 벡터와의 코사인 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))  # 인덱스와 유사도 점수 저장

        # 유사도 기준 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 항목 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results

def ch11_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text

def ch11_chunk_text(text, n=1000, overlap=200):
    """
    주어진 텍스트를 n자 단위로 분할하되, 각 청크 간에 overlap만큼 겹치게 합니다.

    Args:
        text (str): 분할할 텍스트
        n (int): 각 청크의 문자 수 (기본값: 1000)
        overlap (int): 청크 간 겹치는 문자 수 (기본값: 200)

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    chunks = []  # 청크들을 저장할 리스트 초기화

    # n - overlap 만큼 이동하면서 청크 생성
    for i in range(0, len(text), n - overlap):
        # i에서 i + n까지의 텍스트 조각을 청크로 추가
        chunks.append(text[i:i + n])

    return chunks  # 생성된 청크 리스트 반환

def ch11_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 임베딩을 생성합니다.

    Args:
        text (str 또는 List[str]): 임베딩을 생성할 입력 텍스트(또는 텍스트 리스트)
        model (str): 사용할 임베딩 모델 이름

    Returns:
        List[float] 또는 List[List[float]]: 생성된 임베딩 벡터 또는 벡터 리스트
    """
    # 입력이 문자열 하나일 수도 있고, 문자열 리스트일 수도 있으므로 리스트 형태로 통일
    input_text = text if isinstance(text, list) else [text]

    # 지정된 모델을 사용하여 임베딩 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(           
        model=model,
        input=input_text
    )

    # 입력이 단일 문자열이었을 경우, 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding

    # 여러 문자열일 경우, 모든 임베딩 리스트 반환
    return [item.embedding for item in response.data]

def ch11_get_user_feedback(query, response, relevance, quality, comments=""):
    """
    사용자 피드백을 딕셔너리 형태로 포맷합니다.
    
    Args:
        query (str): 사용자의 질의
        response (str): 시스템의 응답
        relevance (int): 관련성 점수 (1~5)
        quality (int): 품질 점수 (1~5)
        comments (str): 선택적인 코멘트 (기본값: 빈 문자열)
        
    Returns:
        Dict: 포맷된 피드백 딕셔너리
    """
    return {
        "query": query,                      # 사용자 질의
        "response": response,                # 시스템 응답
        "relevance": int(relevance),         # 관련성 평가 점수
        "quality": int(quality),             # 품질 평가 점수
        "comments": comments,                # 사용자 코멘트
        "timestamp": datetime.now().isoformat()  # 타임스탬프 (ISO 형식)
    }

def ch11_store_feedback(feedback, feedback_file=r"D:\python_workspace\FastApi\Area\Rag\feedback_data.json"):
    """
    피드백을 JSON 파일에 저장합니다.
    
    Args:
        feedback (Dict): 피드백 데이터
        feedback_file (str): 피드백 파일 경로
    """
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")

def ch11_load_feedback_data(feedback_file=r"D:\python_workspace\FastApi\Area\Rag\feedback_data.json"):
    """
    피드백 데이터를 JSON 파일에서 불러옵니다.

    Args:
        feedback_file (str): 피드백 파일 경로.

    Returns:
        List[Dict]: 피드백 항목으로 구성된 리스트.
    """
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():  # 빈 줄 제외
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("피드백 데이터 파일이 존재하지 않습니다. 빈 리스트로 시작합니다.")

    return feedback_data        

def ch11_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    피드백 루프가 포함된 RAG(Retrieval Augmented Generation)용 문서 처리 함수.
    이 함수는 전체 문서 처리 파이프라인을 다룹니다:
    1. PDF에서 텍스트 추출
    2. 겹침이 있는 텍스트 청크 분할
    3. 각 청크에 대한 임베딩 생성
    4. 메타데이터와 함께 벡터 데이터베이스에 저장

    Args:
        pdf_path (str): 처리할 PDF 파일 경로
        chunk_size (int): 각 텍스트 청크의 문자 수
        chunk_overlap (int): 연속된 청크 간의 겹치는 문자 수

    Returns:
        Tuple[List[str], SimpleVectorStore]: 다음을 포함하는 튜플
            - 텍스트 청크 리스트
            - 임베딩 및 메타데이터가 저장된 벡터 저장소
    """
    # 1단계: PDF 문서에서 원시 텍스트 추출
    print("PDF에서 텍스트 추출 중... (Extracting text from PDF...)")
    extracted_text = ch11_extract_text_from_pdf(pdf_path)
    
    # 2단계: 더 나은 문맥 유지를 위해 텍스트를 겹치는 청크로 분할
    print("텍스트 청크 분할 중... (Chunking text...)")
    chunks = ch11_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"{len(chunks)}개의 텍스트 청크 생성됨 (Created {len(chunks)} text chunks)")
    
    # 3단계: 각 텍스트 청크에 대한 벡터 임베딩 생성
    print("청크에 대한 임베딩 생성 중... (Creating embeddings for chunks...)")
    chunk_embeddings = ch11_create_embeddings(chunks)
    
    # 4단계: 청크와 임베딩을 저장할 벡터 저장소 초기화
    store = ch11_SimpleVectorStore()
    
    # 5단계: 각 청크를 임베딩과 함께 벡터 저장소에 추가
    # 메타데이터에는 피드백 기반 개선을 위한 정보 포함
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,                # 원문 내 위치
                "source": pdf_path,        # 원본 문서 경로
                "relevance_score": 1.0,    # 초기 관련성 점수 (피드백으로 갱신 가능)
                "feedback_count": 0        # 해당 청크에 대한 피드백 횟수
            }
        )
    
    print(f"{len(chunks)}개의 청크가 벡터 저장소에 추가됨 (Added {len(chunks)} chunks to the vector store)")
    return chunks, store

def ch11_assess_feedback_relevance(query, doc_text, feedback):
    """
    이전 피드백 항목이 현재 쿼리 및 문서와 관련 있는지를 LLM을 사용하여 평가합니다.

    이 함수는 현재 쿼리, 과거 쿼리+피드백, 그리고 문서 내용을 LLM에 전달하여 
    과거 피드백이 현재 정보 검색에 영향을 줄 수 있는지를 판단합니다.

    Args:
        query (str): 현재 사용자의 정보 검색 쿼리
        doc_text (str): 평가 대상 문서의 텍스트 내용
        feedback (Dict): 'query'와 'response' 키를 포함하는 이전 피드백 데이터

    Returns:
        bool: 피드백이 현재 쿼리/문서와 관련 있다고 판단되면 True, 그렇지 않으면 False
    """
    # 시스템 프롬프트: LLM이 '관련 있음/없음' 판단만 하도록 지시
    system_prompt = """당신은 과거 피드백이 현재 쿼리 및 문서와 관련이 있는지를 판단하는 AI 시스템입니다.
    답변은 오직 'yes' 또는 'no'로만 하세요. 설명을 제공하지 말고, 오직 관련성 여부만 판단하십시오."""

    # 사용자 프롬프트: 현재 쿼리, 과거 피드백, 문서 내용을 포함
    user_prompt = f"""
    현재 쿼리(Current query): {query}
    피드백이 달렸던 과거 쿼리(Past query that received feedback): {feedback['query']}
    문서 내용(Document content): {doc_text[:500]}... [중략]
    피드백이 달린 응답(Past response that received feedback): {feedback['response'][:500]}... [중략]

    이 과거 피드백이 현재 쿼리 및 문서와 관련이 있습니까? (yes/no)
    """

    # LLM 호출: 출력의 일관성을 위해 temperature=0 사용
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 일관된 응답을 위해 temperature=0 설정
    )
    
    # 응답 텍스트를 추출 및 정규화하여 관련성 판단
    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer  # 응답에 'yes'가 포함되어 있으면 True 반환

def ch11_adjust_relevance_scores(query, results, feedback_data):
    """
    과거 피드백을 기반으로 문서의 관련성 점수를 조정하여 검색 품질을 향상시킵니다.

    이 함수는 사용자 피드백을 분석하여, 현재 질의에 대한 검색 결과의 관련성 점수를 동적으로 조정합니다.
    관련 피드백을 식별하고, 평점 기반으로 점수 보정치를 계산한 후, 결과를 재정렬합니다.

    Args:
        query (str): 현재 사용자 질의
        results (List[Dict]): 검색된 문서들과 원래의 유사도 점수
        feedback_data (List[Dict]): 사용자 피드백 기록 (relevance 평점 포함)

    Returns:
        List[Dict]: 조정된 관련성 점수로 재정렬된 검색 결과 리스트
    """
    # 피드백이 없으면 원래 결과 그대로 반환
    if not feedback_data:
        return results

    print("피드백 기록을 기반으로 관련성 점수 조정 중...")

    # 각 검색된 문서에 대해 반복
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []

        # 문서와 현재 질의 조합에 대해 관련 있는 피드백 탐색
        # LLM을 통해 각 피드백의 관련성 평가
        for feedback in feedback_data:
            is_relevant = ch11_assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)

        # 관련 피드백이 있을 경우 점수 보정 적용
        if relevant_feedback:
            # 관련 피드백의 평균 관련성 평점 계산 (1~5 점수)
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)

            # 평균 평점을 0.5~1.5 사이의 보정 계수로 변환
            # - 3점 미만: 점수 감소 (modifier < 1.0)
            # - 3점 초과: 점수 증가 (modifier > 1.0)
            modifier = 0.5 + (avg_relevance / 5.0)

            # 원래 유사도 점수에 보정 계수 적용
            original_score = result["similarity"]
            adjusted_score = original_score * modifier

            # 결과 딕셔너리 업데이트
            result["original_similarity"] = original_score   # 원래 점수 저장
            result["similarity"] = adjusted_score            # 보정된 점수로 업데이트
            result["relevance_score"] = adjusted_score       # 관련성 점수도 업데이트
            result["feedback_applied"] = True                # 피드백 반영 여부 표시
            result["feedback_count"] = len(relevant_feedback)  # 사용된 피드백 수

            # 로그 출력
            print(
                f"  문서 {i+1}: {len(relevant_feedback)}개의 피드백을 기반으로 점수 {original_score:.4f} → {adjusted_score:.4f}로 조정됨 "
                f"(Document {i+1}: Adjusted score from {original_score:.4f} to {adjusted_score:.4f} based on {len(relevant_feedback)} feedback(s))"
            )

    # 보정된 점수 기준으로 결과 재정렬
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results

def ch11_fine_tune_index(current_store, chunks, feedback_data):
    """
    고품질 피드백을 활용하여 벡터 스토어를 향상시킵니다.

    이 함수는 다음 과정을 통해 벡터 스토어의 검색 품질을 지속적으로 개선합니다:
    1. 높은 평점을 받은 Q&A 쌍을 기반으로 고품질 피드백을 식별합니다.
    2. 성공적인 상호작용을 기반으로 새로운 검색 항목을 생성합니다.
    3. 이를 가중치를 높여 벡터 스토어에 추가합니다.

    Args:
        current_store (SimpleVectorStore): 원본 문서 조각들을 포함한 현재 벡터 스토어
        chunks (List[str]): 원본 문서의 텍스트 조각 리스트
        feedback_data (List[Dict]): 관련성과 품질 평점을 포함한 사용자 피드백 기록

    Returns:
        SimpleVectorStore: 원본 조각과 피드백 기반 콘텐츠를 포함한 향상된 벡터 스토어
    """
    print("고품질 피드백으로 인덱스를 미세 조정 중...")

    # 관련성 및 품질이 모두 4 이상인 고품질 피드백만 필터링합니다.
    # 가장 성공적인 상호작용만 학습에 사용하기 위함입니다.
    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    if not good_feedback:
        print("미세 조정에 사용할 고품질 피드백이 없습니다.")
        return current_store  # 고품질 피드백이 없을 경우 기존 스토어 그대로 반환

    # 원본 및 향상된 콘텐츠를 모두 포함할 새로운 스토어를 초기화합니다.
    new_store = ch11_SimpleVectorStore()

    # 기존 문서 조각과 메타데이터를 새로운 스토어에 추가합니다.
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()  # 참조 문제를 피하기 위해 copy 사용
        )

    # 고품질 피드백을 기반으로 향상된 콘텐츠를 생성하고 추가합니다.
    for feedback in good_feedback:
        # 질문과 고품질 응답을 결합하여 새로운 문서를 생성합니다.
        # 이는 사용자의 질문에 직접 대응하는 검색 가능한 콘텐츠입니다.
        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"

        # 새로운 문서에 대해 임베딩 벡터를 생성합니다.
        embedding = ch11_create_embeddings(enhanced_text)

        # 피드백 기반 콘텐츠임을 나타내는 메타데이터와 함께 벡터 스토어에 추가합니다.
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",     # 피드백 기반 콘텐츠로 표시
                "query": feedback["query"],      # 원본 질문 저장
                "relevance_score": 1.2,          # 검색 시 우선순위를 높이기 위한 가중치
                "feedback_count": 1,             # 피드백 반영 횟수
                "original_feedback": feedback    # 전체 피드백 기록 저장
            }
        )

        print(f"피드백에서 향상된 콘텐츠 추가됨: {feedback['query'][:50]}...")

    # 향상된 벡터 스토어의 항목 수를 출력합니다.
    print(f"미세 조정된 인덱스 항목 수: {len(new_store.texts)}개 (원본: {len(chunks)}개)")
    return new_store

def ch11_generate_response(query, context, model="gpt-4o-mini"):
    """
    질의와 문맥을 기반으로 응답을 생성합니다.

    Args:
        query (str): 사용자 질의
        context (str): 검색된 문서에서 가져온 문맥 텍스트
        model (str): 사용할 LLM 모델 이름

    Returns:
        str: 생성된 응답 문자열
    """
    # LLM의 동작을 안내하는 시스템 프롬프트 정의
    system_prompt = """당신은 유용한 AI 비서입니다. 
    제공된 문맥에만 근거하여 사용자의 질문에 답변하세요. 
    문맥에서 답을 찾을 수 없는 경우 정보가 충분하지 않다고 말합니다."""
    
    # 문맥과 질의를 결합하여 사용자 프롬프트 생성
    user_prompt = f"""
        Context:
        {context}

        Question: {query}

        위의 문맥에만 근거하여 포괄적인 답변을 제공하세요.
    """
    
    # 시스템 및 사용자 프롬프트를 기반으로 OpenAI API를 호출하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 일관된 결과를 위해 temperature=0 사용
    )
    
    # 생성된 응답 텍스트 반환
    return response.choices[0].message.content

def ch11_rag_with_feedback_loop(query, vector_store, feedback_data, k=5, model="gpt-4o-mini"):
    """
    피드백 루프가 통합된 RAG 파이프라인 전체 프로세스

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 문서 청크가 저장된 벡터 저장소
        feedback_data (List[Dict]): 피드백 이력 데이터
        k (int): 검색할 문서 수
        model (str): 응답 생성을 위한 LLM 모델

    Returns:
        Dict: 질의, 검색된 문서, 생성된 응답을 포함한 결과 딕셔너리
    """
    print(f"\n***Processing query with feedback-enhanced RAG***")
    print(f"Query: {query}")

    # 1단계: 질의 임베딩 생성
    query_embedding = ch11_create_embeddings(query)

    # 2단계: 질의 임베딩을 기반으로 초기 문서 검색 수행
    results = vector_store.similarity_search(query_embedding, k=k)

    # 3단계: 검색된 문서들의 관련성 점수를 피드백 기반으로 조정
    adjusted_results = ch11_adjust_relevance_scores(query, results, feedback_data)

    # 4단계: 조정된 결과에서 텍스트만 추출하여 문맥 구성용 리스트 생성
    retrieved_texts = [result["text"] for result in adjusted_results]

    # 5단계: 검색된 텍스트들을 연결하여 응답 생성용 문맥 구성
    context = "\n\n---\n\n".join(retrieved_texts)

    # 6단계: 문맥과 질의를 기반으로 응답 생성
    print("Generating response...")
    response = ch11_generate_response(query, context, model)

    # 7단계: 최종 결과 딕셔너리 구성
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }

    print("\n***Response***")
    print(response)

    return result

def ch11_full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file=r"D:\python_workspace\FastApi\Area\Rag\feedback_data.json", fine_tune=False):
    """
    피드백 통합을 통한 RAG 전체 워크플로우 실행

    이 함수는 Retrieval-Augmented Generation(RAG)의 전체 과정을 실행하며,
    피드백 기반 벡터 인덱스 향상과 사용자 피드백 수집을 포함합니다.

    주요 단계:
    1. 이전 피드백 데이터 로드
    2. 문서 처리 및 청크 분할
    3. 고품질 Q&A를 기반으로 벡터 인덱스 향상 (선택적)
    4. 피드백 기반 관련성 보정으로 검색 및 응답 생성
    5. 사용자 피드백 수집
    6. 피드백 저장 → 지속 학습 기반 마련

    Args:
        pdf_path (str): 처리할 PDF 문서 경로
        query (str): 사용자의 자연어 질의
        feedback_data (List[Dict], optional): 사전 로드된 피드백 데이터 (없으면 파일에서 불러옴)
        feedback_file (str): 피드백 이력을 저장할 JSON 파일 경로
        fine_tune (bool): 성공적인 Q&A를 벡터 인덱스에 반영할지 여부

    Returns:
        Dict: 생성된 응답 및 검색 관련 메타데이터를 포함한 결과
    """
    # 1단계: 피드백 데이터가 전달되지 않은 경우 파일에서 불러오기
    if feedback_data is None:
        feedback_data = ch11_load_feedback_data(feedback_file)
        print(f"{feedback_file}에서 {len(feedback_data)}개의 피드백 항목을 불러왔습니다.")

    # 2단계: 문서 처리 (텍스트 추출, 청크 분할, 임베딩 및 벡터 저장소 생성)
    chunks, vector_store = ch11_process_document(pdf_path)

    # 3단계: 이전 피드백을 활용하여 벡터 인덱스 개선 (fine_tune이 True일 때만)
    if fine_tune and feedback_data:
        vector_store = ch11_fine_tune_index(vector_store, chunks, feedback_data)

    # 4단계: 피드백 기반 RAG 실행 → 관련성 보정된 검색 + 응답 생성
    result = ch11_rag_with_feedback_loop(query, vector_store, feedback_data)

    # 5단계: 사용자 피드백 수집 (콘솔 입력)
    print("\n***이번 응답에 대한 피드백을 제공하시겠습니까?***")
    print("관련성 점수 (1~5, 5점이 가장 관련 있음):")
    relevance = input()

    print("품질 점수 (1~5, 5점이 가장 우수함):")
    quality = input()

    print("기타 코멘트가 있다면 입력해주세요 (건너뛰려면 Enter):")
    comments = input()

    # 6단계: 피드백 객체 구성
    feedback = ch11_get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )

    # 7단계: 피드백 저장 (지속적인 향후 학습을 위한 기록)
    ch11_store_feedback(feedback, feedback_file)
    print("피드백이 저장되었습니다. 감사합니다!")

    return result

def ch11_calculate_similarity(text1, text2):
    """
    두 텍스트 간 의미적 유사도를 임베딩 기반으로 계산합니다.

    Args:
        text1 (str): 첫 번째 텍스트
        text2 (str): 두 번째 텍스트

    Returns:
        float: 0과 1 사이의 유사도 점수 (1에 가까울수록 유사함)
    """
    # 두 텍스트에 대한 임베딩 생성
    embedding1 = ch11_create_embeddings(text1)
    embedding2 = ch11_create_embeddings(text2)
    
    # 임베딩을 NumPy 배열로 변환
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # 코사인 유사도 계산: 내적 / (벡터 크기의 곱)
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return similarity

def ch11_compare_results(queries, round1_results, round2_results, reference_answers=None):
    """
    두 차례의 RAG 실행 결과를 비교 분석합니다.

    Args:
        queries (List[str]): 테스트 질의 목록
        round1_results (List[Dict]): 라운드 1 결과 (피드백 미사용)
        round2_results (List[Dict]): 라운드 2 결과 (피드백 반영)
        reference_answers (List[str], optional): 참조 정답 (있을 경우 평가에 포함)

    Returns:
        str: 각 질의별 비교 분석 결과 목록
    """
    print("\n=== COMPARING RESULTS ===")

    # 시스템 프롬프트: LLM에게 비교 기준을 안내
    system_prompt = """당신은 RAG 시스템의 전문 평가자입니다. 두 버전의 응답을 비교하세요:
    1. 표준 RAG: 피드백 사용 안 함
    2. 피드백 강화 RAG: 피드백 루프를 사용하여 검색을 개선합니다.

    어떤 버전이 다음과 같은 측면에서 더 나은 응답을 제공하는지 분석하세요:
    - 쿼리와의 관련성
    - 정보의 정확성
    - 완전성
    - 명확성 및 간결성
    """

    comparisons = []

    # 각 질의에 대해 두 라운드의 응답 비교
    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):
        # 비교용 사용자 프롬프트 구성
        comparison_prompt = f"""
        Query: {query}

        Standard RAG Response:
        {r1["response"]}

        Feedback-enhanced RAG Response:
        {r2["response"]}
        """

        # 참조 정답이 있다면 포함
        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"""
            Reference Answer:
            {reference_answers[i]}
            """

        comparison_prompt += """
        두 응답을 비교하고 어떤 응답이 더 나은지 그 이유를 설명하세요.
        특히 피드백 루프가 응답 품질을 어떻게 개선했는지(또는 개선하지 않았는지)에 초점을 맞춰 설명하세요.
        """

        # LLM을 통해 비교 분석 생성
        _client = openai
        _client.api_key = util.getEnv('openai_api_key')          
        response = _client.chat.completions.create(        
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )

        # 분석 결과 저장
        comparisons.append({
            "query": query,
            "analysis": response.choices[0].message.content
        })

        # 분석 요약 출력
        print(f"\nQuery {i+1}: {query}")
        print(f"Analysis: {response.choices[0].message.content[:200]}...")

    return comparisons

def ch11_evaluate_feedback_loop(pdf_path, test_queries, reference_answers=None):
    """
    피드백 루프가 RAG 품질에 미치는 영향을 평가합니다.

    이 함수는 피드백 통합 전후의 성능을 비교하는 통제 실험을 수행합니다.
    다음과 같은 절차로 구성됩니다:
    1. 라운드 1: 피드백 없이 질의 실행 → 기준선(Baseline) 성능 측정
    2. 참조 답변이 있는 경우, 이를 기반으로 synthetic feedback 생성
    3. 라운드 2: 이전 피드백을 반영하여 동일 질의 재실행
    4. 결과를 비교하여 피드백 기반 개선의 정량적 효과 분석

    Args:
        pdf_path (str): 지식 베이스로 사용할 PDF 문서 경로
        test_queries (List[str]): 성능 평가용 테스트 질의 리스트
        reference_answers (List[str], optional): 정답 또는 참조 응답 리스트 (synthetic feedback 생성 및 비교용)

    Returns:
        Dict: 다음 정보를 포함하는 평가 결과
            - round1_results: 피드백 없이 실행한 응답 결과 목록
            - round2_results: 피드백을 반영한 응답 결과 목록
            - comparison: 두 라운드 간의 정량적 비교 결과
    """
    print("***피드백 루프 성능 평가 시작***")

    # 피드백을 임시 저장할 JSON 파일 경로 (세션 내 임시 용도)
    temp_feedback_file = r"D:\python_workspace\FastApi\Area\Rag\temp_evaluation_feedback.json"

    # 초기 피드백 리스트 (라운드 1에서는 없음)
    feedback_data = []

    # ----------------------- 라운드 1: 피드백 없이 -----------------------
    print("\n***라운드 1: 피드백 없음***")
    round1_results = []

    for i, query in enumerate(test_queries):
        print(f"\n질의 {i+1}: {query}")

        # 문서 처리 및 벡터 저장소 생성
        chunks, vector_store = ch11_process_document(pdf_path)

        # 피드백 없이 RAG 실행
        result = ch11_rag_with_feedback_loop(query, vector_store, [])
        round1_results.append(result)

        # 참조 답변이 있다면 → 이를 기반으로 synthetic feedback 생성
        if reference_answers and i < len(reference_answers):
            similarity_to_ref = ch11_calculate_similarity(result["response"], reference_answers[i])
            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))

            feedback = ch11_get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"참조 답변 유사도 기반 synthetic feedback: {similarity_to_ref:.2f}"
            )

            # 피드백을 메모리와 파일에 저장
            feedback_data.append(feedback)
            ch11_store_feedback(feedback, temp_feedback_file)

    # ----------------------- 라운드 2: 피드백 반영 -----------------------
    print("\n***라운드 2: 피드백 반영***")
    round2_results = []

    # 문서를 재처리하고 피드백 기반 인덱스 향상
    chunks, vector_store = ch11_process_document(pdf_path)
    vector_store = ch11_fine_tune_index(vector_store, chunks, feedback_data)

    for i, query in enumerate(test_queries):
        print(f"\n질의 {i+1}: {query}")

        # 피드백 반영된 RAG 실행
        result = ch11_rag_with_feedback_loop(query, vector_store, feedback_data)
        round2_results.append(result)

    # ----------------------- 결과 비교 -----------------------
    comparison = ch11_compare_results(test_queries, round1_results, round2_results, reference_answers)

    # 임시 피드백 파일 삭제 (실험 종료 후 정리)
    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)

    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }
#-----------------ch11 피드백 루프 Feedback Loop RAG : end

#-----------------ch12 적응형검색 adaptive Retrieval RAG : start
class ch12_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 구현체입니다.
    """
    def __init__(self):
        """
        벡터 저장소 초기화
        """
        self.vectors = []     # 임베딩 벡터 리스트
        self.texts = []       # 원본 텍스트 리스트
        self.metadata = []    # 각 텍스트의 메타데이터 리스트

    def add_item(self, text, embedding, metadata=None):
        """
        단일 텍스트 항목을 벡터 저장소에 추가합니다.

        Args:
            text (str): 원본 텍스트
            embedding (List[float]): 텍스트의 임베딩 벡터
            metadata (dict, optional): 추가 메타데이터 (기본값: 빈 딕셔너리)
        """
        self.vectors.append(np.array(embedding))  # 임베딩을 NumPy 배열로 변환하여 저장
        self.texts.append(text)                   # 원본 텍스트 저장
        self.metadata.append(metadata or {})      # 메타데이터 저장 (None일 경우 빈 딕셔너리)

    def similarity_search(self, query_embedding, k=5):
        """
        질의 임베딩과 가장 유사한 텍스트를 검색합니다.

        Args:
            query_embedding (List[float]): 질의 임베딩 벡터
            k (int): 반환할 상위 결과 개수

        Returns:
            List[Dict]: 유사한 항목 리스트 (텍스트, 메타데이터, 유사도 포함)
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없으면 빈 리스트 반환

        # 질의 벡터를 NumPy 배열로 변환
        query_vector = np.array(query_embedding)

        # 각 벡터와의 코사인 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))  # 인덱스와 유사도 점수 저장

        # 유사도 기준 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 항목 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results
    
def ch12_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text
def ch12_chunk_text(text, n, overlap):
    """
    주어진 텍스트를 n자 단위로 분할하되, 각 청크 간에 overlap만큼 겹치게 합니다.

    Args:
        text (str): 분할할 텍스트
        n (int): 각 청크의 문자 수 (기본값: 1000)
        overlap (int): 청크 간 겹치는 문자 수 (기본값: 200)

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    chunks = []  # 청크들을 저장할 리스트 초기화

    # n - overlap 만큼 이동하면서 청크 생성
    for i in range(0, len(text), n - overlap):
        # i에서 i + n까지의 텍스트 조각을 청크로 추가
        chunks.append(text[i:i + n])

    return chunks  # 생성된 청크 리스트 반환

def ch12_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 임베딩을 생성합니다.

    Args:
        text (str 또는 List[str]): 임베딩을 생성할 입력 텍스트(또는 텍스트 리스트)
        model (str): 사용할 임베딩 모델 이름

    Returns:
        List[float] 또는 List[List[float]]: 생성된 임베딩 벡터 또는 벡터 리스트
    """
    # 입력이 문자열 하나일 수도 있고, 문자열 리스트일 수도 있으므로 리스트 형태로 통일
    input_text = text if isinstance(text, list) else [text]

    # 지정된 모델을 사용하여 임베딩 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(           
        model=model,
        input=input_text
    )

    # 입력이 단일 문자열이었을 경우, 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding

    # 여러 문자열일 경우, 모든 임베딩 리스트 반환
    return [item.embedding for item in response.data]

def ch12_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    적응형 검색을 위한 문서 처리 함수

    Args:
        pdf_path (str): 처리할 PDF 파일 경로
        chunk_size (int): 각 청크의 문자 수
        chunk_overlap (int): 청크 간 겹치는 문자 수

    Returns:
        Tuple[List[str], SimpleVectorStore]: 텍스트 청크 리스트와 벡터 저장소
    """
    # PDF 파일에서 텍스트 추출
    print("PDF에서 텍스트 추출 중...")
    extracted_text = ch12_extract_text_from_pdf(pdf_path)
    
    # 텍스트를 일정 길이로 청크 분할
    print("텍스트를 청크 단위로 분할 중...")
    chunks = ch12_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"{len(chunks)}개의 텍스트 청크 생성 완료")

    # 각 청크에 대해 임베딩 생성
    print("청크에 대한 임베딩 생성 중...")
    chunk_embeddings = ch12_create_embeddings(chunks)

    # 벡터 저장소 초기화
    store = ch12_SimpleVectorStore()

    # 각 청크와 임베딩, 메타데이터를 저장소에 추가
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"총 {len(chunks)}개의 청크가 벡터 저장소에 추가됨")

    # 텍스트 청크와 벡터 저장소 반환
    return chunks, store

def ch12_classify_query(query, model="gpt-4o-mini"):
    """
    사용자의 질의를 다음 네 가지 중 하나로 분류합니다: Factual, Analytical, Opinion, Contextual

    Args:
        query (str): 사용자 질의
        model (str): 사용할 LLM 모델

    Returns:
        str: 분류된 질의 유형
    """
    # 시스템 프롬프트: LLM에게 분류 기준과 출력 형식을 안내
    system_prompt = """귀하는 질문을 분류하는 전문가입니다.
    주어진 쿼리를 다음 카테고리 중 정확히 한 가지로 분류하세요:
    - 사실: 구체적이고 검증 가능한 정보를 찾는 쿼리.
    - 분석적: 종합적인 분석이나 설명이 필요한 쿼리.
    - 의견: 주관적인 사안에 대한 질의 또는 다양한 관점을 추구하는 질의.
    - 컨텍스트: 사용자별 컨텍스트에 따라 달라지는 쿼리.

    설명이나 추가 텍스트 없이 카테고리 이름만 반환합니다.
    """

    # 사용자 질의를 포함한 프롬프트 구성
    user_prompt = f"Classify this query: {query}"
    
    # LLM을 호출하여 질의 분류 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 분류 결과 추출 및 정제
    category = response.choices[0].message.content.strip()
    
    # 유효한 분류 카테고리 정의
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    
    # 응답이 유효한 카테고리 중 하나에 포함되는지 확인
    for valid in valid_categories:
        if valid in category:
            return valid
    
    # 분류 실패 시 기본값으로 "Factual" 반환
    return "Factual"

def ch12_factual_retrieval_strategy(query, vector_store, k=4):
    """
    사실 기반 질의에 적합한 검색 전략 (정확도 중심)

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 벡터 저장소
        k (int): 반환할 문서 수

    Returns:
        List[Dict]: 검색된 문서 목록
    """
    print(f"Executing Factual retrieval strategy for: '{query}'")
    
    # 질의 정밀도를 높이기 위한 LLM 기반 질의 개선 프롬프트 정의
    system_prompt = """귀하는 검색 쿼리를 개선하는 전문가입니다.
    귀하의 임무는 주어진 사실 쿼리를 재구성하여 정보 검색을 위해 더 정확하고
    정보 검색을 위해 구체화하는 것입니다. 주요 엔터티와 그 관계에 집중하세요.

    설명 없이 개선된 쿼리만 제공하세요.
    """

    user_prompt = f"Enhance this factual query: {query}"
    
    # LLM을 통해 개선된 질의 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 개선된 질의 추출 및 출력
    enhanced_query = response.choices[0].message.content.strip()
    print(f"Enhanced query: {enhanced_query}")
    
    # 개선된 질의에 대해 임베딩 생성
    query_embedding = ch12_create_embeddings(enhanced_query)
    
    # 유사도 검색을 통해 후보 문서 검색 (후보 수는 2배로 확장)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    
    # 문서별 관련성 평가 결과 저장용 리스트 초기화
    ranked_results = []
    
    # LLM을 통해 각 문서의 질의에 대한 관련성 점수 산정
    for doc in initial_results:
        relevance_score = ch12_score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    
    # 관련성 점수를 기준으로 결과 정렬 (내림차순)
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # 상위 k개 문서 반환
    return ranked_results[:k]

def ch12_analytical_retrieval_strategy(query, vector_store, k=4):
    """
    분석형(Analytical) 질의에 대한 검색 전략: 주제 전반을 포괄하는 정보 중심

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 벡터 저장소
        k (int): 반환할 문서 수

    Returns:
        List[Dict]: 검색된 문서 목록
    """
    print(f"Executing Analytical retrieval strategy for: '{query}'")
    
    # LLM이 복잡한 질의를 하위 질문으로 분해하도록 유도하는 시스템 프롬프트
    system_prompt = """귀하는 복잡한 질문을 세분화하는 데 전문가입니다.
    기본 분석 쿼리의 다양한 측면을 탐구하는 하위 질문을 생성하세요.
    이러한 하위 질문은 주제를 폭넓게 다루어야 하고
    포괄적인 정보를 검색하는 데 도움이 되어야 합니다.

    정확히 3개의 하위 질문 목록을 한 줄에 하나씩 반환합니다.
    """

    # 사용자 질의를 포함한 프롬프트 생성
    user_prompt = f"이 분석 쿼리에 대한 하위 질문 생성하기: {query}"
    
    # LLM을 통해 하위 질문(sub-questions) 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # 응답에서 하위 질문 추출 및 정리
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"Generated sub-queries: {sub_queries}")
    
    # 각 하위 질문에 대해 문서 검색 수행
    all_results = []
    for sub_query in sub_queries:
        sub_query_embedding = ch12_create_embeddings(sub_query)
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)
    
    # 중복 문서 제거 및 다양한 출처 확보
    unique_texts = set()
    diverse_results = []
    
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    
    # 필요한 문서 수가 부족할 경우, 원래 질의로 직접 검색하여 보완
    if len(diverse_results) < k:
        main_query_embedding = ch12_create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)
        
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    
    # 상위 k개의 결과만 반환
    return diverse_results[:k]

def ch12_opinion_retrieval_strategy(query, vector_store, k=4):
    """
    의견형 질의에 대한 검색 전략
    다양한 관점을 중심으로 정보를 수집하여 응답의 균형성과 깊이를 높이는 목적

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 벡터 저장소
        k (int): 반환할 문서 수

    Returns:
        List[Dict]: 검색된 문서 목록
    """
    print(f"의견형 검색 전략 실행 중: '{query}'")

    # LLM이 다양한 관점을 생성하도록 유도하는 시스템 프롬프트
    system_prompt = """귀하는 한 주제에 대한 다양한 관점을 식별하는 데 전문가입니다.
    의견이나 관점에 대한 주어진 쿼리에 대해 사람들이 이 주제에 대해 가질 수 있는 다양한 관점을 식별하세요.
    정확히 3개의 서로 다른 관점을 한 줄에 하나씩 반환하세요.
    """

    # 사용자 질의를 포함한 프롬프트 구성
    user_prompt = f"Identify different perspectives on: {query}"

    # LLM을 통해 관점 목록 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    # 응답에서 관점 목록 추출 및 정제
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"도출된 관점 목록: {viewpoints}")

    # 각 관점에 대해 문서 검색 수행
    all_results = []
    for viewpoint in viewpoints:
        # 원래 질의에 관점을 결합하여 쿼리를 강화
        combined_query = f"{query} {viewpoint}"
        viewpoint_embedding = ch12_create_embeddings(combined_query)
        results = vector_store.similarity_search(viewpoint_embedding, k=2)

        # 검색 결과에 해당 관점 정보 포함
        for result in results:
            result["viewpoint"] = viewpoint

        all_results.extend(results)

    # 관점별로 하나씩 대표 문서를 선정
    selected_results = []
    for viewpoint in viewpoints:
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])

    # 부족한 문서 수는 유사도 순으로 추가 확보
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    # 최종 k개의 문서 반환
    return selected_results[:k]

def ch12_contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    컨텍스트 기반 질의에 대한 검색 전략
    사용자 맥락을 통합하여 검색 관련성을 향상시키는 방식

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 벡터 저장소
        k (int): 반환할 문서 수
        user_context (str): 명시적 또는 질의로부터 추론된 사용자 맥락 정보

    Returns:
        List[Dict]: 검색된 문서 목록
    """
    print(f"컨텍스트 기반 검색 전략 실행 중: '{query}'")

    # 사용자 맥락이 명시되지 않은 경우, LLM을 통해 질의로부터 추론
    if not user_context:
        system_prompt = """귀하는 질문에 내포된 문맥을 이해하는 데 전문가입니다.
        주어진 쿼리에 대해 어떤 문맥 정보가 관련되거나 암시되는지 추론하세요.
        이 쿼리에 답하는 데 도움이 될 수 있는 배경 정보를 중심으로 작성하세요.
        간결한 문장 하나로 암시된 컨텍스트를 반환하세요."""

        user_prompt = f"이 쿼리에서 암시된 컨텍스트 추론하기: {query}"

        # LLM에게 컨텍스트 추론 요청
        _client = openai
        _client.api_key = util.getEnv('openai_api_key')          
        response = _client.chat.completions.create(        
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        # 추론된 사용자 맥락 추출
        user_context = response.choices[0].message.content.strip()
        print(f"추론된 컨텍스트: {user_context}")

    # 질의에 사용자 맥락을 통합하여 질의를 재구성
    system_prompt = """귀하는 문맥에 맞게 질문을 재구성하는 데 전문가입니다.
    쿼리와 주어진 컨텍스트 정보를 기반으로 보다 구체적이고 명확한 질의로 바꿔주세요.
    설명 없이 재작성된 쿼리만 출력하세요."""

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    이 컨텍스트를 반영하여 쿼리를 다시 작성하세요."""

    # LLM에게 질의 재작성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 재작성된 컨텍스트 기반 질의 추출
    contextualized_query = response.choices[0].message.content.strip()
    print(f"컨텍스트 반영 질의: {contextualized_query}")

    # 재작성된 질의에 대한 임베딩 생성 및 문서 검색 수행
    query_embedding = ch12_create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)

    # 검색된 문서들에 대해 문맥 관련성 평가
    ranked_results = []
    for doc in initial_results:
        context_relevance = ch12_score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })

    # 맥락 관련성 기준으로 정렬 후 상위 k개 문서 반환
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]

def ch12_score_document_relevance(query, document, model="gpt-4o-mini"):
    """
    LLM을 사용하여 문서의 질의에 대한 관련성을 평가합니다.

    Args:
        query (str): 사용자 질의
        document (str): 평가할 문서 텍스트
        model (str): 사용할 LLM 모델

    Returns:
        float: 0~10 사이의 관련성 점수
    """
    # LLM에게 문서 관련성 평가 방법을 안내하는 시스템 프롬프트
    system_prompt = """귀하는 문서 관련성을 평가하는 전문가입니다.
    문서와 쿼리의 관련성을 0에서 10까지의 척도로 평가하세요:
    0 = 전혀 관련 없음
    10 = 쿼리를 완벽하게 해결

    0에서 10 사이의 숫자 점수만 반환하고 그 외에는 반환하지 않습니다.
    """

    # 문서가 너무 길 경우 미리보기로 잘라내기
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # 사용자 프롬프트 구성: 질의와 문서 제공 후 관련성 점수 요청
    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """
    
    # LLM 호출을 통해 관련성 점수 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 응답으로부터 점수 텍스트 추출
    score_text = response.choices[0].message.content.strip()
    
    # 정규표현식을 이용하여 숫자 추출
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 점수를 0~10 범위로 제한
    else:
        # 점수 추출 실패 시 기본값 반환
        return 5.0
    
def ch12_score_document_context_relevance(query, context, document, model="gpt-4o-mini"):
    """
    질의와 사용자 맥락을 함께 고려하여 문서의 관련성을 평가합니다.

    Args:
        query (str): 사용자 질의
        context (str): 사용자 맥락 정보
        document (str): 평가할 문서 텍스트
        model (str): 사용할 LLM 모델

    Returns:
        float: 0~10 사이의 관련성 점수
    """
    # LLM에게 질의 + 맥락 기반으로 문서 관련성 평가 방법을 안내하는 시스템 프롬프트
    system_prompt = """귀하는 문맥을 고려하여 문서 관련성을 평가하는 전문가입니다.
    문서가 쿼리를 얼마나 잘 처리하는지에 따라 0에서 10까지의 척도로 평가하세요.
    쿼리를 얼마나 잘 처리하는지에 따라 0에서 10까지 평가합니다:
    0 = 전혀 관련 없음
    10 = 주어진 맥락에서 쿼리를 완벽하게 해결함

    0에서 10 사이의 숫자 점수만 반환하고 다른 점수는 반환하지 않습니다.
    """

    # 문서가 너무 길 경우, 앞부분만 사용하여 평가하도록 자름
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # 사용자 프롬프트 구성: 질의, 맥락, 문서를 함께 제공하고 점수 요청
    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    컨텍스트를 고려한 관련성 점수(0~10):
    """
    
    # LLM을 호출하여 점수 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # 응답으로부터 점수 텍스트 추출
    score_text = response.choices[0].message.content.strip()
    
    # 정규표현식을 이용하여 숫자 추출
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 점수를 0~10 범위로 제한
    else:
        # 점수 추출 실패 시 기본값 반환
        return 5.0

def ch12_adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    질의 유형에 따라 적절한 검색 전략을 자동으로 선택하여 실행하는 적응형 검색 함수

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 벡터 저장소
        k (int): 검색할 문서 수
        user_context (str): 사용자 맥락 정보 (컨텍스트 기반 질의에 사용)

    Returns:
        List[Dict]: 검색된 문서 리스트
    """
    # 질의 유형 분류 (Factual, Analytical, Opinion, Contextual 중 하나)
    query_type = ch12_classify_query(query)
    print(f"분류된 질의 유형: {query_type}")

    # 질의 유형에 따라 대응하는 검색 전략 실행
    if query_type == "Factual":
        # 사실 기반 정보 검색
        results = ch12_factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        # 분석적 사고나 복합적 개념이 필요한 경우
        results = ch12_analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        # 다양한 관점이나 의견을 요구하는 질의
        results = ch12_opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        # 사용자의 맥락 정보가 중요한 질의
        results = ch12_contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # 분류되지 않은 경우, 기본적으로 사실 기반 검색 수행
        results = ch12_factual_retrieval_strategy(query, vector_store, k)

    # 최종 검색 결과 반환
    return results

def ch12_generate_response(query, results, query_type, model="gpt-4o-mini"):
    """
    질의 유형과 검색된 문서를 기반으로 적절한 응답을 생성하는 함수

    Args:
        query (str): 사용자 질의
        results (List[Dict]): 검색된 문서 목록
        query_type (str): 질의 유형 (Factual, Analytical, Opinion, Contextual 등)
        model (str): 사용할 LLM 모델

    Returns:
        str: 생성된 응답 텍스트
    """
    # 검색된 문서 내용을 하나의 context로 구성
    context = "\n\n---\n\n".join([r["text"] for r in results])

    # 질의 유형에 따라 시스템 프롬프트 설정
    if query_type == "Factual":
        system_prompt = """귀하는 사실 기반 정보를 제공하는 조력자입니다.
        제공된 문맥을 바탕으로 질문에 정확하고 간결하게 답하세요.
        문맥에 정보가 부족하다면 그 한계를 명확히 밝혀주세요."""

    elif query_type == "Analytical":
        system_prompt = """귀하는 분석적 설명을 제공하는 조력자입니다.
        문맥을 종합적으로 분석하고 주제의 여러 측면을 다루어 주세요.
        정보 간 충돌이 있다면 이를 인정하고 균형 있게 설명하세요."""

    elif query_type == "Opinion":
        system_prompt = """귀하는 다양한 관점을 제시하는 조력자입니다.
        주제에 대한 여러 시각을 공정하게 나열하고 편향되지 않게 설명하세요.
        관점이 제한적일 경우 그 점도 함께 알려주세요."""

    elif query_type == "Contextual":
        system_prompt = """귀하는 문맥에 민감하게 반응하는 조력자입니다.
        질문과 문맥을 모두 고려하여 연관성 높은 응답을 제공합니다.
        문맥이 부족하거나 불완전한 경우 그 한계를 분명히 하세요."""

    else:
        system_prompt = """귀하는 유용한 조력자입니다. 문맥을 바탕으로 질문에 답하고, 문맥이 부족할 경우 명확하게 밝혀주세요."""

    # 사용자 프롬프트 구성
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    위 문맥을 바탕으로 질문에 적절한 응답을 작성하세요.
    """

    # LLM 호출을 통해 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 최종 응답 반환
    return response.choices[0].message.content

def ch12_rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    적응형 검색을 활용한 RAG(Retrieval-Augmented Generation) 파이프라인 전체 실행 함수

    Args:
        pdf_path (str): 처리할 PDF 문서 경로
        query (str): 사용자 질의
        k (int): 검색할 문서 수
        user_context (str): (선택 사항) 사용자 맥락 정보

    Returns:
        Dict: 질의, 질의 유형, 검색된 문서, 응답을 포함한 결과 딕셔너리
    """
    print("\n***RAG WITH ADAPTIVE RETRIEVAL***")
    print(f"Query: {query}")
    
    # 1단계: PDF 문서에서 텍스트 추출, 청크 분할, 임베딩 생성
    chunks, vector_store = ch12_process_document(pdf_path)
    
    # 2단계: 질의 유형 분류 (Factual, Analytical, Opinion, Contextual)
    query_type = ch12_classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # 3단계: 분류된 유형에 따라 적절한 검색 전략 실행
    retrieved_docs = ch12_adaptive_retrieval(query, vector_store, k, user_context)
    
    # 4단계: 질의, 검색 문서, 질의 유형을 기반으로 응답 생성
    response = ch12_generate_response(query, retrieved_docs, query_type)
    
    # 5단계: 결과 딕셔너리 구성
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
    
    print("\n***RESPONSE***")
    print(response)
    
    return result

def ch12_evaluate_adaptive_vs_standard(pdf_path, test_queries, reference_answers=None):
    """
    적응형 검색과 표준 검색을 테스트 질의에 대해 비교 평가하는 함수

    이 함수는 다음의 과정을 수행한다:
    - 문서를 전처리하여 텍스트 청크 및 벡터 저장소 생성
    - 각 질의에 대해 표준 검색과 적응형 검색을 모두 수행
    - 참조 정답이 제공된 경우, 생성된 응답과 비교하여 품질을 평가

    Args:
        pdf_path (str): 지식 기반으로 사용할 PDF 문서 경로
        test_queries (List[str]): 테스트용 질의 리스트
        reference_answers (List[str], optional): 응답 평가를 위한 참조 정답 리스트

    Returns:
        Dict: 각 질의에 대한 검색 및 응답 결과, 평가 점수를 포함한 딕셔너리
    """
    print("적응형 검색 vs. 표준 검색 성능 평가 시작")

    # 문서 처리: 텍스트 추출, 청크 분할, 벡터 저장소 구축
    chunks, vector_store = ch12_process_document(pdf_path)

    # 결과 저장용 리스트 초기화
    results = []

    # 각 테스트 질의에 대해 검색 및 응답 생성 수행
    for i, query in enumerate(test_queries):
        print(f"\n질의 {i+1}: {query}")

        # 표준 검색 수행
        print("\n표준 검색 실행 중")
        query_embedding = ch12_create_embeddings(query)
        standard_docs = vector_store.similarity_search(query_embedding, k=4)
        standard_response = ch12_generate_response(query, standard_docs, "General")

        # 적응형 검색 수행
        print("\n적응형 검색 실행 중")
        query_type = ch12_classify_query(query)
        adaptive_docs = ch12_adaptive_retrieval(query, vector_store, k=4)
        adaptive_response = ch12_generate_response(query, adaptive_docs, query_type)

        # 결과 저장
        result = {
            "query": query,
            "query_type": query_type,
            "standard_retrieval": {
                "documents": standard_docs,
                "response": standard_response
            },
            "adaptive_retrieval": {
                "documents": adaptive_docs,
                "response": adaptive_response
            }
        }

        # 참조 정답이 존재할 경우 함께 저장
        if reference_answers and i < len(reference_answers):
            result["reference_answer"] = reference_answers[i]

        results.append(result)

        # 응답 미리보기 출력
        print("\n응답 비교")
        print(f"표준 검색 응답: {standard_response[:200]}...")
        print(f"적응형 검색 응답: {adaptive_response[:200]}...")

    # 참조 정답이 있는 경우 응답 품질 비교 평가 수행
    if reference_answers:
        comparison = ch12_compare_responses(results)
        print("\n응답 비교 평가 결과")
        print(comparison)
    else:
        comparison = "참조 정답이 없어 평가 생략됨"

    # 최종 결과 반환
    return {
        "results": results,
        "comparison": comparison
    }

def ch12_compare_responses(results):
    """
    표준 검색과 적응형 검색의 응답을 참조 정답과 비교하여 분석하는 함수

    Args:
        results (List[Dict]): 표준 및 적응형 응답이 포함된 결과 리스트

    Returns:
        str: 응답 비교 분석 결과 텍스트
    """
    # AI 모델이 응답 비교를 수행할 수 있도록 시스템 프롬프트 정의
    comparison_prompt = """귀하는 정보 검색 시스템의 전문 평가자입니다.
    각 쿼리에 대한 표준 검색 응답과 적응형 검색 응답을 비교하세요.
    정확성, 관련성, 포괄성, 참조 답변과의 일치성 등을 기준으로 분석하세요.
    각 방식의 강점과 약점에 대해 한글로 자세히 설명하세요."""

    # 분석 결과를 담을 텍스트 초기화
    comparison_text = "표준 검색 vs 적응형 검색 응답 평가\n\n"

    # 각 질의 결과에 대해 비교 수행
    for i, result in enumerate(results):
        # 참조 정답이 없는 경우는 평가 생략
        if "reference_answer" not in result:
            continue

        # 질의 정보 기록
        comparison_text += f"질의 {i+1}: {result['query']}\n"
        comparison_text += f"(질의 유형: {result['query_type']})\n\n"
        comparison_text += f"[참조 정답]\n{result['reference_answer']}\n\n"

        # 표준 검색 응답 기록
        comparison_text += f"[표준 검색 응답]\n{result['standard_retrieval']['response']}\n\n"

        # 적응형 검색 응답 기록
        comparison_text += f"[적응형 검색 응답]\n{result['adaptive_retrieval']['response']}\n\n"

        # 사용자 프롬프트 구성
        user_prompt = f"""
        Reference Answer: {result['reference_answer']}

        Standard Retrieval Response: {result['standard_retrieval']['response']}

        Adaptive Retrieval Response: {result['adaptive_retrieval']['response']}

        두 응답을 비교 분석해주세요.
        """

        # AI 모델을 사용해 비교 분석 생성
        _client = openai
        _client.api_key = util.getEnv('openai_api_key')          
        response = _client.chat.completions.create(        
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": comparison_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        # 분석 결과 추가
        comparison_text += f"[비교 분석 결과]\n{response.choices[0].message.content}\n\n"

    # 전체 비교 분석 결과 반환
    return comparison_text
#-----------------ch12 적응형검색 adaptive Retrieval RAG : end

#-----------------ch13 Self-RAG : start
class ch13_SimpleVectorStore:
    """
    NumPy를 활용한 간단한 벡터 저장소 클래스
    텍스트, 임베딩, 메타데이터를 함께 관리하고 유사도 기반 검색을 지원
    """
    def __init__(self):
        """
        벡터 저장소 초기화
        """
        self.vectors = []   # 임베딩 벡터들을 저장하는 리스트
        self.texts = []     # 원본 텍스트를 저장하는 리스트
        self.metadata = []  # 각 텍스트에 대한 메타데이터를 저장하는 리스트

    def add_item(self, text, embedding, metadata=None):
        """
        벡터 저장소에 새로운 항목을 추가

        Args:
            text (str): 원본 텍스트
            embedding (List[float]): 해당 텍스트의 임베딩 벡터
            metadata (dict, optional): 텍스트와 관련된 부가 정보
        """
        self.vectors.append(np.array(embedding))        # 임베딩을 NumPy 배열로 변환하여 저장
        self.texts.append(text)                         # 원본 텍스트 저장
        self.metadata.append(metadata or {})            # 메타데이터가 없으면 빈 딕셔너리 저장

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        주어진 질의 임베딩과 가장 유사한 항목 k개를 검색

        Args:
            query_embedding (List[float]): 질의 임베딩 벡터
            k (int): 반환할 유사 항목 수
            filter_func (callable, optional): 메타데이터 필터링 함수

        Returns:
            List[Dict]: 유사도가 높은 상위 k개의 항목 리스트 (텍스트, 메타데이터, 유사도 포함)
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없으면 빈 리스트 반환

        # 질의 임베딩을 NumPy 배열로 변환
        query_vector = np.array(query_embedding)

        # 코사인 유사도를 기반으로 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 필터 함수가 있는 경우, 조건을 만족하지 않으면 건너뜀
            if filter_func and not filter_func(self.metadata[i]):
                continue

            # 코사인 유사도 계산
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 인덱스와 유사도 저장

        # 유사도 기준으로 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개의 결과를 구성하여 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],          # 관련 텍스트
                "metadata": self.metadata[idx],   # 관련 메타데이터
                "similarity": score               # 유사도 점수
            })

        return results  # 유사 항목 리스트 반환

def ch13_extract_text_from_pdf(pdf_path):
    """
    PDF 파일로부터 텍스트를 추출하는 함수

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # PDF의 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 현재 페이지 가져오기
        text = page.get_text("text")  # 해당 페이지에서 텍스트 추출
        all_text += text  # 추출한 텍스트를 누적

    # 최종적으로 전체 텍스트 반환
    return all_text

def ch13_chunk_text(text, n, overlap):
    """
    주어진 텍스트를 일정 길이로 나누되, 일부 겹치는 부분을 포함하여 청크로 분할하는 함수

    Args:
        text (str): 분할할 원본 텍스트
        n (int): 각 청크의 문자 수
        overlap (int): 청크 간 겹치는 문자 수

    Returns:
        List[str]: 분할된 텍스트 청크 리스트
    """
    chunks = []  # 분할된 청크들을 저장할 리스트 초기화

    # 시작 인덱스를 (n - overlap) 간격으로 이동하면서 청크 생성
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]  # i부터 i+n까지의 텍스트를 하나의 청크로 추출
        chunks.append(chunk)  # 추출한 청크를 리스트에 추가

    # 생성된 청크 리스트 반환
    return chunks

def ch13_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 임베딩을 생성하는 함수

    Args:
        text (str 또는 List[str]): 임베딩을 생성할 입력 텍스트 또는 텍스트 리스트
        model (str): 임베딩 생성을 위한 모델 이름

    Returns:
        List[float] 또는 List[List[float]]: 생성된 임베딩 벡터 또는 벡터 리스트
    """
    # 입력이 문자열이면 리스트로 변환하여 처리
    input_text = text if isinstance(text, list) else [text]

    # 지정한 모델을 사용하여 임베딩 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(           
        model=model,
        input=input_text
    )

    # 단일 문자열 입력인 경우 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding

    # 리스트 입력인 경우 모든 임베딩 반환
    return [item.embedding for item in response.data]

def ch13_process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Self-RAG을 위한 문서 처리 함수

    Args:
        pdf_path (str): PDF 파일 경로
        chunk_size (int): 각 청크의 문자 수
        chunk_overlap (int): 청크 간 겹치는 문자 수

    Returns:
        SimpleVectorStore: 문서 청크와 임베딩을 포함하는 벡터 저장소
    """
    # PDF 파일에서 텍스트 추출
    print("PDF에서 텍스트 추출 중...")
    extracted_text = ch13_extract_text_from_pdf(pdf_path)

    # 추출된 텍스트를 일정 길이로 분할
    print("텍스트 청크 분할 중...")
    chunks = ch13_chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"{len(chunks)}개의 텍스트 청크 생성 완료")

    # 각 청크에 대해 임베딩 생성
    print("청크 임베딩 생성 중...")
    chunk_embeddings = ch13_create_embeddings(chunks)

    # 벡터 저장소 초기화
    store = ch13_SimpleVectorStore()

    # 각 청크와 임베딩, 메타데이터를 저장소에 추가
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )

    print(f"{len(chunks)}개의 청크가 벡터 저장소에 추가됨")

    # 벡터 저장소 반환
    return store

def ch13_determine_if_retrieval_needed(query):
    """
    주어진 질의에 대해 외부 정보 검색이 필요한지 여부를 판단하는 함수

    Args:
        query (str): 사용자 질의

    Returns:
        bool: 검색이 필요한 경우 True, 그렇지 않으면 False
    """
    # AI에게 검색 필요 여부를 판별하는 기준을 알려주는 시스템 프롬프트
    system_prompt = """당신은 주어진 질의에 대해 검색이 필요한지를 판별하는 AI 어시스턴트입니다.
    사실 기반 질문, 특정 정보 요청, 사건, 인물, 개념에 대한 질문이라면 "Yes"라고 답하세요.
    의견, 가정적 시나리오, 일반 상식에 해당하는 질문은 "No"라고 답하세요.
    반드시 "Yes" 또는 "No"로만 답변하세요."""

    # 사용자 질의를 포함한 프롬프트 구성
    user_prompt = f"Query: {query}\n\nIs retrieval necessary to answer this query accurately?"

    # AI 모델 호출을 통해 검색 필요 여부 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 응답 내용에서 "yes" 또는 "no" 추출 후 소문자로 변환
    answer = response.choices[0].message.content.strip().lower()

    # "yes"가 포함되어 있으면 True 반환, 아니면 False
    return "yes" in answer

def ch13_evaluate_relevance(query, context):
    """
    주어진 문서 내용이 사용자 질의와 관련이 있는지를 평가하는 함수

    Args:
        query (str): 사용자 질의
        context (str): 문서 또는 텍스트 콘텐츠

    Returns:
        str: 'relevant' 또는 'irrelevant' 중 하나
    """
    # AI에게 문서 관련성 판단 기준을 안내하는 시스템 프롬프트
    system_prompt = """당신은 문서가 특정 질의와 관련이 있는지를 판단하는 AI 어시스턴트입니다.
    문서가 질의에 답하는 데 도움이 되는 정보를 포함하고 있는지를 기준으로 판단하세요.
    반드시 "Relevant" 또는 "Irrelevant" 중 하나로만 답변하세요."""

    # 너무 긴 문맥은 잘라서 처리 (토큰 초과 방지)
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 사용자 프롬프트 구성: 질의와 문서 내용 포함
    user_prompt = f"""Query: {query}
    Document content:
    {context}

    이 문서가 쿼리와 관련이 있나요? "관련 있음" 또는 "관련 없음"으로만 답변하세요.
    """

    # AI 모델 호출을 통해 관련성 판단 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 응답을 소문자로 변환하여 반환
    answer = response.choices[0].message.content.strip().lower()

    return answer

def ch13_assess_support(response, context):
    """
    응답이 문맥에 의해 얼마나 잘 뒷받침되는지를 평가하는 함수

    Args:
        response (str): 생성된 응답
        context (str): 응답의 근거가 되는 문서 또는 텍스트

    Returns:
        str: 'fully supported', 'partially supported', 'no support' 중 하나
    """
    # 문맥이 응답을 뒷받침하는지를 판단하는 기준 안내용 시스템 프롬프트
    system_prompt = """당신은 주어진 문맥이 응답 내용을 얼마나 잘 뒷받침하는지를 평가하는 AI 어시스턴트입니다.
    응답 내 주장, 사실, 정보가 문맥에 의해 근거를 갖는지를 판단하세요.
    반드시 다음 중 하나로만 답변하세요:
    - "Fully supported": 응답의 모든 정보가 문맥에 명확하게 근거함
    - "Partially supported": 일부 정보는 문맥에 근거하지만 일부는 아님
    - "No support": 문맥에 전혀 근거가 없거나 모순되는 정보가 포함됨
    """

    # 문맥이 너무 길 경우 일부만 사용 (토큰 초과 방지)
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 사용자 프롬프트 구성: 문맥과 응답 내용을 포함
    user_prompt = f"""Context:
    {context}

    Response:
    {response}

    이 응답이 문맥에 의해 얼마나 잘 뒷받침되는지 평가하세요. 반드시 "Fully supported", "Partially supported", 또는 "No support" 중 하나로만 답변하세요.
    """

    # AI 모델 호출을 통해 평가 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 응답에서 평가 결과 추출 후 소문자로 변환
    answer = response.choices[0].message.content.strip().lower()

    return answer  # 평가 결과 반환

def ch13_rate_utility(query, response):
    """
    질의에 대한 응답의 유용성을 평가하는 함수

    Args:
        query (str): 사용자 질의
        response (str): 생성된 응답

    Returns:
        int: 유용성 평가 점수 (1점부터 5점까지)
    """
    # AI에게 유용성 평가 기준을 안내하는 시스템 프롬프트
    system_prompt = """당신은 질의에 대한 응답의 유용성을 평가하는 AI 어시스턴트입니다.
    응답이 질의를 얼마나 잘 해결하는지, 정보의 완전성, 정확성, 실용성을 고려하세요.
    다음 기준에 따라 1점에서 5점 사이로 평가하세요:
    - 1: 전혀 유용하지 않음
    - 2: 거의 유용하지 않음
    - 3: 보통 수준으로 유용함
    - 4: 매우 유용함
    - 5: 탁월하게 유용함
    반드시 1~5 중 하나의 숫자만 답변하세요."""

    # 사용자 질의와 응답을 포함한 프롬프트 구성
    user_prompt = f"""Query: {query}
    Response:
    {response}

    위 응답의 유용성을 1점에서 5점 사이로 평가하세요:"""

    # AI 모델 호출을 통해 유용성 점수 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 생성된 응답에서 숫자 추출
    rating = response.choices[0].message.content.strip()

    # 응답 내에서 1~5 사이의 숫자만 추출
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # 숫자만 정수로 변환하여 반환

    return 3  # 실패 시 중간값 3점 반환
def ch13_generate_response(query, context=None):
    """
    질의와 선택적 문맥을 기반으로 응답을 생성하는 함수

    Args:
        query (str): 사용자 질의
        context (str, optional): 참고할 문맥 텍스트 (선택)

    Returns:
        str: 생성된 응답 텍스트
    """
    # AI에게 도움이 되는 응답을 생성하라고 안내하는 시스템 프롬프트
    system_prompt = """당신은 유용한 AI 어시스턴트입니다. 명확하고 정확하며 정보에 기반한 응답을 제공하세요."""

    # 문맥이 제공된 경우, 문맥을 포함하여 사용자 프롬프트 구성
    if context:
        user_prompt = f"""Context:
        {context}

        Query: {query}

        위 문맥을 기반으로 질의에 응답하세요."""
    else:
        # 문맥이 없는 경우, 질의만 포함
        user_prompt = f"""Query: {query}

        최선을 다해 질의에 응답하세요."""

    # AI 모델을 호출하여 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 응답 텍스트 추출 후 양쪽 공백 제거하여 반환
    return response.choices[0].message.content.strip()

def ch13_self_rag(query, vector_store, top_k=3):
    """
    Self-RAG 전체 파이프라인을 수행하는 함수

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 문서 청크를 담고 있는 벡터 저장소
        top_k (int): 초기 검색 시 반환할 문서 수

    Returns:
        dict: 질의, 생성된 응답, 과정 중 수집된 평가 메트릭을 포함한 결과 딕셔너리
    """
    print(f"\nSelf-RAG 시작: {query}\n")

    # 1단계: 외부 검색이 필요한 질의인지 판단
    print("1단계: 검색 필요 여부 판단 중...")
    retrieval_needed = ch13_determine_if_retrieval_needed(query)
    print(f"검색 필요 여부: {retrieval_needed}")

    # Self-RAG 과정 중 측정할 메트릭 초기화
    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }

    best_response = None
    best_score = -1

    if retrieval_needed:
        # 2단계: 질의 임베딩을 기반으로 문서 검색
        print("\n2단계: 관련 문서 검색 중...")
        query_embedding = ch13_create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        print(f"{len(results)}개의 문서 검색됨")

        # 3단계: 검색된 문서들의 관련성 평가
        print("\n3단계: 문서 관련성 평가 중...")
        relevant_contexts = []

        for i, result in enumerate(results):
            context = result["text"]
            relevance = ch13_evaluate_relevance(query, context)
            print(f"문서 {i+1} 관련성: {relevance}")
            if relevance == "relevant":
                relevant_contexts.append(context)

        metrics["relevant_documents"] = len(relevant_contexts)
        print(f"관련 문서 수: {len(relevant_contexts)}")

        # 4단계: 관련 문서 각각에 대해 응답 생성 및 평가
        if relevant_contexts:
            print("\n4단계: 관련 문서 기반 응답 생성 및 평가 중...")
            for i, context in enumerate(relevant_contexts):
                print(f"\n문맥 {i+1}/{len(relevant_contexts)} 처리 중...")

                print("응답 생성 중...")
                response = ch13_generate_response(query, context)

                print("응답의 문맥 기반 근거 평가 중...")
                support_rating = ch13_assess_support(response, context)
                print(f"근거 평가: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)

                print("응답 유용성 평가 중...")
                utility_rating = ch13_rate_utility(query, response)
                print(f"유용성 점수: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)

                # 응답의 전반적 점수 계산
                support_score = {
                    "fully supported": 3,
                    "partially supported": 1,
                    "no support": 0
                }.get(support_rating, 0)

                overall_score = support_score * 5 + utility_rating
                print(f"전체 점수: {overall_score}")

                # 가장 높은 점수의 응답 저장
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    print("새로운 최적 응답 업데이트됨")

        # 관련 문서가 없거나 응답 품질이 낮은 경우
        if not relevant_contexts or best_score <= 0:
            print("\n적절한 문맥이 없거나 응답 품질이 낮아 문맥 없이 직접 생성 중...")
            best_response = ch13_generate_response(query)
    else:
        # 검색 없이 직접 응답 생성
        print("\n검색 없이 직접 응답 생성 중...")
        best_response = ch13_generate_response(query)

    # 최종 메트릭 정리
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0

    print("\nSelf-RAG 완료")

    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }

def ch13_run_self_rag_example():
    """
    Self-RAG 시스템의 전체 작동 예시를 보여주는 함수
    """
    # 문서 전처리 수행
    pdf_path = "dataset/AI_Understanding.pdf"  # 처리할 PDF 문서 경로
    print(f"문서 처리 중: {pdf_path}")
    vector_store = ch13_process_document(pdf_path)  # 문서를 벡터 저장소로 변환

    # 예제 1: 검색이 필요할 가능성이 높은 질의
    query1 = "AI 개발의 주요 윤리적 문제는 무엇인가요?"
    print("\n" + "="*80)
    print(f"예제 1: {query1}")
    result1 = ch13_self_rag(query1, vector_store)  # Self-RAG 실행
    print("\n최종 응답:")
    print(result1["response"])  # 생성된 응답 출력
    print("\n메트릭:")
    print(json.dumps(result1["metrics"], indent=2))  # 평가 지표 출력

    # 예제 2: 검색 없이 직접 생성해도 되는 창작형 질의
    query2 = "인공지능에 대한 시 한 편을 쓸 수 있나요?"
    print("\n" + "="*80)
    print(f"예제 2: {query2}")
    result2 = ch13_self_rag(query2, vector_store)
    print("\n최종 응답:")
    print(result2["response"])
    print("\n메트릭:")
    print(json.dumps(result2["metrics"], indent=2))

    # 예제 3: 문서와 관련 있지만 추가 지식이 필요한 복합 질의
    query3 = "AI가 개발도상국의 경제 성장에 어떤 영향을 미칠까요?"
    print("\n" + "="*80)
    print(f"예제 3: {query3}")
    result3 = ch13_self_rag(query3, vector_store)
    print("\n최종 응답:")
    print(result3["response"])
    print("\n메트릭:")
    print(json.dumps(result3["metrics"], indent=2))

    # 결과 딕셔너리로 반환
    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }

def ch13_traditional_rag(query, vector_store, top_k=3):
    """
    전통적인 RAG 방식으로 질의에 응답을 생성하는 함수 (비교용)

    Args:
        query (str): 사용자 질의
        vector_store (SimpleVectorStore): 문서 청크를 포함한 벡터 저장소
        top_k (int): 검색할 문서 수

    Returns:
        str: 생성된 응답 텍스트
    """
    print(f"\n전통적인 RAG 실행 중: {query}\n")

    # 질의 임베딩 생성 및 유사 문서 검색
    print("문서 검색 중...")
    query_embedding = ch13_create_embeddings(query)
    results = vector_store.similarity_search(query_embedding, k=top_k)
    print(f"{len(results)}개의 문서 검색됨")

    # 검색된 문서들의 텍스트를 하나의 문맥으로 결합
    contexts = [result["text"] for result in results]
    combined_context = "\n\n".join(contexts)

    # 결합된 문맥을 기반으로 응답 생성
    print("응답 생성 중...")
    response = ch13_generate_response(query, combined_context)

    return response

def ch13_evaluate_rag_approaches(pdf_path, test_queries, reference_answers=None):
    """
    Self-RAG과 전통적인 RAG 방식을 비교 평가하는 함수

    Args:
        pdf_path (str): 문서 경로
        test_queries (List[str]): 테스트 질의 리스트
        reference_answers (List[str], optional): 평가용 참조 정답 리스트

    Returns:
        dict: 평가 결과 딕셔너리 (질의별 비교 결과 및 전체 분석 포함)
    """
    print("RAG 방식 비교 평가 시작")

    # 문서를 처리하여 벡터 저장소 생성
    vector_store = ch13_process_document(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        print(f"\n질의 {i+1} 처리 중: {query}")

        # Self-RAG 실행
        self_rag_result = ch13_self_rag(query, vector_store)
        self_rag_response = self_rag_result["response"]

        # 전통적인 RAG 실행
        trad_rag_response = ch13_traditional_rag(query, vector_store)

        # 참조 정답이 있다면 불러옴
        reference = reference_answers[i] if reference_answers and i < len(reference_answers) else None

        # 응답 비교 수행 (정량 또는 정성 평가)
        comparison = ch13_compare_responses(
            query,
            self_rag_response,
            trad_rag_response,
            reference
        )

        # 결과 저장
        results.append({
            "query": query,
            "self_rag_response": self_rag_response,
            "traditional_rag_response": trad_rag_response,
            "reference_answer": reference,
            "comparison": comparison,
            "self_rag_metrics": self_rag_result["metrics"]
        })

    # 전체 결과 분석 수행 (예: 점수 평균, 빈도, 요약 등)
    overall_analysis = ch13_generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis
    }

def ch13_compare_responses(query, self_rag_response, trad_rag_response, reference=None):
    """
    Self-RAG과 전통 RAG 응답을 비교 분석하는 함수

    Args:
        query (str): 사용자 질의
        self_rag_response (str): Self-RAG 응답
        trad_rag_response (str): 전통 RAG 응답
        reference (str, optional): 참조 정답 (사실 검증용)

    Returns:
        str: 응답 비교 분석 결과
    """
    # 시스템 프롬프트: 비교 기준과 역할 정의
    system_prompt = """당신은 RAG 시스템 응답 평가 전문가입니다.
    당신의 임무는 두 가지 RAG 접근 방식의 응답을 비교 분석하는 것입니다:

    1. Self-RAG: 검색 필요 여부를 동적으로 판단하고, 관련성과 응답 품질을 평가함
    2. 전통 RAG: 항상 문서를 검색하여 그 내용을 기반으로 응답을 생성함

    다음 기준에 따라 응답을 비교하세요:
    - 질의와의 관련성
    - 사실적 정확성
    - 정보의 완전성과 유익함
    - 간결성 및 초점의 명확성"""

    # 사용자 프롬프트 구성
    user_prompt = f"""질의:
    {query}

    Self-RAG 응답:
    {self_rag_response}

    전통 RAG 응답:
    {trad_rag_response}
    """

    # 참조 정답이 있다면 포함
    if reference:
        user_prompt += f"""
    참조 정답 (사실 검증용):
    {reference}
    """

    # 평가 요청 문구 추가
    user_prompt += """
    위 두 응답을 비교하고 어떤 응답이 더 나은지 그 이유를 설명하세요.
    정확성, 관련성, 정보의 완전성, 응답 품질을 중심으로 평가해주세요.
    """

    # LLM을 통해 비교 분석 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",  # 평가에 적합한 모델 사용
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 분석 결과 텍스트 반환
    return response.choices[0].message.content

def ch13_compare_responses(query, self_rag_response, trad_rag_response, reference=None):
    """
    Self-RAG과 전통 RAG 응답을 비교 분석하는 함수

    Args:
        query (str): 사용자 질의
        self_rag_response (str): Self-RAG 응답
        trad_rag_response (str): 전통 RAG 응답
        reference (str, optional): 참조 정답 (사실 검증용)

    Returns:
        str: 응답 비교 분석 결과
    """
    # 시스템 프롬프트: 비교 기준과 역할 정의
    system_prompt = """당신은 RAG 시스템 응답 평가 전문가입니다.
    당신의 임무는 두 가지 RAG 접근 방식의 응답을 비교 분석하는 것입니다:

    1. Self-RAG: 검색 필요 여부를 동적으로 판단하고, 관련성과 응답 품질을 평가함
    2. 전통 RAG: 항상 문서를 검색하여 그 내용을 기반으로 응답을 생성함

    다음 기준에 따라 응답을 비교하세요:
    - 질의와의 관련성
    - 사실적 정확성
    - 정보의 완전성과 유익함
    - 간결성 및 초점의 명확성"""

    # 사용자 프롬프트 구성
    user_prompt = f"""질의:
    {query}

    Self-RAG 응답:
    {self_rag_response}

    전통 RAG 응답:
    {trad_rag_response}
    """

    # 참조 정답이 있다면 포함
    if reference:
        user_prompt += f"""
    참조 정답 (사실 검증용):
    {reference}
    """

    # 평가 요청 문구 추가
    user_prompt += """
    위 두 응답을 비교하고 어떤 응답이 더 나은지 그 이유를 설명하세요.
    정확성, 관련성, 정보의 완전성, 응답 품질을 중심으로 평가해주세요.
    """

    # LLM을 통해 비교 분석 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",  # 평가에 적합한 모델 사용
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 분석 결과 텍스트 반환
    return response.choices[0].message.content

def ch13_generate_overall_analysis(results):
    """
    Self-RAG과 전통적인 RAG의 테스트 결과를 바탕으로 종합 분석을 생성하는 함수

    Args:
        results (List[Dict]): evaluate_rag_approaches에서 생성된 비교 결과 리스트

    Returns:
        str: 전체 분석 결과 텍스트
    """
    # LLM에게 비교 분석의 기준과 작성 방향을 안내하는 시스템 프롬프트
    system_prompt = """당신은 RAG 시스템 평가 전문가입니다. 여러 테스트 질의 결과를 바탕으로
    Self-RAG과 전통적인 RAG을 비교 분석하세요.

    다음 항목에 중점을 두어 분석을 작성하세요:
    1. Self-RAG이 더 잘 작동하는 경우와 그 이유
    2. 전통 RAG이 더 잘 작동하는 경우와 그 이유
    3. Self-RAG의 동적 검색 판단이 미치는 영향
    4. Self-RAG 내 관련성 및 근거 평가의 가치
    5. 질의 유형에 따른 접근 방식 선택에 대한 권장사항"""

    # 각 질의별 비교 결과 요약 텍스트 생성
    comparisons_summary = ""
    for i, result in enumerate(results):
        comparisons_summary += f"질의 {i+1}: {result['query']}\n"
        comparisons_summary += f"Self-RAG 메트릭: 검색 필요 여부: {result['self_rag_metrics']['retrieval_needed']}, "
        comparisons_summary += f"관련 문서 수: {result['self_rag_metrics']['relevant_documents']}/{result['self_rag_metrics']['documents_retrieved']}\n"
        comparisons_summary += f"비교 요약: {result['comparison'][:200]}...\n\n"

    # 사용자 프롬프트 구성: 비교 요약 전체 포함
    user_prompt = f"""다음은 총 {len(results)}개의 테스트 질의에 대한 Self-RAG vs 전통 RAG 비교 요약입니다:

    {comparisons_summary}

    이 요약을 기반으로 두 접근 방식에 대한 종합 분석을 작성해주세요."""

    # LLM 호출을 통해 분석 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 생성된 분석 결과 텍스트 반환
    return response.choices[0].message.content
#-----------------ch13 Self-RAG : end

#-----------------ch14 명제 청킹 : Proposition Chunking RAG : start
class ch14_SimpleVectorStore:
    """
    NumPy를 사용한 간단한 벡터 저장소 구현입니다.
    """
    def __init__(self):
        """
        벡터 저장소를 초기화합니다.
        """
        self.vectors = []  # 임베딩 벡터를 저장할 리스트
        self.texts = []  # 원본 텍스트를 저장할 리스트
        self.metadata = []  # 각 텍스트의 메타데이터를 저장할 리스트
    
    def add_item(self, text, embedding, metadata=None):
        """
        항목을 벡터 저장소에 추가합니다.

        Args:
        text (str): 원본 텍스트
        embedding (List[float]): 임베딩 벡터
        metadata (dict, 선택): 추가적인 메타데이터
        """
        self.vectors.append(np.array(embedding))  # 임베딩을 넘파이 배열로 변환하여 vectors 리스트에 추가
        self.texts.append(text)  # 원본 텍스트를 texts 리스트에 추가
        self.metadata.append(metadata or {})  # 메타데이터를 metadata 리스트에 추가, 없으면 빈 딕셔너리 사용
    
    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        쿼리 임베딩과 가장 유사한 항목들을 찾습니다.

        Args:
        query_embedding (List[float]): 쿼리 임베딩 벡터
        k (int): 반환할 결과 수
        filter_func (callable, 선택): 결과를 필터링할 함수

        Returns:
        List[Dict]: 텍스트와 메타데이터, 유사도 점수를 포함한 상위 k개 유사 항목 리스트
        """
        if not self.vectors:
            return []  # 저장된 벡터가 없다면 빈 리스트 반환
        
        # 쿼리 임베딩을 넘파이 배열로 변환
        query_vector = np.array(query_embedding)
        
        # 코사인 유사도를 사용하여 유사도 계산
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 필터 함수가 있다면 해당 메타데이터를 기준으로 필터링
            if filter_func and not filter_func(self.metadata[i]):
                continue
                
            # 코사인 유사도 계산
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 인덱스와 유사도 점수를 추가
        
        # 유사도를 기준으로 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개의 결과 반환
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 텍스트 추가
                "metadata": self.metadata[idx],  # 메타데이터 추가
                "similarity": score  # 유사도 점수 추가
            })
        
        return results  # 상위 k개 결과 리스트 반환

def ch14_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text

def ch14_chunk_text(text, chunk_size=800, overlap=100):
    """
    텍스트를 일정 길이로 겹치게 분할합니다.

    Args:
        text (str): 분할할 원본 텍스트
        chunk_size (int): 각 청크의 문자 수 (기본: 800자)
        overlap (int): 청크 간 중첩 길이 (기본: 100자)

    Returns:
        List[Dict]: 텍스트와 메타데이터를 포함한 청크 딕셔너리 리스트
    """
    chunks = []  # 청크들을 저장할 빈 리스트 초기화

    # 지정된 청크 크기와 중첩 길이에 따라 텍스트 분할
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]  # 해당 범위만큼 청크 추출
        if chunk:  # 빈 청크는 제외
            chunks.append({
                "text": chunk,  # 청크 본문
                "chunk_id": len(chunks) + 1,  # 청크 고유 ID
                "start_char": i,  # 청크 시작 인덱스
                "end_char": i + len(chunk)  # 청크 끝 인덱스
            })

    # 생성된 청크 수 출력
    print(f"Total {len(chunks)}개의 텍스트 청크가 생성되었습니다.")
    return chunks  # 청크 리스트 반환

def ch14_create_embeddings(text, model="text-embedding-3-small"):
    """
    주어진 텍스트에 대해 임베딩을 생성합니다.

    Args:
        text (str 또는 List[str]): 임베딩을 생성할 입력 텍스트(또는 텍스트 리스트)
        model (str): 사용할 임베딩 모델 이름

    Returns:
        List[float] 또는 List[List[float]]: 생성된 임베딩 벡터 또는 벡터 리스트
    """
    # 입력이 문자열 하나일 수도 있고, 문자열 리스트일 수도 있으므로 리스트 형태로 통일
    input_text = text if isinstance(text, list) else [text]

    # 지정된 모델을 사용하여 임베딩 생성 요청
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    response = _client.embeddings.create(       
        model=model,
        input=input_text
    )

    # 입력이 단일 문자열이었을 경우, 첫 번째 임베딩만 반환
    if isinstance(text, str):
        return response.data[0].embedding

    # 여러 문자열일 경우, 모든 임베딩 리스트 반환
    return [item.embedding for item in response.data]

def ch14_generate_propositions(chunk):
    """
    텍스트 청크에서 원자적이고 자족적인 명제들을 생성합니다.

    매개변수:
        chunk (Dict): 텍스트와 메타데이터를 포함한 청크

    반환값:
        List[str]: 생성된 명제 리스트
    """
    # 명제 생성을 위한 시스템 프롬프트 정의
    system_prompt = """다음 텍스트를 단순하고 자족적인 명제들로 분해해 주세요.
    각 명제는 다음 기준을 충족해야 합니다:

    1. 하나의 사실만 표현할 것: 각 명제는 하나의 구체적인 사실이나 주장만을 담아야 합니다.
    2. 문맥 없이 이해 가능할 것: 명제는 자족적이어야 하며, 추가적인 문맥 없이도 이해되어야 합니다.
    3. 대명사 대신 전체 이름 사용할 것: 대명사나 모호한 지시어 대신, 전체 엔터티 이름을 사용하세요.
    4. 관련 날짜/수식어 포함: 필요한 경우 명확성을 위해 날짜, 시간, 수식어를 포함하세요.
    5. 하나의 주어-술어 관계만 포함: 연결사 없이 하나의 주어와 그에 해당하는 동작 또는 속성만 표현하세요.

    명제 리스트만 출력하고, 그 외의 설명이나 추가 텍스트는 포함하지 마세요."""


    # 사용자 프롬프트: 명제로 변환할 텍스트 청크
    user_prompt = f"명제(proposition)로 변환할 텍스트:\n\n{chunk['text']}"
    
    # 모델 호출
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 창의성보다 정확성을 중시
    )
    
    # 응답으로부터 명제 줄 단위 추출
    raw_propositions = response.choices[0].message.content.strip().split('\n')
    
    # 불필요한 번호, 기호 등 제거하여 명제 정리
    clean_propositions = []
    for prop in raw_propositions:
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # 너무 짧거나 빈 명제는 제외
            clean_propositions.append(cleaned)
    
    return clean_propositions

def evaluate_proposition(proposition, original_text):
    """
    Evaluate a proposition's quality based on accuracy, clarity, completeness, and conciseness.
    
    Args:
        proposition (str): The proposition to evaluate
        original_text (str): The original text for comparison
        
    Returns:
        Dict: Scores for each evaluation dimension
    """
    # System prompt to instruct the AI on how to evaluate the proposition
    system_prompt = """You are an expert at evaluating the quality of propositions extracted from text.
    Rate the given proposition on the following criteria (scale 1-10):

    - Accuracy: How well the proposition reflects information in the original text
    - Clarity: How easy it is to understand the proposition without additional context
    - Completeness: Whether the proposition includes necessary details (dates, qualifiers, etc.)
    - Conciseness: Whether the proposition is concise without losing important information

    The response must be in valid JSON format with numerical scores for each criterion:
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """

    # User prompt containing the proposition and the original text
    user_prompt = f"""Proposition: {proposition}

    Original Text: {original_text}

    Please provide your evaluation scores in JSON format."""

    # Generate response from the model
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model= "gpt-4o-mini",  #"meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    # Parse the JSON response
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }
    
    
def ch14_evaluate_proposition(proposition, original_text):
    """
    명제를 정확성, 명확성, 완전성, 간결성 기준으로 평가합니다.

    Args:
        proposition (str): 평가할 명제
        original_text (str): 명제가 추출된 원본 텍스트

    Returns:
        Dict: 각 평가 기준에 대한 점수 (1~10)
    """
    # 평가 기준을 설명하는 시스템 프롬프트
    system_prompt = """귀하는 텍스트에서 추출한 명제의 품질을 평가하는 전문가입니다.
    다음 기준에 따라 주어진 명제를 평가하세요(1~10점 척도):

    - Accuracy: 명제가 원문 텍스트의 정보를 얼마나 잘 반영하는지 여부
    - Clarity: 추가적인 맥락 없이도 명제를 얼마나 쉽게 이해할 수 있는지 여부
    - Completeness: 명제에 필요한 세부 사항(날짜, 한정어 등)이 포함되어 있는지 여부
    - Conciseness: 명제가 중요한 정보를 놓치지 않고 간결한지 여부

    응답은 각 기준에 대한 수치 점수가 포함된 유효한 JSON 형식이어야 합니다:
        {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
        """

    # 사용자 입력: 명제와 원문
    user_prompt = f"""Proposition: {proposition}

    Original Text: {original_text}

    평가 점수를 JSON 형식으로 제공해 주세요."""

    # LLM을 호출하여 평가 점수 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},  # JSON 형식 응답 요청
        temperature=0  # 일관된 평가를 위해 창의성 최소화
    )
    
    # 모델 응답에서 JSON 파싱 시도
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # JSON 파싱 실패 시, 기본 점수 반환
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }
    
def ch14_process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100, quality_thresholds=None):
    """
    문서를 처리하여 품질 기준을 통과한 명제들을 생성합니다.

    Args:
        pdf_path (str): PDF 파일 경로
        chunk_size (int): 각 청크의 문자 수
        chunk_overlap (int): 청크 간 중첩 문자 수
        quality_thresholds (Dict): 명제 품질 평가 기준 점수

    Returns:
        Tuple[List[Dict], List[Dict]]: 원본 청크 리스트, 품질 필터링된 명제 리스트
    """
    # 품질 기준이 없을 경우 기본 기준 설정
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }
    
    # PDF에서 텍스트 추출
    text = ch14_extract_text_from_pdf(pdf_path)
    
    # 추출된 텍스트를 청크 단위로 분할
    chunks = ch14_chunk_text(text, chunk_size, chunk_overlap)
    
    all_propositions = []  # 전체 명제 저장 리스트 초기화

    print("청크로부터 명제를 생성 중...")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}/{len(chunks)} 번째 청크 처리 중...")

        # 현재 청크에 대해 명제 생성
        chunk_propositions = ch14_generate_propositions(chunk)
        print(f"생성된 명제 수: {len(chunk_propositions)}")

        # 각 명제를 메타데이터와 함께 저장
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)

    # 명제 품질 평가 단계
    print("\n명제 품질 평가 중...")
    quality_propositions = []  # 품질 기준을 통과한 명제 리스트

    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # 10개마다 진행 상황 출력
            print(f"{i+1}/{len(all_propositions)} 번째 명제 평가 중...")

        # 해당 명제의 품질 점수 평가
        scores = ch14_evaluate_proposition(prop["text"], prop["source_text"])
        prop["quality_scores"] = scores

        # 모든 기준 점수를 통과하는지 확인
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break

        if passes_quality:
            quality_propositions.append(prop)
        else:
            print(f"품질 기준 미달 명제: {prop['text'][:50]}...")

    print(f"\n최종 통과 명제 수: {len(quality_propositions)}/{len(all_propositions)}")

    # 결과 반환: 전체 청크와, 품질 기준 통과 명제
    return chunks, quality_propositions

def ch14_build_vector_stores(chunks, propositions):
    """
    문서 청크와 명제 기반의 벡터 저장소를 생성합니다.

    Args:
        chunks (List[Dict]): 원본 문서 청크 리스트
        propositions (List[Dict]): 품질 필터링된 명제 리스트

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 청크 기반, 명제 기반 벡터 저장소
    """
    # 청크 기반 벡터 저장소 생성
    chunk_store = ch14_SimpleVectorStore()
    
    # 청크 텍스트 추출 및 임베딩 생성
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"{len(chunk_texts)}개의 청크에 대해 임베딩 생성 중...")
    chunk_embeddings = ch14_create_embeddings(chunk_texts)
    
    # 메타데이터 생성 후 벡터 저장소에 추가
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)
    
    # 명제 기반 벡터 저장소 생성
    prop_store = ch14_SimpleVectorStore()
    
    # 명제 텍스트 추출 및 임베딩 생성
    prop_texts = [prop["text"] for prop in propositions]
    print(f"{len(prop_texts)}개의 명제에 대해 임베딩 생성 중...")
    prop_embeddings = ch14_create_embeddings(prop_texts)
    
    # 명제 메타데이터 생성 후 저장소에 추가
    prop_metadata = [
        {
            "type": "proposition", 
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        } 
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)
    
    # 두 개의 저장소 반환
    return chunk_store, prop_store

def ch14_retrieve_from_store(query, vector_store, k=5):
    """
    쿼리를 기반으로 벡터 저장소에서 관련 항목들을 검색합니다.

    Args:
        query (str): 사용자 쿼리
        vector_store (SimpleVectorStore): 검색 대상 벡터 저장소
        k (int): 반환할 결과 수 (기본값: 5개)

    Returns:
        List[Dict]: 유사도 점수와 메타데이터가 포함된 검색 결과 리스트
    """
    # 쿼리를 임베딩으로 변환
    query_embedding = ch14_create_embeddings(query)

    # 벡터 저장소에서 상위 k개 유사 항목 검색
    results = vector_store.similarity_search(query_embedding, k=k)

    return results

def ch14_compare_retrieval_approaches(query, chunk_store, prop_store, k=5):
    """
    하나의 쿼리에 대해 청크 기반과 명제 기반 검색 방식을 비교합니다.

    Args:
        query (str): 사용자 검색 쿼리
        chunk_store (SimpleVectorStore): 청크 기반 벡터 저장소
        prop_store (SimpleVectorStore): 명제 기반 벡터 저장소
        k (int): 각 저장소에서 검색할 결과 수

    Returns:
        Dict: 두 검색 방식의 결과를 포함한 비교 정보
    """
    print(f"\n=== 쿼리: {query} ===")
    
    # 명제 기반 검색
    print("\n[명제 기반 검색 수행 중...]")
    prop_results = ch14_retrieve_from_store(query, prop_store, k)
    
    # 청크 기반 검색
    print("[청크 기반 검색 수행 중...]")
    chunk_results = ch14_retrieve_from_store(query, chunk_store, k)
    
    # 명제 기반 결과 출력
    print("\n=== 명제 기반 결과 ===")
    for i, result in enumerate(prop_results):
        print(f"{i+1}) {result['text']} (유사도: {result['similarity']:.4f})")
    
    # 청크 기반 결과 출력
    print("\n=== 청크 기반 결과 ===")
    for i, result in enumerate(chunk_results):
        # 너무 긴 텍스트는 150자까지만 출력
        truncated_text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"{i+1}) {truncated_text} (유사도: {result['similarity']:.4f})")
    
    # 결과 딕셔너리로 반환
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }

def ch14_generate_response(query, results, result_type="proposition"):
    """
    검색된 결과를 기반으로 AI 응답을 생성합니다.

    Args:
        query (str): 사용자 질문
        results (List[Dict]): 검색된 항목 리스트
        result_type (str): 검색 결과의 유형 ('proposition' 또는 'chunk')

    Returns:
        str: 생성된 AI 응답
    """
    # 검색된 텍스트들을 하나의 문맥(context) 문자열로 결합
    context = "\n\n".join([result["text"] for result in results])
    
    # AI에게 응답 지침을 주는 시스템 프롬프트 정의
    system_prompt = f"""당신은 검색된 정보를 바탕으로 질문에 답하는 AI 어시스턴트입니다.
    당신의 답변은 지식 기반에서 검색된 다음의 {result_type}들을 기반으로 해야 합니다.
    검색된 정보만으로 질문에 답할 수 없다면, 그 한계를 명확히 인정해야 합니다."""

    # 사용자 프롬프트: 질문 + 검색된 문맥
    user_prompt = f"""Query: {query}

    Retrieved {result_type}s:
    {context}

    검색된 정보를 바탕으로 쿼리에 답변해 주세요."""

    # OpenAI 또는 호환 클라이언트를 통해 응답 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2  # 비교적 낮은 창의성 (정보 충실도 중시)
    )
    
    # 응답 텍스트만 반환
    return response.choices[0].message.content

def ch14_evaluate_responses(query, prop_response, chunk_response, reference_answer=None):
    """
    명제 기반 응답과 청크 기반 응답을 비교 평가합니다.

    Args:
        query (str): 사용자 질문
        prop_response (str): 명제 기반 검색으로 생성된 응답
        chunk_response (str): 청크 기반 검색으로 생성된 응답
        reference_answer (str, 선택): 비교용 정답 (있을 경우 정확성 기준 제공)

    Returns:
        str: 평가 분석 결과 (자연어 텍스트)
    """
    # 평가 시스템 프롬프트 정의: 평가 기준과 방식 설명
    system_prompt = """당신은 정보 검색 시스템 평가 전문가입니다.
    하나의 쿼리에 대해 생성된 두 개의 응답을 비교하세요. 
    하나는 명제 기반 검색(proposition-based retrieval), 다른 하나는 청크 기반 검색(chunk-based retrieval)에 의해 생성된 응답입니다.

    다음 기준에 따라 두 응답을 평가하십시오:
    1. 정확성(Accuracy): 어느 응답이 사실적으로 더 정확한 정보를 제공하는가?
    2. 관련성(Relevance): 어느 응답이 쿼리의 의도에 더 잘 부합하는가?
    3. 간결성(Conciseness): 어느 응답이 핵심을 놓치지 않으면서 더 간결하게 설명하는가?
    4. 명확성(Clarity): 어느 응답이 더 이해하기 쉬운가?

    각 방식의 강점과 약점을 구체적으로 서술하십시오."""


    # 사용자 프롬프트 구성: 쿼리 및 두 응답 포함
    user_prompt = f"""Query: {query}

    Response from Proposition-Based Retrieval:
    {prop_response}

    Response from Chunk-Based Retrieval:
    {chunk_response}"""

    # 참조 정답이 제공된 경우, 프롬프트에 포함하여 사실성 비교 가능하도록 함
    if reference_answer:
        user_prompt += f"""

    Reference Answer (for factual checking):
    {reference_answer}"""

    # 사용자에게 비교 평가 요청
    user_prompt += """
    이 두 가지 응답을 자세히 비교하여 어떤 접근 방식이 더 나은 성과를 냈는지, 그 이유는 무엇인지 설명해 주세요."""

    # 평가 분석 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 일관된 평가를 위해 창의성 최소화
    )
    
    # 평가 결과 텍스트 반환
    return response.choices[0].message.content

def ch14_run_proposition_chunking_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    명제 기반 청크화 vs 일반 청크화에 대한 종합 평가를 실행합니다.

    Args:
        pdf_path (str): PDF 파일 경로
        test_queries (List[str]): 테스트할 질문 리스트
        reference_answers (List[str], 선택): 정답 리스트 (있을 경우 정확성 평가에 사용)

    Returns:
        Dict: 평가 결과, 전체 분석, 명제/청크 개수 포함
    """
    print("***명제 기반 청크화 평가 시작***\n")
    
    # 문서 처리 → 청크 및 명제 생성
    chunks, propositions = ch14_process_document_into_propositions(pdf_path)
    
    # 벡터 저장소 구축 (청크용, 명제용)
    chunk_store, prop_store = ch14_build_vector_stores(chunks, propositions)
    
    results = []  # 전체 평가 결과 저장 리스트
    
    # 쿼리별 테스트 실행
    for i, query in enumerate(test_queries):
        print(f"\n\n***쿼리 {i+1}/{len(test_queries)} 테스트 중***")
        print(f"질문: {query}")
        
        # 청크 기반 vs 명제 기반 검색 결과 비교
        retrieval_results = ch14_compare_retrieval_approaches(query, chunk_store, prop_store)
        
        # 명제 기반 결과로 응답 생성
        print("\n명제 기반 응답 생성 중...")
        prop_response = ch14_generate_response(
            query, 
            retrieval_results["proposition_results"], 
            "proposition"
        )
        
        # 청크 기반 결과로 응답 생성
        print("청크 기반 응답 생성 중...")
        chunk_response = ch14_generate_response(
            query, 
            retrieval_results["chunk_results"], 
            "chunk"
        )
        
        # 정답이 있다면 포함
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        
        # 두 응답 평가
        print("\n응답 비교 평가 중...")
        evaluation = ch14_evaluate_responses(query, prop_response, chunk_response, reference)
        
        # 현재 쿼리 결과 정리
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }
        results.append(query_result)
        
        # 결과 출력
        print("\n***명제 기반 응답***")
        print(prop_response)
        print("\n***청크 기반 응답***")
        print(chunk_response)
        print("\n***평가 결과***")
        print(evaluation)
    
    # 전체 종합 분석 생성
    print("\n\n***전체 분석 생성 중***")
    overall_analysis = ch14_generate_overall_analysis(results)
    print("\n" + overall_analysis)
    
    # 최종 결과 반환
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }

def ch14_generate_overall_analysis(results):
    """
    명제 기반 vs 청크 기반 접근 방식의 종합 분석을 생성합니다.

    Args:
        results (List[Dict]): 각 테스트 쿼리의 평가 결과 리스트

    Returns:
        str: 종합 분석 결과 (자연어 텍스트)
    """
    # 시스템 프롬프트: LLM에게 평가자 역할 및 비교 관점을 지시
    system_prompt = """당신은 정보 검색 시스템을 평가하는 전문가입니다.
    여러 테스트 쿼리를 바탕으로, RAG(Retrieval-Augmented Generation) 시스템에서 
    명제 기반 검색(proposition-based retrieval)과 청크 기반 검색(chunk-based retrieval)을 비교하여 종합 분석을 제공하세요.

    다음 사항에 중점을 두어 평가하십시오:
    1. 명제 기반 검색이 더 우수한 경우
    2. 청크 기반 검색이 더 우수한 경우
    3. 각 접근 방식의 전반적인 강점과 약점
    4. 각 접근 방식을 어떤 상황에서 사용하는 것이 좋은지에 대한 추천"""


    # 각 쿼리 평가의 요약 내용을 생성
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Evaluation Summary: {result['evaluation'][:200]}...\n\n"  # 앞부분만 요약 출력

    # 사용자 프롬프트: 전체 평가 요약을 기반으로 종합 분석 요청
    user_prompt = f"""다음은 명제 기반 검색(proposition-based retrieval)과 청크 기반 검색(chunk-based retrieval)에 대한 {len(results)}개의 쿼리 평가 결과입니다. 
    이 평가들을 바탕으로 두 접근 방식에 대한 종합적인 비교 분석을 작성해 주세요:

    {evaluations_summary}

    명제 기반 검색과 청크 기반 검색의 상대적인 강점과 약점을 중심으로,
    RAG(Retrieval-Augmented Generation) 시스템에서 두 방식의 성능을 포괄적으로 분석해 주세요."""


    # LLM을 통해 종합 분석 생성
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')          
    response = _client.chat.completions.create(    
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 일관성 있는 분석을 위해 창의성 최소화
    )
    
    # 생성된 분석 결과 반환
    return response.choices[0].message.content
#-----------------ch14 명제 청킹 : Proposition Chunking RAG : end

#----------------- ollama 를 이용한 온프라미스 환경에서의 RAG : start
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL     = "embeddinggemma"

def ollama_extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.

    Args:
        pdf_path (str): PDF 파일 경로

    Returns:
        str: PDF에서 추출된 전체 텍스트
    """
    # PDF 파일 열기
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 전체 텍스트를 저장할 문자열 초기화

    # 각 페이지를 순회하며 텍스트 추출
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]               # 해당 페이지 가져오기
        text = page.get_text("text")         # 텍스트 형식으로 내용 추출
        all_text += text                     # 추출된 텍스트 누적

    # 추출된 전체 텍스트 반환
    return all_text

def ollama_chunk_text(text , n , overlap):
    '''
    주어진 텍스트를 n 자 단위로, 지정된 overlap만큼 겹치도록 분할합니다.
    
    :param text: 분할할 원본 텍스트
    :param n: 각 청크의 문자수
    :param overlap: 청크간 겹치는 문자수.
    '''
    chunks = [] # 청크를 저장할 빈리스트 초기화
    for i in range(0 , len(text) , n - overlap):
        chunks.append(text[i:i + n ])
    return chunks

def ollama_create_embeddings( texts ):
    ''' 주어진 텍스트에 대한 지정된 모델을 사용하여 임베딩을 생성하는 함수
    text : 임베딩을 생성할 입력 텍스트
    model : 사용할 임베딩 모델(기본값 BAAI/bge0-en-icl)
    return 
        dict : openAI API로 부터 받은 임베딩 응답 결과.
    '''
    url     = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": EMBED_MODEL, "input": texts}
    r       = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    embs = data.get("embeddings")
    if not embs:
        raise RuntimeError(f"Unexpected response: {data}")
    return embs



def ollama_embedding(texts: list[str]) -> list[list[float]]:
    url     = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": EMBED_MODEL, "input": texts}
    r       = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    embs = data.get("embeddings")
    if not embs:
        raise RuntimeError(f"Unexpected response: {data}")
    return embs

def ollama_norm(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def ollama_cosine_similarity(vec1 , vec2):
    ''' 코사인 유사도를 구현하여 사용자 쿼리에 가장 관련성 높은 텍스트 청크를 찾는다.
    vec1 : 첫번째 벡터
    vec2 : 두번째 벡터
    return :
        float :  두벡터 간의 코사인 유사도(값의 범위 -1 ~ 1)
    '''
    return np.dot(vec1 , vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) )


def ollama_semantic_search( query_embedding , text_chunks , embeddings , k=5 ):
    '''
    주어진 쿼리와 임베딩을 사용하여 텍스트 청크에서 의미 기반 검색을 수행합니다.
    
    :param query(str): embedding된 질문 자료 의미 검색에 사용할 쿼리 텍스트
    :param text_chunks(List[str]): 검색 대상이 되는 텍스트 청크 리스트
    :param embeddings(List[dict]): 각 청크에 대한 임베딩 객체 리스트
    :param k: 상위 k개의 관련 텍스트 청크를 반환
    return :
        List[str] : 쿼리와 가장 관련 있는 텍스트 청크 상위 k개
    '''
    # 쿼리에 대한 임베딩 생성    
    similarity_scores   = [] # 유사도 점수를 저장할 리스트 초기화

    #각 텍스트 청크의 임베딩과 쿼리 임베딩 간의 코사인 유사도 계산.
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = ollama_cosine_similarity(
                np.array(query_embedding) , 
                np.array(chunk_embedding) ,
                )
        similarity_scores.append( (i , similarity_score) ) # 인덱스와 유사도 함께 저장


    # 유사도 점수를 기준으로 내림차순 정렬
    similarity_scores.sort(key=lambda x : x[1] , reverse=True)

    # 상위 k개의 청크 인덱스를 추출
    top_indices = [index for index , _ in similarity_scores[:k]]

    # 상위 k개의 관련 텍스트 청크 반환
    return [text_chunks[index] for index in top_indices]

'''
def ollama_generate_response(system_prompt , user_message , model='gemma3:1b'):    
    #시스템 프롬프트와 사용자 메시지를 기반으로 AI모델의 응답을 생성합니다.    
    #:param system_prompt(str) : AI의 응답 방식을 지정하는 시스템 메세지
    #:param user_message(str): 사용자 질의 또는 메시지
    #:param model(str): 사용할 언어 모델 이름
    #return :
    #    dict 생성된 AI응답을 포함한 API응답 객체    
    
    _client = openai
    _client.api_key = util.getEnv('openai_api_key')
    res = _client.chat.completions.create(
        model = model , 
        temperature = 0 ,
        messages = [
            {'role':'system' , 'content':system_prompt} ,
            {'role':'user' , 'content':user_message} ,
        ]
     )
    return res
'''

def ollama_generate_response(system_prompt, user_message, model="gemma2:2b"):
    """
    Ollama 네이티브 API(/api/chat)로 chat 호출
    return: dict (Ollama API 응답 JSON)
    """
    url = "http://127.0.0.1:11434/api/chat"  # 온프라미스면 IP/도메인으로 변경
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()

import requests

def ollama_generate_response(system_prompt, user_message, model="gemma2:2b"):
    '''
    시스템 프롬프트와 사용자 메시지를 기반으로
    Ollama 온프라미스 모델의 응답을 생성합니다.

    :param system_prompt(str): AI의 응답 방식을 지정하는 시스템 메시지
    :param user_message(str): 사용자 질의 또는 메시지
    :param model(str): 사용할 Ollama 모델 이름
    :return: dict (Ollama API 응답 객체)
    '''

    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model"     : model,
        "stream"    : False,
        "messages"  : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options"   : {
            "temperature": 0
        }
    }

    response = requests.post(url, json=payload, timeout=240)
    response.raise_for_status()

    return response.json()



#----------------- ollama 를 이용한 온프라미스 환경에서의 RAG : end


