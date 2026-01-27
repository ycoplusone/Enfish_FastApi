
# main.py           => 실행 파일.
# database.py       => 데이터베이스 연결 객체.
# statics           => 정적파일 js libary 같은것.
# area              => models , cruds , dantics , router 등을 모듈 단위로 합쳐서 다시 개발한다.
# Cruds             => crud 처리 부분 => area 에 통합 예정
# models            => SQLAlchemy 모델 정의. => area 에 통합 예정
# Dantics           => Pydantic 의 모델 폴더. => area 에 통합 예정
# Ways              => router 파일 폴더. => area 에 통합 예정
# Templates         => ui 파일 폴더.    => 외부 sveltekit 으로 변경 예정.


# 설치 모듈 목록
    pip install fastapi
    pip install pydantic
    pip install SQLAlchemy
    pip install mysql    
    pip install passlib
    pip install python-multipart
    pip install "pydantic[email]"
    pip install passlib   
    pip install bcrypt==4.3.0
    pip install PyJWT 
    pip install numpy
    pip install requests

    https://github.com/mondersky/tabscolor-vscode : vscode 에서의 프로젝트별 탭 색생 변경 가이드
    
    -  OAuth2PasswordRequestForm과 jwt 을 사용하기 위한 라이브러리
        => pip install python-multipart
        => pip install "python-jose[cryptography]"

# 실행방법.
    uvicorn main:app --reload

# 문서기반
    http://127.0.0.1:8000/docs      => swagger  기반 api 문서
    http://127.0.0.1:8000/redoc     => Redoc    기반 문서

# Rag 관련
    - 실습코드와 데이터셋 : https://github.com/no-wave/llm-master-rag-techniques
    - OpenAI 라이브러리 설치 : pip install openai
    - dotenn 설치 : pip install python-dotenv
    - fitz 설치 : pip install PyMuPDF
# OLLAMA 관련
    - 올마 서버 실행 : ollama serve 
