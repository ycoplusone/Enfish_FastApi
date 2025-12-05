from fastapi import APIRouter , Query , Path  , Body , Depends , HTTPException , Request , status , Response
from fastapi.responses import RedirectResponse
from passlib.context import CryptContext # 암호화
from datetime import datetime, timedelta , timezone
import jwt 




class utils():    
    __access_token_expire_minutes = 720.0                                               #토큰 만료 시간(분)
    __secret_key = '09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7'   #암호화키
    __algorithm  = "HS256"

    __pwd_context = CryptContext(schemes=['bcrypt_sha256'] )    #암호화

    def __init__(self):
        '''초기'''
        #print('utils init')
    

    def getHash(self , txt: str) -> str:
        ''' 암호화 시킴'''            
        return self.__pwd_context.hash(txt)

    def verifyHash(self , plain_txt: str, hashed_txt: str) -> bool:
        ''' 암호화 검증 '''
        return self.__pwd_context.verify( plain_txt , hashed_txt )
    
    def getTokenExpireMinutes(self) -> float:
        '''토큰 만료 시간 리턴'''
        return self.__access_token_expire_minutes
    
    def create_access_token(self , data: dict , expires_delta: int = 0 ):
        ''' jwt 토큰 생성 '''
        to_encode = data.copy()        
        if expires_delta != 0 :
            expire = datetime.now(timezone.utc) + timedelta(minutes=expires_delta)
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=30)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.__secret_key , algorithm=self.__algorithm)        
        return encoded_jwt
    
    def token_decoe(self,token):
        return jwt.decode(token , self.__secret_key ,algorithms=self.__algorithm )

    def is_token_check(self , token: str):
        '''토큰 이상 확인'''
        try:
            payload = jwt.decode(token, self.__secret_key, algorithms=self.__algorithm)
            return True   # 문제 없음
        except Exception:
            return False    # 만료됨


