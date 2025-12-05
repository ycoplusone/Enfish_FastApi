/* 사용자 정의 javascript 코드 */

/*async function fnCurl( _method='GET' , _curl = '' , _json = ''){ //curl 명령어 실행 함수
    if (_method == 'GET') {
        const _res = await fetch( _curl 
            , {
                method: _method 
                , headers: {  'Accept': 'application/json' , 'Content-Type': 'application/json' }
        })
        ;
        const _data = await _res.json();                
        return _data;
    }else{
        const _res = await fetch( _curl 
            , {
                method: _method 
                , headers: {  'Accept': 'application/json' , 'Content-Type': 'application/json' }
                , body: JSON.stringify( _json ) // JSON 데이터를 문자열로 변환하여 전송
        })
        ;
        const _data = await _res.json();        
        return _data;
    }    
}*/

async function fnCurl(_method = 'GET', _curl = '', _json = '') {
    let url = _curl;

    // GET일 경우 쿼리스트링으로 변환
    /*if (_method.toUpperCase() === 'GET' && _json && typeof _json === 'object') {
        const params = new URLSearchParams(_json).toString();
        url += (url.includes('?') ? '&' : '?') + params;
    }*/

    const options = {
        method: _method,
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    };

    // GET 외에는 body 포함
    if (_method.toUpperCase() !== 'GET' && _json) {
        options.body = JSON.stringify(_json);
    }

    const _res = await fetch(url, options);
    const _data = await _res.json();
    return _data;
}


function logout() { // 로그아웃 함수
    fetch( '/sys/logout', {
        method: 'DELETE',
        headers: {
        'Content-Type': 'application/json'
        },
    })
    .then(response =>  {
        console.log("HTTP 상태 코드:", response.status);
        if (response.status === 200) {
            console.warn("성공"); // 성공후 바로 화면으로 넘긴다.
            location.replace("/sys/");
        }           
    });      
}

function getCookie(name) { // 쿠키 값을 가져오는 함수
    const cookieString = document.cookie; // 모든 쿠키 문자열 가져오기
    const cookies = cookieString.split('; ');
    for (let i = 0; i < cookies.length; i++) {
    const [key, value] = cookies[i].split('=');
        if (key === name) {
            return decodeURIComponent(value);
        }
    }
    return null;
}

