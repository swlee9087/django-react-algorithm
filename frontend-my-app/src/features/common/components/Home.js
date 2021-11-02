import { Logout } from 'features/user';
import React, { useEffect, useState } from 'react';
import { useHistory  } from 'react-router-dom';
export default function Home() {
  const history = useHistory()
  const [sessionName, setSessionName] = useState('')
  useEffect(() => {
    if(localStorage.length > 0){
      const sessionUser = JSON.parse(localStorage.getItem('sessionUser'))
      setSessionName(sessionUser.name)
    }else{
      setSessionName('None')
    }
  });
  return (
    <div>
      {sessionName !== 'None' ? <><h1> {sessionName} 님 접속중</h1> </>: <>접속자 없음</>}
      <h1>시간이란.. 22</h1>
      <p>내일 죽을 것처럼 오늘을 살고 
          영원히 살 것처럼 내일을 꿈구어라.</p>
      {localStorage.length == 0 ? <Logout/> :
       <><button onClick = {() => window.location.href='/users/add'}>회원가입</button>
       <button onClick = {() => window.location.href='/users/login'}>로그인</button></>}
    </div>
  );
}