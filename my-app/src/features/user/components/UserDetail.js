import React, { useState, useEffect, useCallback } from 'react';
import { Link, useHistory } from 'react-router-dom';
import axios from 'axios';


export default function UserDetail() {
    const SERVER = 'http://localhost:8080'
    const history = useHistory()
    const [detail, setDetail] = useState({})
    // details/JSON obj reception must use {}

    const fetchOne = () => {
        const sessionUser = JSON.parse(localStorage.getItem('sessionUser')); 
        alert('사용자 아이디: '+sessionUser.userId)
        axios.get(`${SERVER}/users/${sessionUser.userId}`)
        .then(res => {
            alert(`회원 정보 조회 SUCCESS : ${res.data}`)
            setDetail(res.data)
        })
        .catch(err => {
            alert(`회원 정보 조회 FAIL : ${err}`)
        })
    }
    useEffect(()=>{ // useEffect는 들어오자마자 데이터가 없어도 실행하라는 뜻
        fetchOne()
    }, [])
    const logout = e => {
        e.preventDefault()
        localStorage.setItem('sessionUser', '')
        history.push('users/list')
    }


  return (
    <div>
         <h1>회원정보</h1>
    
        <ul>
           <li><label><span>회원 번호 : {detail.userId}</span></label>    </li>
            <li>
                <label>
                    <span>아이디 : {detail.username} </span>
                </label>                
            </li>
            <li>
                <label>
                <span>이메일 : {detail.email} </span>
                </label>
            </li>
            <li>
                <label>
                    <span>비밀 번호 :  *******  </span>
                </label>
            </li>
            <li>
                <label>
                    <span>이름 : {detail.name} </span>
                </label>
            </li>
           
            <li>
                <input type="button" value="회원정보수정" onClick={()=> history.push('/users/modify')}/>
            </li>
            <li>
                <input type="button" value="로그아웃" onClick={logout}/>
            </li>
        </ul>
   
    </div>
  );
}