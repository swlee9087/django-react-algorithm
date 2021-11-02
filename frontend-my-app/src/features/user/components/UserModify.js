import axios from 'axios';
import React, { useCallback, useState } from 'react';
import { useHistory  } from 'react-router-dom';

export default function UserModify() {
    const history = useHistory()
    const sessionUser = JSON.parse(localStorage.getItem('sessionUser')); 
    const [modify, setModify] = useState({
        userId: sessionUser.userId,
        username:sessionUser.username, 
        password:sessionUser.password, 
        email:sessionUser.email, 
        name:sessionUser.name, 
        regDate: sessionUser.regDate
    })
    const {userId, username, password, email, name} = modify
    const handleChange = e => {
        const { value, name } = e.target
        setModify({
            ...modify,
            [name] : value
        })
    }
    const headers = {
        'Content-Type' : 'application/json',
        'Authorization': 'JWT fefege..'
      }
    
    const handleSubmit = useCallback(
        e => {
            e.preventDefault()
        const modifyRequest = {...modify}
        alert(`회원수정 정보: ${JSON.stringify(modifyRequest)}`)
        axios
        .put(`http://localhost:8080/users`, JSON.stringify(modifyRequest),{headers})
        .then(res =>{
            alert(`회원 정보 수정 성공 ${res.data}`)
            localStorage.setItem('sessionUser', JSON.stringify(res.data))
            history.push("/users/detail")
        })
        .catch(err =>{
            alert(`회원수정 실패 : ${err}`)
        })
        }
    )


  return (
    <div>
         <h1>회원정보 수정</h1>
    <form onSubmit={handleSubmit} method='PUT'>
        <ul>
            <li>
              <label>
                    <span>회원번호 : {sessionUser.userId} </span>
                </label>
            </li>
            <li>
                <label>
                    <span>아이디 : {sessionUser.username} </span>
                </label>
            </li>
            <li>
                <label>
                    이메일 : <input type="email" id="email" name="email" placeholder={sessionUser.email}
                                  value={email}
                                 onChange={handleChange}/>
                </label>
            </li>
            <li>
                <label>
                    비밀 번호 : <input type="password" id="password" name="password" placeholder={sessionUser.password} 
                    value={password}
                    onChange={handleChange}/>
                </label>
            </li>
            <li>
                <label>
                    이름 : <input type="text" id="name" name="name" placeholder={sessionUser.name}
                    value={name}
                    onChange={handleChange}/>
                </label>
            </li>
           
            <li>
                <input type="submit" value="수정확인"/>
            </li>

        </ul>
    </form>
    </div>
  );
}