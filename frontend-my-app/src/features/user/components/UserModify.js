import React, { useCallback, useState } from 'react';
import { useDispatch } from 'react-redux';
import { modifyPage } from 'features/user/reducer/userSlice'

export default function UserModify() {
    const dispatch = useDispatch()
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
  return (
    <div>
         <h1>회원정보 수정</h1>
    <form onSubmit={useCallback(
        e => {
            e.preventDefault()
            dispatch(modifyPage({...modify}))
        }
    )} method='PUT'>
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