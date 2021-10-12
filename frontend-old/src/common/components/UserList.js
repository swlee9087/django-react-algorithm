import React from "react";
import { useSelector, useDispatch } from "react-redux"; 
import { deleteUserAction  } from "reducers/user.reducer";

export default function UserList(){
    const users = useSelector ( state => state.userReducer.users )
    const dispatch = useDispatch()
    const deleteUser = email => dispatch(deleteUserAction(email))

    return(<>
        {users.length === 0 && (<h1>등록된 회원 목록이 없습니다.</h1>)}
        {users.length !== 0 && (<h1> 등록된 회원 목록이 {users.length}명이 있습니다.</h1>)}
        {users.length !== 0 && users.map(
            user => (<div key={user.email}>
                <span>{user.email}</span>
                <button onClick = {deleteUser.bind(null, user.email)}>X</button>
            </div>)
        )}
    </>)
}
