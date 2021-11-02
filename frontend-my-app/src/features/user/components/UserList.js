// import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { UserListForm } from '..';


export default function UserList() {
  const [list, setList] = useState([])
 

  const userList = () => {
    userList()
      .then(res => setList(res.data) )
      .catch(err => console.log(err))
  }

  useEffect(() =>{
    userList() 
  }, [])
  return (
    <div>
      <h1>사용자 목록</h1>
      <UserListForm list={list}/>
    </div>
  );
}
