import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { UserListForm } from 'features/user/index';

export default function UserList() {
  const SERVER = 'http://localhost:8080'
  const [list, setList] = useState([])
  
  const fetchList = () => {
    axios.get(`${SERVER}/users/list`)
    .then(res =>
      setList(res.data)
    )
    .catch(err => console.log(err))
  }

  useEffect(() => {
    fetchList()
  }, [])
  return (<>
    <div>
      <h1>사용자 목록</h1>
      <UserListForm list = {list}/>
    </div>
  </>);
}