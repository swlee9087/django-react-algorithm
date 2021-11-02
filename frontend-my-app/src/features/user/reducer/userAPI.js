import axios from 'axios';
const SERVER = 'http://localhost:8080' 
const headers = {
  'Content-Type' : 'application/json',
  'Authorization': 'JWT fefege..'
}

const userJoin = x => axios.post(`${SERVER}/users`, JSON.stringify(x),{headers})
const userDetail = x => axios.get(`${SERVER}/users/${x.userId}`)
const userList = () => axios.get(`${SERVER}/users`)
const userLogin = x => axios.post(`${SERVER}/users/login`, JSON.stringify(x),{headers})
const userModify = x => axios.put(`${SERVER}/users`, JSON.stringify(x),{headers})
const userRemove = x => axios.delete(`${SERVER}/users/${x.userId}`)

export default {
  userJoin,
  userDetail,
  userList,
  userLogin,
  userModify,
  userRemove
} 