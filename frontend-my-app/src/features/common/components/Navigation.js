import React from 'react';
import { Link } from 'react-router-dom'
import styled from "styled-components";

export default function Navigation() {
  
  return (
    <div>
        <Ul>
            <Li><Link to="/home">Home</Link></Li>
            <Li><Link to="/users/add">UserAdd</Link></Li>
            <Li><Link to="/users/detail">UserDetail</Link></Li>
            <Li><Link to="/users/list">UserList</Link></Li>
            <Li><Link to="/users/login">UserLogin</Link></Li>
            <Li><Link to="/users/modify">UserModify</Link></Li>
            <Li><Link to="/users/remove">UserRemove</Link></Li>
        </Ul>
    </div>
  );
}

const Ul = styled.ul`
background-color: #FFDAB9;
text-decoration:none
text-align: center;


`
const Li = styled.li`
float: left;
margin-left: 1em;
font-size:20px;
text-align:center;
display:inline-block;
`