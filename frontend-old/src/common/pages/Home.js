import React from 'react';
import { Counter, SignIn, ToDo } from 'common';
import {connect } from 'api'

export default function Home(){
    const handleClick=e=>{
        e.preventDefault()
        alert('Home Clicked')
        connect()
        .then(res => {alert(`Connection Success : ${res.data.connection}`)})
        .catch(err => {alert(`Connection Error : ${err}`)})    
    }
    return(
        <div>

            <button onClick={handleClick}>Connection</button>
            <SignIn />
        </div>
    )
}
