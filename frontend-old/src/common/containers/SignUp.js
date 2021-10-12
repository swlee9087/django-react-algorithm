import React from 'react';
import {UserJoin, UserList} from 'common'
import styled from 'styled-components'

export default function SignUp(){
    return (
        <CounterDiv>
            <h1>Sign Up</h1>
            <UserJoin/>
            <UserList/>
        </CounterDiv>
    )
}

const CounterDiv = styled.div`
    text-align: center;
`