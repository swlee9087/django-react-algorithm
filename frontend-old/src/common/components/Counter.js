import React,{useState} from 'react';
import Button from '@mui/material/Button'
import Badge from '@mui/material/Badge';
import MailIcon from '@mui/icons-material/Mail'
import styled from 'styled-components';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import Stack from '@mui/material/Stack';


export default function Counter() {
    const[count, setCount] = useState(0)
    return(
        <CounterDiv>
            {count === 0 && <Stack sx={{ width: '300px', 'margin':'0 auto' }} spacing={2}>
                <Alert severity="warning">
                <AlertTitle>Warning</AlertTitle>No more emails</Alert>
            </Stack>}
                <br/>
            <Badge badgeContent={count>=0? count: setCount(0)} color="secondary" style={{marginBottom:'10px'}}>
                <MailIcon color="action" />
            </Badge>
            <br/>
            <Button variant="outlined" onClick={()=>setCount(count+1)}>+++</Button><SpanStyle/>
            <Button variant="outlined" onClick={()=>setCount(count-1)}>---</Button>

        </CounterDiv>
    )
}

const CounterDiv=styled.div`
text-align:center;
margin-top: 30px;
`
const SpanStyle=styled.span`margin:10px;`
