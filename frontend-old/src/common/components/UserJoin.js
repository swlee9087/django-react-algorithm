import * as React from 'react';
import {useState} from "react";
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { userRegister } from 'api';
import { addUserAction } from 'reducers/user.reducer';
import { useDispatch } from "react-redux";

const theme = createTheme();

export default function UserJoin() {
  const [user, setUser] = useState({
    username: '',
    password: '',
    name: '',
    email: '',
    birth: '',
    address: ''
    })
  const {username, password, name, email, birth, address} = `user`

  const handleSubmit = (e) => {
    e.preventDefault();
        alert(`가입 회원정보: ${JSON.stringify(user)}`)
        userRegister({user})
        .then(res => {alert(`회원가입완료: ${res.data.result}`)})
        .catch(err => {alert(`회원가입 실패: ${err}`)})    
    }
    const handleChange = e => {
      e.preventDefault()
      const {name, value} = e.target
      // alert(`name: ${name}, value: ${value}`)
      setUser({
        ...user,
        [name]: value
      })
    }

  return (
    <ThemeProvider theme={theme}>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
            <LockOutlinedIcon />
          </Avatar>
          <Typography component="h1" variant="h5">
            Sign in
          </Typography>
          <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
          <TextField
              margin="normal"
              required
              fullWidth
              name="username"
              label="username"
              type="text"
              id="username"
              value = {username}
              onChange = {handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              value = {password}
              onChange = {handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="name"
              label="name"
              type="text"
              id="name"
              value = {name}
              onChange = {handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              type="text"
              label="Email Address"
              name="email"
              value = {email}
              autoFocus
              onChange = {handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="birth"
              label="birth"
              type="text"
              id="birth"
              value = {birth}
              onChange = {handleChange}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="address"
              label="address"
              type="text"
              id="address"
              value = {address}
              onChange = {handleChange}
            />
            <FormControlLabel
              control={<Checkbox value="remember" color="primary" />}
              label="Remember me"
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Sign In
            </Button>
            <Grid container>
              <Grid item xs>
                <Link href="#" variant="body2">
                  Forgot password?
                </Link>
              </Grid>
              <Grid item>
                <Link href="#" variant="body2">
                  {"Don't have an account? Sign Up"}
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Container>
    </ThemeProvider>
  );
}