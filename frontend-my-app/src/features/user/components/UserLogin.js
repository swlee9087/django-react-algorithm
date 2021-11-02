import React from 'react';
import { useDispatch } from 'react-redux';
import { loginPage } from 'features/user/reducer/userSlice'
import { useForm } from "react-hook-form";
import styled from 'styled-components'

export default function UserLogin() {
  const dispatch = useDispatch()
  const { register, handleSubmit, formState: { errors } } = useForm();

  return (
    <div>
         <h1>로그인</h1>
    <form method='POST' 
    onSubmit={ 
      handleSubmit(async (data) => {await dispatch(loginPage(data))})}>
        <ul>
            <li>
                <label>아이디 : </label>
                <input type="text" id="username" 
                    {...register('username', { required: true, maxLength: 30 })}
                    size="10" minlength="4" maxlength="15"/>
                    {errors.username && errors.username.type === "required" && (
                        <Span role="alert">아이디는 필수값입니다</Span>
                    )}
                    {errors.username && errors.username.type === "maxLength" && (
                        <Span role="alert">아이디는 4자에서 15 글자이어야 합니다.</Span>
                    )}
                <br/>
                <small>4~15자리 이내의 영문과 숫자</small>
            </li>
            <li>
                <label>비밀 번호 : </label>
                <input type="password" id="password" 
                    aria-invalid={errors.name ? "true" : "false"}
                    {...register('password', { required: true, maxLength: 30 })}
                    size="10" minlength="1" maxlength="15"/>
                {errors.password && errors.password.type === "required" && (
                    <Span role="alert">비밀 번호는 필수값입니다</Span>
                )}
                {errors.password && errors.password.type === "maxLength" && (
                    <Span role="alert">비밀 번호는 1자에서 15 글자이어야 합니다.</Span>
                )}
            </li>
        </ul>
        <input type="submit" value="로그인"/> 
    </form>
    </div>
  );
}
const Span = styled.span`
    color: red
`