
import React from 'react';
import { useDispatch } from 'react-redux';
import { joinPage } from 'features/user/reducer/userSlice'
import { useForm } from "react-hook-form";
import styled from 'styled-components'
// https://react-hook-form.com/kr/advanced-usage/
export default function UserAdd() {
    const dispatch = useDispatch()
    const { register, handleSubmit, formState: { errors } } = useForm();
    return (
        <div>
            <h1>회원 가입</h1>
        <form method='POST' 
        onSubmit={ handleSubmit(async (data) => {await dispatch(joinPage({...data, 
                                            regDate: new Date().toLocaleDateString()}))})}>
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
                    <label> 이메일 : </label>
                    <input type="email" id="email" 
                        aria-invalid={errors.name ? "true" : "false"}
                        {...register('email', { required: true, maxLength: 30 })}
                        size="10" minlength="4" maxlength="15"/>
                    {/* use role="alert" to announce the error message */}
                    {errors.email && errors.email.type === "required" && (
                        <Span role="alert">아이디는 필수값입니다</Span>
                    )}
                    {errors.email && errors.email.type === "maxLength" && (
                        <Span role="alert">아이디는 4자에서 15 글자이어야 합니다.</Span>
                    )}
                </li>
                <li>
                    <label>비밀 번호 : </label>
                    <input type="password" id="password" 
                        aria-invalid={errors.password ? "true" : "false"}
                        {...register('password', { required: true, maxLength: 30 })}
                        size="10" minlength="1" maxlength="15"/>
                    {errors.password && errors.password.type === "required" && (
                        <Span role="alert">비밀 번호는 필수값입니다</Span>
                    )}
                    {errors.password && errors.password.type === "maxLength" && (
                        <Span role="alert">비밀 번호는 1자에서 15 글자이어야 합니다.</Span>
                    )}
                </li>
                <li>
                    <label> 이름 : </label>
                    <input type="text" id="name" 
                        aria-invalid={errors.name ? "true" : "false"}
                        {...register('name', { required: true, maxLength: 30 })}
                        size="10" minlength="2" maxlength="5"/>
                    {errors.name && errors.name.type === "required" && (
                        <Span role="alert">이름은 필수값입니다</Span>
                    )}
                    {errors.name && errors.name.type === "maxLength" && (
                        <Span role="alert">이름은 2자에서 5 글자이어야 합니다.</Span>
                    )}
                </li>
            </ul>
            <input type="submit" value="회원가입"/> 
        </form>
        </div>
  );
}
const Span = styled.span`
    color: red
`