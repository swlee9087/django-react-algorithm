import React from 'react';
import {Link} from 'react-router-dom'
import styled from 'styled-components'

const Navigation=()=>(<>
    
    <Nav class="navi">
        <NavList>
            <NavItem><Link to='/backtrack'>BackTrack</Link></NavItem>
            <NavItem><Link to='/bruteforce'>Brute Force</Link></NavItem>
            <NavItem><Link to='/divcon'>DivCon</Link></NavItem>
            <NavItem><Link to='/dynprogramming'>Dynamic Programming</Link></NavItem>
            <NavItem><Link to='/greedy'>Greedy</Link></NavItem>

            <NavItem><Link to='/linear'>Linear</Link></NavItem>
            <NavItem><Link to='/mathematics'>Mathematics</Link></NavItem>
            <NavItem><Link to='/nonlinear'>Non Linear</Link></NavItem>

            <NavItem><Link to='/counter'>Counter</Link></NavItem>
            <NavItem><Link to='/todo'>To Do's</Link></NavItem>

        </NavList>
        
    </Nav>
    </>    
)
export default Navigation

const Nav = styled.div`
    width: 100%;
`

const NavList = styled.ul`
    width: 1080px;
    display: flex;
    margin: 0 auto;
    padding: 0 auto;
`

const NavItem = styled.li`
    // margin-left: 20px;
    // margin-top: 20px;
    margin: 40px;
    display: flex;
    font-size: 10pt;
    font-weight: bold;
    font-family: 'Helvetica Neue';
`