import React from 'react';
import { Route, Redirect, Switch } from 'react-router-dom';
import { Home, Navigation }from 'features/common/index';
import {UserAdd, UserDetail, UserList, UserLogin, UserModify, UserRemove} from 'features/user/index';
import { BrowserRouter as Router } from 'react-router-dom';
const App= () => {
  return (
    <div className="App">
      <Router>
        {/* <Home/> */}
      <Navigation/>
      <Switch>
        <Route exact path='/' component= {Home}/>
        <Redirect from='/home' to ={'/'}/>
        <Route exact path='/users/add' component={UserAdd}/>
        <Route exact path='/users/detail' component={UserDetail}/>
        <Route exact path='/users/list' component={UserList}/>
        <Route exact path='/users/login' component={UserLogin}/>
        <Route exact path='/users/modify' component={UserModify}/>
        <Route exact path='/users/remove' component={UserRemove}/>
      </Switch>
      </Router>
    </div>
  );
}

export default App;