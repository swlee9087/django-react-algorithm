import React from 'react';
import { Route, Switch, Redirect } from 'react-router-dom';
import { Home, Navigation, Counter, ToDo, SignUp, SignIn } from 'common';
import {BackTrack,BruteForce,DivCon,DynProgramming,Greedy} from 'algorithm';
import {Linear,Mathematics,NonLinear} from 'datastructure'
import {combineReducers, createStore} from 'redux'
import {Provider} from 'react-redux'
import {todoReducer, userReducer} from 'reducers'

const rootReducer = combineReducers({todoReducer, userReducer})
const store = createStore(rootReducer)

const App=()=>(
  <Provider store ={store}>    
  <Navigation />
    <Switch>
      <Route exact path='/' component= { Home }/>
      <Redirect from='/home' to= { '/' }/>
      <Route exact path='/counter' component={Counter}/>
      <Route exact path='/todo' component={ToDo}/>
      <Route exact path='/signin' component={SignIn}/>
      <Route exact path='/signup' component={SignUp}/>

      <Route exact path='/linear' component={Linear}/>
      <Route exact path='/math' component={Mathematics}/>
      <Route exact path='/nonlinear' component={NonLinear}/>
      
      <Route exact path='/backtrack' component={BackTrack}/>
      <Route exact path='/bruteforce' component={BruteForce}/>
      <Route exact path='/divcon' component={DivCon}/>
      <Route exact path='/dynprogramming' component={DynProgramming}/>
      <Route exact path='/greedy' component={Greedy}/>
    </Switch>
    </Provider>
)
export default App;
