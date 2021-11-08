import {
  configureStore,
  combineReducers, // redux의 Reducer 의 집합과 같다.
  getDefaultMiddleware
} from "@reduxjs/toolkit";
import logger from 'redux-logger'
import counterReducer from 'features/counter/counterSlice';
import userReducer from 'features/user/reducer/userSlice';

const rootReducer = combineReducers({ userReducer })

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(logger),
});