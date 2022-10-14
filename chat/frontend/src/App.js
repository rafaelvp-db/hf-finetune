import React from "react";
import ReactDOM from "react-dom/client";
import "./style.css";
import Chat from "./Chat"

export default function App() {
  return (
    <div className="mainSection">
      <div className="heading">
        <img src="avatar.jpeg" className="larry-avatar"/>
        <span> Don Ate</span>
      </div>  
      <Chat />
    </div>
  );
}