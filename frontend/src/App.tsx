import React from 'react'
import Dashboard from './dashboard/PineLabUnifiedDashboard'

export default function App(){ 
  // Change to your backend URL if not localhost
  const apiBase = 'http://localhost:8080'
  return <div style={{fontFamily:'Inter,system-ui,Arial', background:'#f6f7fb', minHeight:'100vh'}}>
    <div style={{maxWidth:1200, margin:'0 auto', padding:20}}>
      <h1 style={{fontSize:20, fontWeight:700, marginBottom:12}}>PineScript Autogen Lab</h1>
      <Dashboard apiBase={apiBase}/>
    </div>
  </div>
}
