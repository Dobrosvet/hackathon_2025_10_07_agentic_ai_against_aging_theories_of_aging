import './App.css'
import { ServiceMonitor } from './components/ServiceMonitor'

function App() {
  return (
    <div style={{ padding: '20px', maxWidth: '1800px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '30px', color: '#f3f4f6' }}>
        Microservices Dashboard
      </h1>

      <ServiceMonitor
        name="PubMed Central Data Fetcher"
        wsUrl="ws://127.0.0.1:8002/ws"
        apiUrl="http://127.0.0.1:8002/api"
      />

      <ServiceMonitor
        name="Full Text Downloader"
        wsUrl="ws://127.0.0.1:8003/ws"
        apiUrl="http://127.0.0.1:8003/api"
      />

      {/* Add more services here as needed */}
    </div>
  )
}

export default App
