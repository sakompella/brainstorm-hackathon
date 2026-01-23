import { useNeuralStream } from './hooks/useNeuralStream'
import { Scene } from './components/Scene'
import './App.css'

function App() {
  const {
    isConnected,
    isConnecting,
    latestData,
    gridSize,
    time,
    fps,
    connect,
  } = useNeuralStream('ws://localhost:8765')

  return (
    <div className="app">
      <div className="scene-container">
        <Scene neuralData={latestData} gridSize={gridSize} />
      </div>

      <div className="hud">
        <div className="hud-top-left">
          <div className="status">
            <span className={`indicator ${isConnected ? 'connected' : ''}`} />
            <span>{isConnected ? 'Live' : 'Disconnected'}</span>
          </div>
          {isConnected && (
            <>
              <span className="stat">t={time.toFixed(2)}s</span>
              <span className="stat">{fps} msg/s</span>
            </>
          )}
        </div>

        <div className="hud-right">
          <button className="connect-btn" onClick={connect} disabled={isConnecting}>
            {isConnecting ? 'Connecting...' : isConnected ? 'Disconnect' : 'Connect'}
          </button>
          {!isConnected && <p className="hint">brainstorm-stream required</p>}
        </div>
      </div>
    </div>
  )
}

export default App
