import React, { useEffect, useState, useRef } from 'react';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

interface ServiceState {
  status: string;
  progress: number;
  total: number;
  current_paper: string;
  errors: number;
  saved: number;
  start_time: string | null;
  search_query: string;
}

interface ServiceMonitorProps {
  name: string;
  wsUrl: string;
  apiUrl: string;
}

export const ServiceMonitor: React.FC<ServiceMonitorProps> = ({ name, wsUrl, apiUrl }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [state, setState] = useState<ServiceState>({
    status: 'idle',
    progress: 0,
    total: 0,
    current_paper: '',
    errors: 0,
    saved: 0,
    start_time: null,
    search_query: ''
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`Connected to ${name} WebSocket`);
        setConnected(true);
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'log') {
          setLogs((prev) => [...prev, message.data]);
        } else if (message.type === 'state') {
          setState(message.data);
        } else if (message.type === 'logs') {
          // Initial logs batch
          setLogs(message.data);
        }
      };

      ws.onclose = () => {
        console.log(`Disconnected from ${name} WebSocket`);
        setConnected(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error(`WebSocket error for ${name}:`, error);
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      wsRef.current?.close();
    };
  }, [name, wsUrl]);

  const handleStart = async () => {
    try {
      const response = await fetch(`${apiUrl}/start`, { method: 'POST' });
      const data = await response.json();
      console.log('Start response:', data);
    } catch (error) {
      console.error('Error starting service:', error);
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch(`${apiUrl}/stop`, { method: 'POST' });
      const data = await response.json();
      console.log('Stop response:', data);
    } catch (error) {
      console.error('Error stopping service:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#4ade80';
      case 'completed':
        return '#3b82f6';
      case 'stopped':
        return '#f59e0b';
      case 'error':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR':
        return '#ef4444';
      case 'WARNING':
        return '#f59e0b';
      case 'INFO':
        return '#3b82f6';
      default:
        return '#6b7280';
    }
  };

  const progressPercentage = state.total > 0 ? (state.progress / state.total) * 100 : 0;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 400px',
      gap: '20px',
      marginBottom: '20px',
      padding: '15px',
      border: '1px solid #374151',
      borderRadius: '8px',
      backgroundColor: '#1f2937'
    }}>
      {/* Left column - Logs */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
          <h3 style={{ margin: 0, flex: 1 }}>{name}</h3>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: connected ? '#4ade80' : '#6b7280',
            marginLeft: '10px'
          }} />
        </div>

        <div style={{
          height: '400px',
          overflowY: 'auto',
          backgroundColor: '#111827',
          padding: '10px',
          borderRadius: '4px',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}>
          {logs.map((log, index) => (
            <div key={index} style={{ marginBottom: '4px' }}>
              <span style={{ color: '#6b7280' }}>
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              {' '}
              <span style={{ color: getLevelColor(log.level), fontWeight: 'bold' }}>
                [{log.level}]
              </span>
              {' '}
              <span style={{ color: '#d1d5db' }}>{log.message}</span>
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Right column - Statistics and Controls */}
      <div>
        <div style={{
          backgroundColor: '#111827',
          padding: '15px',
          borderRadius: '4px',
          marginBottom: '15px'
        }}>
          <div style={{ marginBottom: '15px' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: '5px'
            }}>
              <span>Status:</span>
              <span style={{
                color: getStatusColor(state.status),
                fontWeight: 'bold',
                textTransform: 'uppercase'
              }}>
                {state.status}
              </span>
            </div>

            {state.total > 0 && (
              <>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '5px'
                }}>
                  <span>Progress:</span>
                  <span>{state.progress} / {state.total}</span>
                </div>

                <div style={{
                  width: '100%',
                  height: '20px',
                  backgroundColor: '#374151',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  marginBottom: '10px'
                }}>
                  <div style={{
                    width: `${progressPercentage}%`,
                    height: '100%',
                    backgroundColor: '#3b82f6',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
              </>
            )}

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '10px',
              marginTop: '10px'
            }}>
              <div>
                <div style={{ color: '#6b7280', fontSize: '12px' }}>Saved</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#4ade80' }}>
                  {state.saved}
                </div>
              </div>
              <div>
                <div style={{ color: '#6b7280', fontSize: '12px' }}>Errors</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#ef4444' }}>
                  {state.errors}
                </div>
              </div>
            </div>

            {state.current_paper && (
              <div style={{ marginTop: '10px', fontSize: '12px' }}>
                <div style={{ color: '#6b7280' }}>Current:</div>
                <div style={{ color: '#d1d5db' }}>{state.current_paper}</div>
              </div>
            )}

            {state.start_time && (
              <div style={{ marginTop: '10px', fontSize: '12px' }}>
                <div style={{ color: '#6b7280' }}>Started:</div>
                <div style={{ color: '#d1d5db' }}>
                  {new Date(state.start_time).toLocaleString()}
                </div>
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={handleStart}
            disabled={state.status === 'running'}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: state.status === 'running' ? '#374151' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: state.status === 'running' ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            Start
          </button>
          <button
            onClick={handleStop}
            disabled={state.status !== 'running'}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: state.status !== 'running' ? '#374151' : '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: state.status !== 'running' ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            Stop
          </button>
        </div>

        {state.search_query && (
          <div style={{
            marginTop: '15px',
            padding: '10px',
            backgroundColor: '#111827',
            borderRadius: '4px',
            fontSize: '12px'
          }}>
            <div style={{ color: '#6b7280', marginBottom: '5px' }}>Search Query:</div>
            <div style={{ color: '#d1d5db', fontFamily: 'monospace' }}>
              {state.search_query}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
