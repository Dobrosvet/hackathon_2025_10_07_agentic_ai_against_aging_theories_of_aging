import React, { useEffect, useState, useRef } from 'react';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

interface Fragment {
  text: string;
  start_position: number;
  end_position: number;
  confidence: number;
}

interface QuestionResult {
  question_id: string;
  answer: any;
  confidence: number;
  fragments: Fragment[];
}

interface Classification {
  questions: Record<string, QuestionResult>;
  criteria: Record<string, QuestionResult>;
  timestamp: string;
}

interface State {
  status: string;
  progress: number;
  total: number;
  current_paper: string;
  classified_count: number;
  errors: number;
  saved: number;
  skipped: number;
  db_count: number;
  start_time: string | null;
  message: string;
  current_text_preview: string;
  current_classification: Classification | null;
}

interface Paper {
  pmc_id: string;
  title: string;
  full_text: string;
  is_aging_theory?: boolean;
  questions_classification?: Record<string, QuestionResult>;
  criteria_classification?: Record<string, QuestionResult>;
  manual_annotations?: any[];
}

interface QuestionsClassifierViewerProps {
  name: string;
  wsUrl: string;
  apiUrl: string;
}

type Mode = 'auto' | 'manual';

// Цвета для разных классов
const CLASS_COLORS: Record<string, string> = {
  Q1: '#ef4444', Q2: '#f97316', Q3: '#f59e0b', Q4: '#eab308', Q5: '#84cc16',
  Q6: '#22c55e', Q7: '#10b981', Q8: '#14b8a6', Q9: '#06b6d4',
  C1: '#0ea5e9', C2: '#3b82f6', C3: '#6366f1', C4: '#8b5cf6'
};

export const QuestionsClassifierViewer: React.FC<QuestionsClassifierViewerProps> = ({ name, wsUrl, apiUrl }) => {
  const [mode, setMode] = useState<Mode>('auto');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [state, setState] = useState<State>({
    status: 'idle',
    progress: 0,
    total: 0,
    current_paper: '',
    classified_count: 0,
    errors: 0,
    saved: 0,
    skipped: 0,
    db_count: 0,
    start_time: null,
    message: '',
    current_text_preview: '',
    current_classification: null
  });
  const [connected, setConnected] = useState(false);
  const [currentPaper, setCurrentPaper] = useState<Paper | null>(null);
  const [selectedText, setSelectedText] = useState('');
  const [selectionRange, setSelectionRange] = useState<{start: number, end: number} | null>(null);
  const [showAnnotationMenu, setShowAnnotationMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({x: 0, y: 0});

  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const textContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
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
        setLogs([]);
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'log') {
          setLogs((prev) => [...prev, message.data]);
        } else if (message.type === 'state') {
          setState(message.data);
        } else if (message.type === 'logs') {
          setLogs(message.data);
        }
      };

      ws.onclose = () => {
        console.log(`Disconnected from ${name} WebSocket`);
        setConnected(false);
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

  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const selectedText = selection.toString().trim();
    if (selectedText.length < 10) return;

    const range = selection.getRangeAt(0);
    const container = textContainerRef.current;

    if (!container || !currentPaper) return;

    // Вычислить позицию выделения
    const fullText = currentPaper.full_text;
    const startOffset = fullText.indexOf(selectedText);

    if (startOffset === -1) return;

    setSelectedText(selectedText);
    setSelectionRange({
      start: startOffset,
      end: startOffset + selectedText.length
    });

    // Показать меню аннотации
    const rect = range.getBoundingClientRect();
    setMenuPosition({
      x: rect.left + window.scrollX,
      y: rect.bottom + window.scrollY
    });
    setShowAnnotationMenu(true);
  };

  const handleAnnotate = async (classType: string, answer: any) => {
    if (!currentPaper || !selectionRange) return;

    try {
      const response = await fetch(`${apiUrl}/paper/${currentPaper.pmc_id}/annotate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          annotation_type: classType,
          answer: answer,
          fragment: {
            text: selectedText,
            start_position: selectionRange.start,
            end_position: selectionRange.end
          }
        })
      });

      const data = await response.json();
      if (data.success) {
        // Reload paper
        const paperResp = await fetch(`${apiUrl}/paper/${currentPaper.pmc_id}`);
        const updatedPaper = await paperResp.json();
        setCurrentPaper(updatedPaper);
      }
    } catch (error) {
      console.error('Error annotating:', error);
    }

    setShowAnnotationMenu(false);
    setSelectedText('');
    setSelectionRange(null);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#4ade80';
      case 'completed': return '#3b82f6';
      case 'stopped': return '#f59e0b';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return '#ef4444';
      case 'WARNING': return '#f59e0b';
      case 'INFO': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const progressPercentage = state.total > 0 ? (state.progress / state.total) * 100 : 0;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '400px 1fr 450px',
        gap: '20px',
        marginBottom: '20px',
        padding: '15px',
        border: '1px solid #374151',
        borderRadius: '8px',
        backgroundColor: '#1f2937',
      }}
    >
      {/* Left column - Logs */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
          <h3 style={{ margin: 0, flex: 1, fontSize: '16px' }}>{name}</h3>
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: connected ? '#4ade80' : '#6b7280',
              marginLeft: '10px',
            }}
          />
        </div>

        <div
          style={{
            display: 'flex',
            gap: '5px',
            marginBottom: '10px',
            backgroundColor: '#111827',
            padding: '4px',
            borderRadius: '4px',
          }}
        >
          <button
            onClick={() => setMode('auto')}
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: mode === 'auto' ? '#3b82f6' : 'transparent',
              color: 'white',
              border: 'none',
              borderRadius: '2px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: mode === 'auto' ? 'bold' : 'normal',
            }}
          >
            Автоклассификация
          </button>
          <button
            onClick={() => setMode('manual')}
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: mode === 'manual' ? '#3b82f6' : 'transparent',
              color: 'white',
              border: 'none',
              borderRadius: '2px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: mode === 'manual' ? 'bold' : 'normal',
            }}
          >
            Ручная разметка
          </button>
        </div>

        <div
          ref={logsContainerRef}
          style={{
            height: '560px',
            overflowY: 'auto',
            backgroundColor: '#111827',
            padding: '10px',
            borderRadius: '4px',
            fontFamily: 'monospace',
            fontSize: '11px',
          }}
        >
          {logs.map((log, index) => (
            <div key={index} style={{ marginBottom: '4px' }}>
              <span style={{ color: '#6b7280' }}>{new Date(log.timestamp).toLocaleTimeString()}</span>{' '}
              <span style={{ color: getLevelColor(log.level), fontWeight: 'bold' }}>[{log.level}]</span>{' '}
              <span style={{ color: '#d1d5db' }}>{log.message}</span>
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Middle column - Text with annotations */}
      <div>
        <h4 style={{ margin: '0 0 10px 0', color: '#f3f4f6', fontSize: '14px' }}>
          {currentPaper ? `PMC${currentPaper.pmc_id}: ${currentPaper.title}` : 'Текст статьи с разметкой'}
        </h4>
        <div
          ref={textContainerRef}
          onMouseUp={mode === 'manual' ? handleTextSelection : undefined}
          style={{
            height: '600px',
            overflowY: 'auto',
            backgroundColor: '#111827',
            padding: '15px',
            borderRadius: '4px',
            border: '1px solid #374151',
            fontFamily: 'monospace',
            fontSize: '13px',
            color: '#d1d5db',
            whiteSpace: 'pre-wrap',
            userSelect: mode === 'manual' ? 'text' : 'none',
            cursor: mode === 'manual' ? 'text' : 'default'
          }}
        >
          {currentPaper ? currentPaper.full_text : state.current_text_preview || 'Нет текста для отображения'}
        </div>

        {showAnnotationMenu && (
          <div
            style={{
              position: 'fixed',
              left: menuPosition.x,
              top: menuPosition.y,
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '4px',
              padding: '10px',
              zIndex: 1000,
              maxWidth: '300px'
            }}
          >
            <div style={{ color: '#f3f4f6', fontSize: '12px', marginBottom: '8px', fontWeight: 'bold' }}>
              Выберите класс:
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px' }}>
              {Object.keys(CLASS_COLORS).map((classType) => (
                <button
                  key={classType}
                  onClick={() => handleAnnotate(classType, true)}
                  style={{
                    padding: '6px',
                    backgroundColor: CLASS_COLORS[classType],
                    color: 'white',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: 'bold'
                  }}
                >
                  {classType}
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowAnnotationMenu(false)}
              style={{
                marginTop: '8px',
                width: '100%',
                padding: '6px',
                backgroundColor: '#374151',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                fontSize: '11px'
              }}
            >
              Отмена
            </button>
          </div>
        )}
      </div>

      {/* Right column - Statistics and Controls */}
      <div>
        {mode === 'auto' ? (
          <>
            {state.message && (
              <div
                style={{
                  backgroundColor: state.status === 'completed' ? '#065f46' : '#1e3a8a',
                  padding: '12px',
                  borderRadius: '4px',
                  marginBottom: '15px',
                  color: '#fff',
                  fontSize: '13px',
                  fontWeight: 'bold',
                  textAlign: 'center',
                }}
              >
                {state.message}
              </div>
            )}

            <div
              style={{
                backgroundColor: '#111827',
                padding: '15px',
                borderRadius: '4px',
                marginBottom: '15px',
                textAlign: 'center',
              }}
            >
              <div style={{ color: '#6b7280', fontSize: '12px', marginBottom: '5px' }}>
                Классифицировано статей
              </div>
              <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#3b82f6' }}>
                {state.db_count.toLocaleString()}
              </div>
            </div>

            <div
              style={{
                backgroundColor: '#111827',
                padding: '15px',
                borderRadius: '4px',
                marginBottom: '15px',
              }}
            >
              <div style={{ marginBottom: '15px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '5px',
                    fontSize: '13px',
                  }}
                >
                  <span>Статус:</span>
                  <span
                    style={{
                      color: getStatusColor(state.status),
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                    }}
                  >
                    {state.status}
                  </span>
                </div>

                {state.total > 0 && (
                  <>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '5px',
                        fontSize: '13px',
                      }}
                    >
                      <span>Прогресс:</span>
                      <span>
                        {state.progress.toLocaleString()} / {state.total.toLocaleString()}
                      </span>
                    </div>

                    <div
                      style={{
                        width: '100%',
                        height: '24px',
                        backgroundColor: '#374151',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        marginBottom: '10px',
                        position: 'relative',
                      }}
                    >
                      <div
                        style={{
                          width: `${progressPercentage}%`,
                          height: '100%',
                          backgroundColor: '#3b82f6',
                          transition: 'width 0.3s ease',
                        }}
                      />
                      <div
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '12px',
                          fontWeight: 'bold',
                          color: '#fff',
                        }}
                      >
                        {progressPercentage.toFixed(1)}%
                      </div>
                    </div>
                  </>
                )}

                {state.current_paper && state.status === 'running' && (
                  <div style={{ marginTop: '12px', fontSize: '11px' }}>
                    <div style={{ color: '#6b7280' }}>Обрабатывается:</div>
                    <div style={{ color: '#d1d5db', wordBreak: 'break-all' }}>{state.current_paper}</div>
                  </div>
                )}

                {state.start_time && (
                  <div style={{ marginTop: '10px', fontSize: '11px' }}>
                    <div style={{ color: '#6b7280' }}>Начало:</div>
                    <div style={{ color: '#d1d5db' }}>{new Date(state.start_time).toLocaleString()}</div>
                  </div>
                )}
              </div>
            </div>

            {/* Classification Results */}
            {state.current_classification && (
              <div
                style={{
                  backgroundColor: '#111827',
                  padding: '15px',
                  borderRadius: '4px',
                  marginBottom: '15px',
                  maxHeight: '300px',
                  overflowY: 'auto'
                }}
              >
                <div style={{ color: '#f3f4f6', fontWeight: 'bold', marginBottom: '10px', fontSize: '14px' }}>
                  Результаты классификации
                </div>
                {Object.entries(state.current_classification.questions).map(([qId, result]) => (
                  <div key={qId} style={{ marginBottom: '8px', fontSize: '11px' }}>
                    <span style={{ color: CLASS_COLORS[qId], fontWeight: 'bold' }}>{qId}:</span>{' '}
                    <span style={{ color: '#d1d5db' }}>{String(result.answer)}</span>{' '}
                    <span style={{ color: '#9ca3af' }}>({(result.confidence * 100).toFixed(0)}%)</span>
                  </div>
                ))}
                <div style={{ borderTop: '1px solid #374151', marginTop: '10px', paddingTop: '10px' }}>
                  {Object.entries(state.current_classification.criteria).map(([cId, result]) => (
                    <div key={cId} style={{ marginBottom: '8px', fontSize: '11px' }}>
                      <span style={{ color: CLASS_COLORS[cId], fontWeight: 'bold' }}>{cId}:</span>{' '}
                      <span style={{ color: '#d1d5db' }}>{String(result.answer)}</span>{' '}
                      <span style={{ color: '#9ca3af' }}>({(result.confidence * 100).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
              <button
                onClick={handleStart}
                disabled={state.status === 'running'}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: state.status === 'running' ? '#374151' : '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: state.status === 'running' ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '14px',
                }}
              >
                {state.status === 'running' ? 'Обработка...' : 'Запустить'}
              </button>
              <button
                onClick={handleStop}
                disabled={state.status !== 'running'}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: state.status !== 'running' ? '#374151' : '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: state.status !== 'running' ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '14px',
                }}
              >
                Остановить
              </button>
            </div>
          </>
        ) : (
          <div
            style={{
              backgroundColor: '#111827',
              padding: '20px',
              borderRadius: '4px',
              textAlign: 'center',
              color: '#9ca3af',
            }}
          >
            Режим ручной разметки.
            <br />
            Выделите текст мышью и выберите класс.
          </div>
        )}
      </div>
    </div>
  );
};
