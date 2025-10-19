import React, { useEffect, useState, useRef } from 'react';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

interface TheorySpan {
  theory_name: string;
  matched_text: string;
  start_position: number;
  end_position: number;
  confidence: number;
  context_snippet: string;
}

interface ClassificationState {
  status: string;
  progress: number;
  total: number;
  current_paper: string;
  classified_as_theory: number;
  classified_as_not_theory: number;
  total_theories_found: number;
  errors: number;
  saved: number;
  skipped: number;
  db_count: number;
  start_time: string | null;
  message: string;
  current_text_preview: string;
  current_highlighted_spans: TheorySpan[];
}

interface Paper {
  id: number;
  pmc_id: string;
  title: string;
  full_text: string;
  is_aging_theory?: boolean;
  classification_confidence?: number;
  aging_theories?: TheorySpan[];
}

interface LabelingStats {
  total_labeled: number;
  labeled_as_theory: number;
  labeled_as_not_theory: number;
  skipped: number;
  unreviewed: number;
}

interface TheoryHighlightViewerProps {
  name: string;
  wsUrl: string;
  apiUrl: string;
}

type Mode = 'auto' | 'manual';

const HighlightedText: React.FC<{ text: string; spans: TheorySpan[] }> = ({ text, spans }) => {
  if (!text || text.length === 0) {
    return (
      <div style={{ color: '#6b7280', fontStyle: 'italic', padding: '20px', textAlign: 'center' }}>
        Нет текста для отображения
      </div>
    );
  }

  if (!spans || spans.length === 0) {
    return (
      <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '13px', color: '#d1d5db' }}>
        {text}
      </div>
    );
  }

  // Сортируем spans по позиции
  const sortedSpans = [...spans].sort((a, b) => a.start_position - b.start_position);

  // Разбиваем текст на фрагменты
  const fragments: Array<{ type: 'text' | 'highlight'; content: string; theory?: string; confidence?: number }> = [];
  let lastPos = 0;

  sortedSpans.forEach((span) => {
    // Обычный текст до подсветки
    if (span.start_position > lastPos) {
      fragments.push({
        type: 'text',
        content: text.slice(lastPos, span.start_position),
      });
    }

    // Подсвеченный фрагмент
    fragments.push({
      type: 'highlight',
      content: text.slice(span.start_position, span.end_position),
      theory: span.theory_name,
      confidence: span.confidence,
    });

    lastPos = span.end_position;
  });

  // Остаток текста
  if (lastPos < text.length) {
    fragments.push({
      type: 'text',
      content: text.slice(lastPos),
    });
  }

  return (
    <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.6' }}>
      {fragments.map((frag, idx) =>
        frag.type === 'highlight' ? (
          <mark
            key={idx}
            style={{
              backgroundColor: '#fbbf24',
              color: '#000',
              padding: '2px 4px',
              borderRadius: '3px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
            title={`${frag.theory} (уверенность: ${((frag.confidence || 0) * 100).toFixed(0)}%)`}
          >
            {frag.content}
          </mark>
        ) : (
          <span key={idx} style={{ color: '#d1d5db' }}>
            {frag.content}
          </span>
        )
      )}
    </div>
  );
};

export const TheoryHighlightViewer: React.FC<TheoryHighlightViewerProps> = ({ name, wsUrl, apiUrl }) => {
  const [mode, setMode] = useState<Mode>('auto');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [state, setState] = useState<ClassificationState>({
    status: 'idle',
    progress: 0,
    total: 0,
    current_paper: '',
    classified_as_theory: 0,
    classified_as_not_theory: 0,
    total_theories_found: 0,
    errors: 0,
    saved: 0,
    skipped: 0,
    db_count: 0,
    start_time: null,
    message: '',
    current_text_preview: '',
    current_highlighted_spans: [],
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  // Manual review state
  const [currentPaper, setCurrentPaper] = useState<Paper | null>(null);
  const [labelingStats, setLabelingStats] = useState<LabelingStats>({
    total_labeled: 0,
    labeled_as_theory: 0,
    labeled_as_not_theory: 0,
    skipped: 0,
    unreviewed: 0,
  });
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(false);

  // Auto-scroll logs to bottom
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

  // Load initial stats when switching to manual mode
  useEffect(() => {
    if (mode === 'manual') {
      loadLabelingStats();
      loadNextPaper();
    }
  }, [mode]);

  // Auto-classification handlers
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

  // Manual review handlers
  const loadLabelingStats = async () => {
    try {
      const response = await fetch(`${apiUrl}/papers/labeled`);
      const data = await response.json();
      if (data.success) {
        setLabelingStats(data.statistics);
      }
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadNextPaper = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/papers/unreviewed?limit=1`);
      const data = await response.json();
      if (data.papers && data.papers.length > 0) {
        setCurrentPaper(data.papers[0]);
        setComment('');
      } else {
        setCurrentPaper(null);
        alert('Нет статей для разметки!');
      }
    } catch (error) {
      console.error('Error loading paper:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLabel = async (isTheory: boolean) => {
    if (!currentPaper) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/papers/${currentPaper.pmc_id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: isTheory, comment }),
      });
      const data = await response.json();
      if (data.success) {
        await loadLabelingStats();
        await loadNextPaper();
      } else {
        alert('Ошибка сохранения: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error labeling paper:', error);
      alert('Ошибка сохранения: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleSkip = async () => {
    if (!currentPaper) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/papers/${currentPaper.pmc_id}/skip`, {
        method: 'POST',
      });
      const data = await response.json();
      if (data.success) {
        await loadLabelingStats();
        await loadNextPaper();
      } else {
        alert('Ошибка пропуска: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error skipping paper:', error);
      alert('Ошибка пропуска: ' + error);
    } finally {
      setLoading(false);
    }
  };

  // Keyboard shortcuts for manual mode
  useEffect(() => {
    if (mode !== 'manual' || !currentPaper) return;

    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement) return; // Don't trigger if typing in comment

      if (e.key === 'y' || e.key === 'Y' || e.key === 'д' || e.key === 'Д') {
        handleLabel(true);
      } else if (e.key === 'n' || e.key === 'N' || e.key === 'т' || e.key === 'Т') {
        handleLabel(false);
      } else if (e.key === 's' || e.key === 'S' || e.key === 'ы' || e.key === 'Ы') {
        handleSkip();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [mode, currentPaper, comment]);

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

        {/* Mode Switcher */}
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

      {/* Middle column - Text Preview */}
      <div>
        <h4 style={{ margin: '0 0 10px 0', color: '#f3f4f6', fontSize: '14px' }}>
          {mode === 'manual' && currentPaper ? (
            <>
              PMC{currentPaper.pmc_id}: {currentPaper.title}
              {currentPaper.is_aging_theory !== undefined && (
                <span
                  style={{
                    marginLeft: '10px',
                    padding: '2px 8px',
                    backgroundColor: currentPaper.is_aging_theory ? '#065f46' : '#7c2d12',
                    borderRadius: '3px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                  }}
                >
                  Авто: {currentPaper.is_aging_theory ? 'Теория' : 'Не теория'} (
                  {((currentPaper.classification_confidence || 0) * 100).toFixed(0)}%)
                </span>
              )}
            </>
          ) : (
            'Предпросмотр текста с подсветкой теорий'
          )}
        </h4>
        <div
          style={{
            height: '600px',
            overflowY: 'auto',
            backgroundColor: '#111827',
            padding: '15px',
            borderRadius: '4px',
            border: '1px solid #374151',
          }}
        >
          {mode === 'manual' && currentPaper ? (
            <HighlightedText text={currentPaper.full_text} spans={currentPaper.aging_theories || []} />
          ) : (
            <HighlightedText text={state.current_text_preview} spans={state.current_highlighted_spans} />
          )}
        </div>

        {mode === 'auto' && state.current_highlighted_spans.length > 0 && (
          <div
            style={{
              marginTop: '10px',
              padding: '10px',
              backgroundColor: '#111827',
              borderRadius: '4px',
              fontSize: '12px',
            }}
          >
            <div style={{ color: '#fbbf24', fontWeight: 'bold', marginBottom: '5px' }}>
              Найдено теорий в предпросмотре: {state.current_highlighted_spans.length}
            </div>
            <div style={{ color: '#d1d5db', fontSize: '11px' }}>
              {Array.from(new Set(state.current_highlighted_spans.map((s) => s.theory_name)))
                .slice(0, 5)
                .join(', ')}
            </div>
          </div>
        )}

        {mode === 'manual' && currentPaper && currentPaper.aging_theories && currentPaper.aging_theories.length > 0 && (
          <div
            style={{
              marginTop: '10px',
              padding: '10px',
              backgroundColor: '#111827',
              borderRadius: '4px',
              fontSize: '12px',
            }}
          >
            <div style={{ color: '#fbbf24', fontWeight: 'bold', marginBottom: '5px' }}>
              Найдено теорий: {currentPaper.aging_theories.length}
            </div>
            <div style={{ color: '#d1d5db', fontSize: '11px' }}>
              {Array.from(new Set(currentPaper.aging_theories.map((s) => s.theory_name)))
                .slice(0, 5)
                .join(', ')}
            </div>
          </div>
        )}
      </div>

      {/* Right column - Statistics and Controls */}
      <div>
        {mode === 'auto' ? (
          <>
            {/* Message */}
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

            {/* Classified Papers Count */}
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

            {/* Classification Statistics */}
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

                {/* Classification Stats Grid */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '10px',
                    marginTop: '10px',
                  }}
                >
                  <div>
                    <div style={{ color: '#6b7280', fontSize: '11px' }}>Теории старения</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#4ade80' }}>
                      {state.classified_as_theory.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6b7280', fontSize: '11px' }}>Не о теориях</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f59e0b' }}>
                      {state.classified_as_not_theory.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6b7280', fontSize: '11px' }}>Найдено теорий</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#fbbf24' }}>
                      {state.total_theories_found.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6b7280', fontSize: '11px' }}>Ошибки</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#ef4444' }}>
                      {state.errors.toLocaleString()}
                    </div>
                  </div>
                </div>

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

            {/* Controls */}
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
          <>
            {/* Manual Review Mode */}
            {/* Labeling Statistics */}
            <div
              style={{
                backgroundColor: '#111827',
                padding: '15px',
                borderRadius: '4px',
                marginBottom: '15px',
              }}
            >
              <div style={{ color: '#f3f4f6', fontWeight: 'bold', marginBottom: '10px', fontSize: '14px' }}>
                Статистика разметки
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '10px',
                }}
              >
                <div>
                  <div style={{ color: '#6b7280', fontSize: '11px' }}>Размечено теорий</div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#4ade80' }}>
                    {labelingStats.labeled_as_theory.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6b7280', fontSize: '11px' }}>Размечено не теорий</div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f59e0b' }}>
                    {labelingStats.labeled_as_not_theory.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6b7280', fontSize: '11px' }}>Пропущено</div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#9ca3af' }}>
                    {labelingStats.skipped.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#6b7280', fontSize: '11px' }}>Не размечено</div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#3b82f6' }}>
                    {labelingStats.unreviewed.toLocaleString()}
                  </div>
                </div>
              </div>
              <div style={{ marginTop: '10px', borderTop: '1px solid #374151', paddingTop: '10px' }}>
                <div style={{ color: '#6b7280', fontSize: '11px' }}>Всего размечено</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fbbf24' }}>
                  {labelingStats.total_labeled.toLocaleString()}
                </div>
              </div>
            </div>

            {/* Labeling Controls */}
            {currentPaper ? (
              <>
                <div
                  style={{
                    backgroundColor: '#111827',
                    padding: '15px',
                    borderRadius: '4px',
                    marginBottom: '15px',
                  }}
                >
                  <div style={{ color: '#f3f4f6', fontWeight: 'bold', marginBottom: '10px', fontSize: '14px' }}>
                    Комментарий (опционально)
                  </div>
                  <textarea
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                    placeholder="Добавьте комментарий к разметке..."
                    style={{
                      width: '100%',
                      height: '80px',
                      backgroundColor: '#374151',
                      color: '#d1d5db',
                      border: '1px solid #4b5563',
                      borderRadius: '4px',
                      padding: '8px',
                      fontSize: '12px',
                      fontFamily: 'monospace',
                      resize: 'none',
                    }}
                  />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <button
                    onClick={() => handleLabel(true)}
                    disabled={loading}
                    style={{
                      padding: '15px',
                      backgroundColor: loading ? '#374151' : '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '14px',
                    }}
                  >
                    ✓ Это теория старения (Y)
                  </button>
                  <button
                    onClick={() => handleLabel(false)}
                    disabled={loading}
                    style={{
                      padding: '15px',
                      backgroundColor: loading ? '#374151' : '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '14px',
                    }}
                  >
                    ✗ Не теория старения (N)
                  </button>
                  <button
                    onClick={handleSkip}
                    disabled={loading}
                    style={{
                      padding: '15px',
                      backgroundColor: loading ? '#374151' : '#6b7280',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '14px',
                    }}
                  >
                    ⏭ Пропустить (S)
                  </button>
                </div>

                <div
                  style={{
                    marginTop: '15px',
                    padding: '10px',
                    backgroundColor: '#111827',
                    borderRadius: '4px',
                    fontSize: '11px',
                    color: '#9ca3af',
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Горячие клавиши:</div>
                  <div>Y - Теория старения</div>
                  <div>N - Не теория</div>
                  <div>S - Пропустить</div>
                </div>
              </>
            ) : (
              <div
                style={{
                  backgroundColor: '#111827',
                  padding: '30px',
                  borderRadius: '4px',
                  textAlign: 'center',
                  color: '#9ca3af',
                }}
              >
                {loading ? 'Загрузка...' : 'Нет статей для разметки'}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
