import React, { useEffect, useState } from 'react';

interface ValidationPaper {
  paper_url: string;
  title: string;
  year: string;
  is_manually_annotated: boolean;
  has_predictions: boolean;
  validation_timestamp?: string;
  questions_timestamp?: string;
}

interface QuestionMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
  confusion_matrix: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
  };
}

interface OverallMetrics {
  accuracy_micro: number;
  accuracy_macro: number;
  accuracy_weighted: number;
  precision_micro: number;
  recall_micro: number;
  f1_micro: number;
  f1_macro: number;
  f1_weighted: number;
  total_support: number;
}

interface PaperComparison {
  paper_url: string;
  paper_name: string;
  paper_year: string;
  questions: {
    [key: string]: {
      manual: string | null;
      predicted: string | null;
      match: boolean | null;
    };
  };
}

interface ValidationMetrics {
  overall: OverallMetrics;
  per_question: {
    [key: string]: QuestionMetrics;
  };
  paper_comparisons: PaperComparison[];
  num_papers: number;
}

interface ValidationMetricsViewerProps {
  apiUrl: string;
}

// Компонент тултипа с информацией о метрике
const MetricTooltip: React.FC<{ text: string; children: React.ReactNode }> = ({ text, children }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <div
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        style={{ cursor: 'help', display: 'inline-flex', alignItems: 'center' }}
      >
        {children}
      </div>
      {showTooltip && (
        <div style={{
          position: 'absolute',
          bottom: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginBottom: '8px',
          padding: '12px',
          backgroundColor: '#1f2937',
          border: '1px solid #4b5563',
          borderRadius: '6px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          minWidth: '300px',
          maxWidth: '400px',
          fontSize: '12px',
          lineHeight: '1.5',
          color: '#d1d5db',
          whiteSpace: 'normal',
          pointerEvents: 'none'
        }}>
          {text}
          <div style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 0,
            height: 0,
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            borderTop: '6px solid #4b5563'
          }} />
        </div>
      )}
    </div>
  );
};

// Описания метрик
const METRIC_DESCRIPTIONS = {
  accuracy: "Accuracy (Точность) — доля правильных предсказаний среди всех предсказаний.\n\nДиапазон: 0.0-1.0 (0%-100%)\n\nИнтерпретация:\n• >0.9 (90%) — отлично\n• 0.7-0.9 — хорошо\n• <0.7 — требует улучшения\n\nОсобенность: Может быть обманчива при несбалансированных классах. Например, если 95% ответов \"Да\", модель всегда отвечающая \"Да\" получит 95% accuracy.",

  precision: "Precision (Точность положительных) — из всех предсказанных \"Да\", какая доля действительно \"Да\".\n\nДиапазон: 0.0-1.0 (0%-100%)\n\nФормула: TP / (TP + FP)\n\nИнтерпретация:\n• Высокая precision — мало ложных срабатываний (FP)\n• Низкая precision — модель часто ошибочно говорит \"Да\"\n\nПример: Если модель сказала \"Да\" 10 раз, и 9 раз была права — precision = 90%.",

  recall: "Recall (Полнота, Чувствительность) — из всех истинных \"Да\", какую долю модель нашла.\n\nДиапазон: 0.0-1.0 (0%-100%)\n\nФормула: TP / (TP + FN)\n\nИнтерпретация:\n• Высокий recall — модель находит почти все \"Да\"\n• Низкий recall — модель пропускает много \"Да\" (много FN)\n\nПример: Если было 10 истинных \"Да\", и модель нашла 8 — recall = 80%.",

  f1_score: "F1-Score — гармоническое среднее между Precision и Recall. Главная метрика для оценки качества!\n\nДиапазон: 0.0-1.0 (0%-100%)\n\nФормула: 2 × (Precision × Recall) / (Precision + Recall)\n\nИнтерпретация:\n• >0.9 (90%) — отличная модель\n• 0.7-0.9 — хорошая модель\n• 0.5-0.7 — средняя модель\n• <0.5 — плохая модель\n\nПочему важна: Балансирует между precision и recall. Модель с высоким F1 хороша и в точности, и в полноте.",

  f1_macro: "F1-Score (Macro) — среднее арифметическое F1 по всем вопросам.\n\nРасчет: Считает F1 для каждого вопроса отдельно, потом усредняет.\n\nОсобенность: Все вопросы имеют равный вес независимо от количества примеров.\n\nКогда использовать: Когда важно качество на каждом вопросе одинаково.",

  f1_weighted: "F1-Score (Weighted) — взвешенное среднее F1 по всем вопросам.\n\nРасчет: Считает F1 для каждого вопроса, затем усредняет с весами пропорциональными количеству примеров.\n\nОсобенность: Вопросы с большим количеством данных влияют сильнее.\n\nКогда использовать: Наиболее объективная общая оценка качества модели.",

  micro: "Micro-averaging — подсчет метрик глобально по всем вопросам сразу.\n\nРасчет: Суммирует все TP, FP, FN по всем вопросам, затем считает метрику.\n\nОсобенность: Вопросы с большим количеством примеров доминируют.\n\nОтличие от Macro: Macro считает метрику для каждого вопроса отдельно, потом усредняет.",

  confusion_matrix: "Confusion Matrix (Матрица ошибок):\n\n• TP (True Positive) — правильно предсказано \"Да\"\n• TN (True Negative) — правильно предсказано \"Нет\"\n• FP (False Positive) — ошибочно предсказано \"Да\" (должно быть \"Нет\")\n• FN (False Negative) — ошибочно предсказано \"Нет\" (должно быть \"Да\")\n\nИдеал: высокие TP и TN, низкие FP и FN."
};

export const ValidationMetricsViewer: React.FC<ValidationMetricsViewerProps> = ({ apiUrl }) => {
  const [papers, setPapers] = useState<ValidationPaper[]>([]);
  const [metrics, setMetrics] = useState<ValidationMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [classifying, setClassifying] = useState(false);
  const [selectedPaper, setSelectedPaper] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Загрузить список валидационных статей
  const loadPapers = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/validation/papers`);
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setPapers(data.papers || []);
      }
    } catch (err) {
      setError(`Failed to load papers: ${err}`);
    }
  };

  // Загрузить метрики
  const loadMetrics = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/api/validation/metrics`);
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setMetrics(data);
      }
    } catch (err) {
      setError(`Failed to load metrics: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Запустить классификацию
  const runClassification = async () => {
    setClassifying(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/api/validation/classify`, {
        method: 'POST'
      });
      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        // Обновить данные
        await loadPapers();
        await loadMetrics();
      }
    } catch (err) {
      setError(`Failed to run classification: ${err}`);
    } finally {
      setClassifying(false);
    }
  };

  useEffect(() => {
    loadPapers();
    loadMetrics();
  }, []);

  // Форматирование процентов
  const formatPercent = (value: number) => {
    return (value * 100).toFixed(2) + '%';
  };

  // Цвет для метрики (темная тема)
  const getMetricColor = (value: number) => {
    if (value >= 0.9) return '#4ade80'; // green
    if (value >= 0.7) return '#fbbf24'; // yellow
    return '#f87171'; // red
  };

  return (
    <div style={{
      padding: '15px',
      marginBottom: '20px',
      border: '1px solid #374151',
      borderRadius: '8px',
      backgroundColor: '#1f2937'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px'
      }}>
        <h3 style={{ margin: 0, fontSize: '20px', color: '#f3f4f6' }}>
          Validation Metrics
        </h3>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={runClassification}
            disabled={classifying || papers.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: classifying || papers.length === 0 ? '#4b5563' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: classifying || papers.length === 0 ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            {classifying ? 'Classifying...' : 'Run Classification'}
          </button>
          <button
            onClick={loadMetrics}
            disabled={loading}
            style={{
              padding: '8px 16px',
              backgroundColor: loading ? '#4b5563' : '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            {loading ? 'Loading...' : 'Refresh Metrics'}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div style={{
          backgroundColor: '#7f1d1d',
          border: '1px solid #991b1b',
          color: '#fca5a5',
          padding: '12px',
          borderRadius: '4px',
          marginBottom: '15px',
          fontSize: '14px'
        }}>
          {error}
        </div>
      )}

      {/* Validation Papers Table */}
      <div style={{
        backgroundColor: '#111827',
        borderRadius: '6px',
        padding: '15px',
        marginBottom: '15px'
      }}>
        <h4 style={{
          margin: '0 0 12px 0',
          fontSize: '16px',
          color: '#f3f4f6',
          fontWeight: '600'
        }}>
          Validation Papers ({papers.length})
        </h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            fontSize: '13px',
            borderCollapse: 'collapse',
            color: '#d1d5db'
          }}>
            <thead>
              <tr style={{ backgroundColor: '#1f2937', borderBottom: '1px solid #374151' }}>
                <th style={{ padding: '10px', textAlign: 'left', color: '#9ca3af' }}>Title</th>
                <th style={{ padding: '10px', textAlign: 'left', color: '#9ca3af' }}>Year</th>
                <th style={{ padding: '10px', textAlign: 'center', color: '#9ca3af' }}>Manual</th>
                <th style={{ padding: '10px', textAlign: 'center', color: '#9ca3af' }}>Predicted</th>
                <th style={{ padding: '10px', textAlign: 'left', color: '#9ca3af' }}>URL</th>
              </tr>
            </thead>
            <tbody>
              {papers.map((paper, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: '1px solid #374151',
                    cursor: 'pointer'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#1f2937';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                  onClick={() => setSelectedPaper(paper.paper_url)}
                >
                  <td style={{ padding: '10px' }}>{paper.title}</td>
                  <td style={{ padding: '10px' }}>{paper.year}</td>
                  <td style={{ padding: '10px', textAlign: 'center', color: paper.is_manually_annotated ? '#4ade80' : '#f87171' }}>
                    {paper.is_manually_annotated ? '✓' : '✗'}
                  </td>
                  <td style={{ padding: '10px', textAlign: 'center', color: paper.has_predictions ? '#4ade80' : '#f87171' }}>
                    {paper.has_predictions ? '✓' : '✗'}
                  </td>
                  <td style={{ padding: '10px', fontSize: '11px' }}>
                    <a
                      href={paper.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: '#3b82f6', textDecoration: 'none' }}
                      onClick={(e) => e.stopPropagation()}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.textDecoration = 'underline';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.textDecoration = 'none';
                      }}
                    >
                      {paper.paper_url.length > 60 ? paper.paper_url.substring(0, 60) + '...' : paper.paper_url}
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Overall Metrics */}
      {metrics && metrics.overall && (
        <div style={{
          backgroundColor: '#111827',
          borderRadius: '6px',
          padding: '15px',
          marginBottom: '15px'
        }}>
          <h4 style={{
            margin: '0 0 12px 0',
            fontSize: '16px',
            color: '#f3f4f6',
            fontWeight: '600'
          }}>
            Overall Metrics
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '12px',
            marginBottom: '15px'
          }}>
            {/* Accuracy (Weighted) */}
            <div style={{
              backgroundColor: '#1e3a5f',
              padding: '12px',
              borderRadius: '6px',
              border: '1px solid #2563eb'
            }}>
              <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                Accuracy (Weighted)
                <MetricTooltip text={METRIC_DESCRIPTIONS.accuracy}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    border: '1px solid #6b7280',
                    fontSize: '11px',
                    color: '#9ca3af',
                    fontWeight: 'bold'
                  }}>?</span>
                </MetricTooltip>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: getMetricColor(metrics.overall.accuracy_weighted) }}>
                {formatPercent(metrics.overall.accuracy_weighted)}
              </div>
            </div>

            {/* F1-Score (Macro) - ЗОЛОТАЯ КАРТОЧКА */}
            <div style={{
              backgroundColor: '#3d2817',
              padding: '12px',
              borderRadius: '6px',
              border: '2px solid #f59e0b',
              boxShadow: '0 0 12px rgba(245, 158, 11, 0.3)'
            }}>
              <div style={{ fontSize: '12px', color: '#fbbf24', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontWeight: '700' }}>⭐ F1-Score (Macro)</span>
                <MetricTooltip text={METRIC_DESCRIPTIONS.f1_macro}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    border: '1px solid #fbbf24',
                    fontSize: '11px',
                    color: '#fbbf24',
                    fontWeight: 'bold'
                  }}>?</span>
                </MetricTooltip>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: getMetricColor(metrics.overall.f1_macro) }}>
                {formatPercent(metrics.overall.f1_macro)}
              </div>
            </div>

            {/* F1-Score (Weighted) - ЗОЛОТАЯ КАРТОЧКА */}
            <div style={{
              backgroundColor: '#3d2817',
              padding: '12px',
              borderRadius: '6px',
              border: '2px solid #f59e0b',
              boxShadow: '0 0 12px rgba(245, 158, 11, 0.3)'
            }}>
              <div style={{ fontSize: '12px', color: '#fbbf24', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontWeight: '700' }}>⭐ F1-Score (Weighted)</span>
                <MetricTooltip text={METRIC_DESCRIPTIONS.f1_weighted}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    border: '1px solid #fbbf24',
                    fontSize: '11px',
                    color: '#fbbf24',
                    fontWeight: 'bold'
                  }}>?</span>
                </MetricTooltip>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: getMetricColor(metrics.overall.f1_weighted) }}>
                {formatPercent(metrics.overall.f1_weighted)}
              </div>
            </div>

            {/* Total Papers */}
            <div style={{
              backgroundColor: '#1f2937',
              padding: '12px',
              borderRadius: '6px',
              border: '1px solid #4b5563'
            }}>
              <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '4px' }}>Total Papers</div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f3f4f6' }}>
                {metrics.num_papers}
              </div>
            </div>
          </div>

          {/* Additional metrics */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '10px',
            fontSize: '13px',
            color: '#d1d5db'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#9ca3af' }}>Accuracy (Micro):</span>
              <MetricTooltip text={METRIC_DESCRIPTIONS.micro}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  border: '1px solid #6b7280',
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontWeight: 'bold'
                }}>?</span>
              </MetricTooltip>
              <span style={{ fontWeight: '600', color: getMetricColor(metrics.overall.accuracy_micro) }}>
                {formatPercent(metrics.overall.accuracy_micro)}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#9ca3af' }}>Accuracy (Macro):</span>
              <MetricTooltip text={METRIC_DESCRIPTIONS.accuracy}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  border: '1px solid #6b7280',
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontWeight: 'bold'
                }}>?</span>
              </MetricTooltip>
              <span style={{ fontWeight: '600', color: getMetricColor(metrics.overall.accuracy_macro) }}>
                {formatPercent(metrics.overall.accuracy_macro)}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#9ca3af' }}>Precision (Micro):</span>
              <MetricTooltip text={METRIC_DESCRIPTIONS.precision}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  border: '1px solid #6b7280',
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontWeight: 'bold'
                }}>?</span>
              </MetricTooltip>
              <span style={{ fontWeight: '600', color: getMetricColor(metrics.overall.precision_micro) }}>
                {formatPercent(metrics.overall.precision_micro)}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#9ca3af' }}>Recall (Micro):</span>
              <MetricTooltip text={METRIC_DESCRIPTIONS.recall}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  border: '1px solid #6b7280',
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontWeight: 'bold'
                }}>?</span>
              </MetricTooltip>
              <span style={{ fontWeight: '600', color: getMetricColor(metrics.overall.recall_micro) }}>
                {formatPercent(metrics.overall.recall_micro)}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#9ca3af' }}>F1 (Micro):</span>
              <MetricTooltip text={METRIC_DESCRIPTIONS.f1_score}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '14px',
                  height: '14px',
                  borderRadius: '50%',
                  border: '1px solid #6b7280',
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontWeight: 'bold'
                }}>?</span>
              </MetricTooltip>
              <span style={{ fontWeight: '600', color: getMetricColor(metrics.overall.f1_micro) }}>
                {formatPercent(metrics.overall.f1_micro)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Per-Question Metrics */}
      {metrics && metrics.per_question && (
        <div style={{
          backgroundColor: '#111827',
          borderRadius: '6px',
          padding: '15px',
          marginBottom: '15px'
        }}>
          <h4 style={{
            margin: '0 0 12px 0',
            fontSize: '16px',
            color: '#f3f4f6',
            fontWeight: '600'
          }}>
            Per-Question Metrics
          </h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              fontSize: '12px',
              borderCollapse: 'collapse',
              color: '#d1d5db'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#1f2937', borderBottom: '1px solid #374151' }}>
                  <th style={{ padding: '10px', textAlign: 'left', color: '#9ca3af' }}>Question</th>
                  <th style={{ padding: '10px', textAlign: 'right', color: '#9ca3af' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      Accuracy
                      <MetricTooltip text={METRIC_DESCRIPTIONS.accuracy}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          border: '1px solid #6b7280',
                          fontSize: '10px',
                          color: '#9ca3af',
                          fontWeight: 'bold'
                        }}>?</span>
                      </MetricTooltip>
                    </div>
                  </th>
                  <th style={{ padding: '10px', textAlign: 'right', color: '#9ca3af' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      Precision
                      <MetricTooltip text={METRIC_DESCRIPTIONS.precision}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          border: '1px solid #6b7280',
                          fontSize: '10px',
                          color: '#9ca3af',
                          fontWeight: 'bold'
                        }}>?</span>
                      </MetricTooltip>
                    </div>
                  </th>
                  <th style={{ padding: '10px', textAlign: 'right', color: '#9ca3af' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      Recall
                      <MetricTooltip text={METRIC_DESCRIPTIONS.recall}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          border: '1px solid #6b7280',
                          fontSize: '10px',
                          color: '#9ca3af',
                          fontWeight: 'bold'
                        }}>?</span>
                      </MetricTooltip>
                    </div>
                  </th>
                  <th style={{ padding: '10px', textAlign: 'right', color: '#fbbf24' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      ⭐ F1-Score
                      <MetricTooltip text={METRIC_DESCRIPTIONS.f1_score}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          border: '1px solid #fbbf24',
                          fontSize: '10px',
                          color: '#fbbf24',
                          fontWeight: 'bold'
                        }}>?</span>
                      </MetricTooltip>
                    </div>
                  </th>
                  <th style={{ padding: '10px', textAlign: 'right', color: '#9ca3af' }}>Support</th>
                  <th style={{ padding: '10px', textAlign: 'center', color: '#9ca3af' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                      Confusion Matrix
                      <MetricTooltip text={METRIC_DESCRIPTIONS.confusion_matrix}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          border: '1px solid #6b7280',
                          fontSize: '10px',
                          color: '#9ca3af',
                          fontWeight: 'bold'
                        }}>?</span>
                      </MetricTooltip>
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_question).map(([questionId, questionMetrics]) => (
                  <tr key={questionId} style={{ borderBottom: '1px solid #374151' }}>
                    <td style={{ padding: '10px', fontWeight: '600' }}>{questionId}</td>
                    <td style={{ padding: '10px', textAlign: 'right', color: getMetricColor(questionMetrics.accuracy) }}>
                      {formatPercent(questionMetrics.accuracy)}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right', color: getMetricColor(questionMetrics.precision) }}>
                      {formatPercent(questionMetrics.precision)}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right', color: getMetricColor(questionMetrics.recall) }}>
                      {formatPercent(questionMetrics.recall)}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right', color: getMetricColor(questionMetrics.f1_score) }}>
                      {formatPercent(questionMetrics.f1_score)}
                    </td>
                    <td style={{ padding: '10px', textAlign: 'right' }}>{questionMetrics.support}</td>
                    <td style={{ padding: '10px' }}>
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '4px',
                        fontSize: '11px'
                      }}>
                        <div style={{
                          backgroundColor: '#1e4620',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          border: '1px solid #22c55e'
                        }}>
                          TP: {questionMetrics.confusion_matrix.tp}
                        </div>
                        <div style={{
                          backgroundColor: '#7f1d1d',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          border: '1px solid #ef4444'
                        }}>
                          FP: {questionMetrics.confusion_matrix.fp}
                        </div>
                        <div style={{
                          backgroundColor: '#7f1d1d',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          border: '1px solid #ef4444'
                        }}>
                          FN: {questionMetrics.confusion_matrix.fn}
                        </div>
                        <div style={{
                          backgroundColor: '#1e4620',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          border: '1px solid #22c55e'
                        }}>
                          TN: {questionMetrics.confusion_matrix.tn}
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Paper-by-Paper Comparison */}
      {metrics && metrics.paper_comparisons && metrics.paper_comparisons.length > 0 && (
        <div style={{
          backgroundColor: '#111827',
          borderRadius: '6px',
          padding: '15px',
          marginBottom: '15px'
        }}>
          <h4 style={{
            margin: '0 0 12px 0',
            fontSize: '16px',
            color: '#f3f4f6',
            fontWeight: '600'
          }}>
            Paper-by-Paper Comparison
          </h4>
          {metrics.paper_comparisons.map((comparison, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: '20px',
                paddingBottom: '20px',
                borderBottom: idx < metrics.paper_comparisons.length - 1 ? '1px solid #374151' : 'none'
              }}
            >
              <h5 style={{
                margin: '0 0 8px 0',
                fontSize: '14px',
                color: '#f3f4f6',
                fontWeight: '600'
              }}>
                {comparison.paper_name} ({comparison.paper_year})
              </h5>
              <div style={{
                fontSize: '11px',
                color: '#6b7280',
                marginBottom: '12px',
                wordBreak: 'break-all'
              }}>
                {comparison.paper_url}
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
                gap: '8px'
              }}>
                {Object.entries(comparison.questions).map(([questionId, result]) => {
                  let bgColor = '#1f2937';
                  let borderColor = '#4b5563';

                  if (result.match === true) {
                    bgColor = '#1e4620';
                    borderColor = '#22c55e';
                  } else if (result.match === false) {
                    bgColor = '#7f1d1d';
                    borderColor = '#ef4444';
                  }

                  return (
                    <div
                      key={questionId}
                      style={{
                        padding: '8px',
                        borderRadius: '4px',
                        backgroundColor: bgColor,
                        border: `1px solid ${borderColor}`,
                        textAlign: 'center',
                        fontSize: '12px',
                        color: '#d1d5db'
                      }}
                    >
                      <div style={{ fontWeight: '600', marginBottom: '4px' }}>{questionId}</div>
                      <div style={{ fontSize: '10px', marginBottom: '4px' }}>
                        <div>M: {result.manual || '?'}</div>
                        <div>P: {result.predicted || '?'}</div>
                      </div>
                      <div style={{ fontSize: '16px' }}>
                        {result.match === true ? '✓' : result.match === false ? '✗' : '?'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* No metrics message */}
      {!metrics && !loading && (
        <div style={{
          backgroundColor: '#78350f',
          border: '1px solid #92400e',
          color: '#fbbf24',
          padding: '12px',
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          No metrics available. Run classification first or check if validation papers have predictions.
        </div>
      )}
    </div>
  );
};
