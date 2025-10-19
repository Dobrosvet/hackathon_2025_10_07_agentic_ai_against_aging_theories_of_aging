import React, { useState, useEffect } from 'react'

interface DatabaseViewerProps {
  name: string
  wsUrl: string
  apiUrl: string
}

interface Statistics {
  total_papers: number
  total_theories: number
  papers_with_questions_classification: number
  papers_with_criteria_classification: number
  papers_fully_classified: number
  classification_progress_percent: number
}

interface TablePreview {
  table_number: number
  table_name: string
  preview: any[]
  total_rows: number
  showing_rows: number
}

interface TableFull {
  table_number: number
  table_name: string
  data: any[]
  pagination: {
    page: number
    page_size: number
    total_rows: number
    total_pages: number
    has_next: boolean
    has_prev: boolean
  }
}

export const DatabaseViewer: React.FC<DatabaseViewerProps> = ({ name, wsUrl, apiUrl }) => {
  const [connected, setConnected] = useState(false)
  const [statistics, setStatistics] = useState<Statistics | null>(null)
  const [selectedTable, setSelectedTable] = useState<number>(1)
  const [tablePreviews, setTablePreviews] = useState<{ [key: number]: TablePreview }>({})
  const [fullTable, setFullTable] = useState<TableFull | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(50)
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setConnected(true)
      console.log(`Connected to ${name}`)
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)

      if (message.type === 'state') {
        if (message.data.statistics) {
          setStatistics(message.data.statistics)
        }
        if (message.data.last_update) {
          setLastUpdate(message.data.last_update)
        }
      }
    }

    ws.onerror = (error) => {
      console.error(`WebSocket error for ${name}:`, error)
    }

    ws.onclose = () => {
      setConnected(false)
      console.log(`Disconnected from ${name}`)
    }

    return () => {
      ws.close()
    }
  }, [wsUrl, name])

  // Load initial data
  useEffect(() => {
    loadStatistics()
    loadTablePreviews()
  }, [])

  // Load full table when selection changes
  useEffect(() => {
    if (selectedTable) {
      loadFullTable(selectedTable, currentPage)
    }
  }, [selectedTable, currentPage])

  const loadStatistics = async () => {
    try {
      const response = await fetch(`${apiUrl}/statistics`)
      const data = await response.json()
      setStatistics(data)
    } catch (error) {
      console.error('Error loading statistics:', error)
    }
  }

  const loadTablePreviews = async () => {
    try {
      const previews: { [key: number]: TablePreview } = {}
      for (let i = 1; i <= 3; i++) {
        const response = await fetch(`${apiUrl}/tables/preview/${i}`)
        const data = await response.json()
        previews[i] = data
      }
      setTablePreviews(previews)
    } catch (error) {
      console.error('Error loading table previews:', error)
    }
  }

  const loadFullTable = async (tableNumber: number, page: number) => {
    setLoading(true)
    try {
      const response = await fetch(
        `${apiUrl}/tables/full/${tableNumber}?page=${page}&page_size=${pageSize}`
      )
      const data = await response.json()
      setFullTable(data)
    } catch (error) {
      console.error('Error loading full table:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleExportTable = async (tableNumber: number, format: 'csv' | 'xlsx') => {
    setExporting(true)
    try {
      const response = await fetch(`${apiUrl}/export/${tableNumber}?format=${format}`, {
        method: 'POST'
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `table_${tableNumber}.${format}`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Error exporting table:', error)
    } finally {
      setExporting(false)
    }
  }

  const handleExportAll = async () => {
    setExporting(true)
    try {
      const response = await fetch(`${apiUrl}/export/all`, {
        method: 'POST'
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'all_tables.xlsx'
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Error exporting all tables:', error)
    } finally {
      setExporting(false)
    }
  }

  const handleRefresh = async () => {
    setLoading(true)
    try {
      await fetch(`${apiUrl}/refresh`, { method: 'POST' })
      await loadStatistics()
      await loadTablePreviews()
      if (selectedTable) {
        await loadFullTable(selectedTable, currentPage)
      }
    } catch (error) {
      console.error('Error refreshing data:', error)
    } finally {
      setLoading(false)
    }
  }

  const renderTablePreview = (tableNumber: number) => {
    const preview = tablePreviews[tableNumber]
    if (!preview) return null

    const tableNames = {
      1: 'Table 1: Aging Theories',
      2: 'Table 2: Collected Papers',
      3: 'Table 3: Papers Analysis (Q1-Q9, C1-C4)'
    }

    return (
      <div
        key={tableNumber}
        onClick={() => {
          setSelectedTable(tableNumber)
          setCurrentPage(1)
        }}
        style={{
          border: selectedTable === tableNumber ? '2px solid #3b82f6' : '1px solid #374151',
          borderRadius: '8px',
          padding: '15px',
          marginBottom: '15px',
          cursor: 'pointer',
          backgroundColor: selectedTable === tableNumber ? '#1e3a5f' : '#1f2937',
          transition: 'all 0.2s'
        }}
      >
        <h3 style={{ margin: '0 0 10px 0', color: '#60a5fa', fontSize: '16px' }}>
          {tableNames[tableNumber as keyof typeof tableNames]}
        </h3>
        <div style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '10px' }}>
          Total rows: {preview.total_rows}
        </div>
        <div style={{ fontSize: '12px', color: '#6b7280' }}>
          Preview: {preview.showing_rows} of {preview.total_rows} rows
        </div>
        {preview.preview.length > 0 && (
          <div style={{ marginTop: '10px', fontSize: '11px', color: '#4b5563' }}>
            Columns: {Object.keys(preview.preview[0]).join(', ')}
          </div>
        )}
      </div>
    )
  }

  const renderFullTable = () => {
    if (!fullTable || loading) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#9ca3af' }}>
          {loading ? 'Loading...' : 'Select a table to view'}
        </div>
      )
    }

    if (fullTable.data.length === 0) {
      return (
        <div style={{ textAlign: 'center', padding: '40px', color: '#9ca3af' }}>
          No data available
        </div>
      )
    }

    const columns = Object.keys(fullTable.data[0])

    return (
      <div>
        <div style={{ marginBottom: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, color: '#60a5fa' }}>{fullTable.table_name}</h3>
          <div style={{ color: '#9ca3af', fontSize: '14px' }}>
            Showing {fullTable.data.length} of {fullTable.pagination.total_rows} rows
          </div>
        </div>

        <div style={{ overflowX: 'auto', marginBottom: '15px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ backgroundColor: '#374151' }}>
                {columns.map((col) => (
                  <th
                    key={col}
                    style={{
                      padding: '10px',
                      textAlign: 'left',
                      borderBottom: '2px solid #4b5563',
                      color: '#f3f4f6',
                      fontWeight: '600',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {fullTable.data.map((row, idx) => (
                <tr
                  key={idx}
                  style={{
                    backgroundColor: idx % 2 === 0 ? '#1f2937' : '#111827',
                    borderBottom: '1px solid #374151'
                  }}
                >
                  {columns.map((col) => (
                    <td
                      key={col}
                      style={{
                        padding: '8px 10px',
                        color: '#d1d5db',
                        maxWidth: '300px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}
                      title={String(row[col])}
                    >
                      {String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ color: '#9ca3af', fontSize: '14px' }}>
            Page {fullTable.pagination.page} of {fullTable.pagination.total_pages}
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={!fullTable.pagination.has_prev}
              style={{
                padding: '8px 16px',
                backgroundColor: fullTable.pagination.has_prev ? '#3b82f6' : '#374151',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: fullTable.pagination.has_prev ? 'pointer' : 'not-allowed',
                fontSize: '14px'
              }}
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={!fullTable.pagination.has_next}
              style={{
                padding: '8px 16px',
                backgroundColor: fullTable.pagination.has_next ? '#3b82f6' : '#374151',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: fullTable.pagination.has_next ? 'pointer' : 'not-allowed',
                fontSize: '14px'
              }}
            >
              Next
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ marginBottom: '30px', backgroundColor: '#111827', borderRadius: '8px', padding: '20px' }}>
      {/* Header */}
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: '0 0 5px 0', color: '#f3f4f6' }}>{name}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: connected ? '#10b981' : '#ef4444'
              }}
            />
            <span style={{ color: '#9ca3af', fontSize: '14px' }}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
            {lastUpdate && (
              <span style={{ color: '#6b7280', fontSize: '12px' }}>
                Last update: {new Date(lastUpdate).toLocaleString()}
              </span>
            )}
          </div>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          style={{
            padding: '10px 20px',
            backgroundColor: '#3b82f6',
            color: '#f3f4f6',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '14px'
          }}
        >
          {loading ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </div>

      {/* 3-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr 1fr', gap: '20px' }}>
        {/* Column 1: Table Previews */}
        <div style={{ backgroundColor: '#1f2937', borderRadius: '8px', padding: '15px' }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#f3f4f6', fontSize: '16px' }}>
            Table Previews
          </h3>
          {renderTablePreview(1)}
          {renderTablePreview(2)}
          {renderTablePreview(3)}
        </div>

        {/* Column 2: Full Table View */}
        <div style={{ backgroundColor: '#1f2937', borderRadius: '8px', padding: '15px' }}>
          {renderFullTable()}
        </div>

        {/* Column 3: Statistics and Export */}
        <div style={{ backgroundColor: '#1f2937', borderRadius: '8px', padding: '15px' }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#f3f4f6', fontSize: '16px' }}>
            Statistics
          </h3>

          {statistics && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#111827', borderRadius: '4px' }}>
                <div style={{ color: '#9ca3af', fontSize: '12px' }}>Total Papers</div>
                <div style={{ color: '#f3f4f6', fontSize: '24px', fontWeight: 'bold' }}>
                  {statistics.total_papers}
                </div>
              </div>

              <div style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#111827', borderRadius: '4px' }}>
                <div style={{ color: '#9ca3af', fontSize: '12px' }}>Total Theories</div>
                <div style={{ color: '#f3f4f6', fontSize: '24px', fontWeight: 'bold' }}>
                  {statistics.total_theories}
                </div>
              </div>

              <div style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#111827', borderRadius: '4px' }}>
                <div style={{ color: '#9ca3af', fontSize: '12px' }}>Fully Classified</div>
                <div style={{ color: '#f3f4f6', fontSize: '24px', fontWeight: 'bold' }}>
                  {statistics.papers_fully_classified}
                </div>
              </div>

              <div style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#111827', borderRadius: '4px' }}>
                <div style={{ color: '#9ca3af', fontSize: '12px' }}>Classification Progress</div>
                <div style={{ color: '#10b981', fontSize: '24px', fontWeight: 'bold' }}>
                  {statistics.classification_progress_percent}%
                </div>
              </div>
            </div>
          )}

          <div style={{ padding: '15px', backgroundColor: '#fef3c7', borderRadius: '4px', marginBottom: '20px' }}>
            <div style={{ color: '#92400e', fontSize: '12px', fontWeight: '600', marginBottom: '5px' }}>
              Note: Evaluation Metrics
            </div>
            <div style={{ color: '#78350f', fontSize: '11px' }}>
              Accuracy, Precision, Recall metrics will be available after ground truth data is provided.
            </div>
          </div>

          <h3 style={{ margin: '20px 0 15px 0', color: '#f3f4f6', fontSize: '16px' }}>
            Export Options
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button
              onClick={() => handleExportTable(1, 'csv')}
              disabled={exporting}
              style={{
                padding: '10px',
                backgroundColor: '#059669',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              Export Table 1 (CSV)
            </button>

            <button
              onClick={() => handleExportTable(2, 'csv')}
              disabled={exporting}
              style={{
                padding: '10px',
                backgroundColor: '#059669',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              Export Table 2 (CSV)
            </button>

            <button
              onClick={() => handleExportTable(3, 'csv')}
              disabled={exporting}
              style={{
                padding: '10px',
                backgroundColor: '#059669',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '13px'
              }}
            >
              Export Table 3 (CSV)
            </button>

            <div style={{ borderTop: '1px solid #374151', margin: '10px 0' }} />

            <button
              onClick={handleExportAll}
              disabled={exporting}
              style={{
                padding: '12px',
                backgroundColor: '#7c3aed',
                color: '#f3f4f6',
                border: 'none',
                borderRadius: '4px',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: '600'
              }}
            >
              {exporting ? 'Exporting...' : 'Export All Tables (Excel)'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
