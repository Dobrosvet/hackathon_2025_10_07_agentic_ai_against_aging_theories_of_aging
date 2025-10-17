# PubMed Central Data Fetcher Microservice

Микросервис для сбора списка URL научных статей из PubMed Central с использованием E-utilities API.

## Функциональность

- Поиск статей в PubMed Central по запросу: `(aging) AND (theory OR paradigm) OR (Aging[MeSh])`
- Получение URL статей и базовых метаданных (через ESummary API)
- Сохранение списка URL в векторную базу данных Qdrant
- Логирование всех операций в `./data/logs`
- Real-time мониторинг через WebSocket
- REST API для управления процессом

## Технологии

- **FastAPI** - веб-фреймворк
- **Uvicorn** - ASGI сервер
- **Biopython** - работа с биологическими данными
- **Qdrant** - векторная база данных
- **WebSockets** - real-time коммуникация
- **Poetry** - управление зависимостями

## Установка

1. Убедитесь, что установлен Python 3.10+ и Poetry
2. Установите зависимости:

```bash
cd data_a_get_urls_list_papers
poetry install
```

## Запуск

### Через run.ps1 (рекомендуется)

Из корневой директории проекта:

```powershell
.\run.ps1
```

### Вручную

```bash
cd data_a_get_urls_list_papers
poetry run python main.py
```

Сервис будет доступен на `http://127.0.0.1:8001`

## API Endpoints

### REST API

- `GET /` - Информация о сервисе
- `GET /api/status` - Текущий статус работы
- `POST /api/start` - Запустить процесс скачивания
- `POST /api/stop` - Остановить процесс

### WebSocket

- `WS /ws` - Real-time обновления логов и статистики

## Структура данных

### State (статус сервиса)

```json
{
  "status": "running",
  "progress": 150,
  "total": 1000,
  "current_paper": "PMC123456",
  "errors": 5,
  "saved": 145,
  "start_time": "2025-10-17T10:30:00",
  "search_query": "(aging) AND (theory OR paradigm) OR (Aging[MeSh])"
}
```

### Log Entry

```json
{
  "timestamp": "2025-10-17T10:30:15",
  "level": "INFO",
  "message": "Fetching paper PMC123456 (150/1000)"
}
```

### Paper Data (сохраняется в Qdrant)

```json
{
  "pmc_id": "123456",
  "title": "Article Title",
  "authors": ["John Doe", "Jane Smith"],
  "source": "Journal Name",
  "pubdate": "2024-01-15",
  "doi": "10.1234/example",
  "pmid": "98765432",
  "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/",
  "pdf_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/pdf/"
}
```

## Структура директорий

```
data_a_get_urls_list_papers/
├── main.py                 # Основное приложение FastAPI
├── pubmed_fetcher.py       # Модуль работы с PubMed API
├── qdrant_storage.py       # Модуль работы с Qdrant
├── pyproject.toml          # Poetry конфигурация
├── poetry.lock             # Зафиксированные версии зависимостей
└── README.md               # Документация

./data/
├── db/                     # Qdrant база данных
└── logs/                   # Логи сервиса
```

## WebSocket Protocol

Клиент получает сообщения трех типов:

1. **log** - Новая запись в логе
   ```json
   {
     "type": "log",
     "data": {
       "timestamp": "2025-10-17T10:30:15",
       "level": "INFO",
       "message": "..."
     }
   }
   ```

2. **state** - Обновление статуса
   ```json
   {
     "type": "state",
     "data": { /* статус сервиса */ }
   }
   ```

3. **logs** - Пакет логов (при подключении)
   ```json
   {
     "type": "logs",
     "data": [ /* массив логов */ ]
   }
   ```

## Мониторинг

Используйте Dashboard в браузере для мониторинга:
- Левая колонка: Real-time логи
- Правая колонка: Статистика и управление

## Примечания

- E-utilities API имеет ограничения по частоте запросов (не более 3 в секунду без API ключа)
- Для больших датасетов рекомендуется использовать API ключ NCBI
- Векторные эмбеддинги создаются детерминированно на основе хеша текста (для production используйте настоящую модель эмбеддингов)
- Сервис собирает только URL и метаданные, полные тексты статей не скачиваются (легкий и быстрый режим)
- Для получения полных текстов используйте собранные URL в других сервисах
