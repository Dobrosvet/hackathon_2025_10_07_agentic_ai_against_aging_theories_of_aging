# Database Viewer and Export Service

Микросервис для просмотра и экспорта данных из Qdrant в формате таблиц согласно требованиям хакатона Aging Theories Challenge.

## Функциональность

### 3 таблицы данных

1. **Таблица 1: Теории старения**
   - `theory_id` - уникальный идентификатор теории
   - `theory_name` - название теории
   - `number_of_collected_papers` - количество собранных статей

2. **Таблица 2: Собранные статьи**
   - `theory_id` - идентификатор теории
   - `paper_url` - URL статьи
   - `paper_name` - название статьи
   - `paper_year` - год публикации

3. **Таблица 3: Анализ статей с вопросами и критериями**
   - `theory_id`, `paper_url`, `paper_name`, `paper_year`
   - `Q1` - `Q9` - ответы на 9 исследовательских вопросов
   - `C1` - `C4` - оценки по 4 критериям

## API Endpoints

- `GET /api/tables/preview/{table_number}` - Предпросмотр таблицы (10 записей)
- `GET /api/tables/full/{table_number}` - Полная таблица с пагинацией
- `GET /api/statistics` - Статистика по данным
- `POST /api/export/{table_number}` - Экспорт таблицы в CSV
- `WebSocket /ws` - Real-time обновления

## Запуск

```bash
poetry install
poetry run python main.py
```

Сервис будет доступен на `http://127.0.0.1:8006`

## UI

Интегрирован в общий dashboard с 3-колоночным layout:
- Колонка 1: Предпросмотр таблиц
- Колонка 2: Полный просмотр выбранной таблицы
- Колонка 3: Статистика и кнопки экспорта

## Экспортированные файлы

Все файлы сохраняются в: `data/exports/`
