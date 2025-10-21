# Model Benchmark Results

Результаты автоматического тестирования биомедицинских моделей для классификации статей.

## Цель

Улучшить F1-Score (Weighted) с текущих **58.91%** до целевых **90%** путём тестирования специализированных биомедицинских моделей.

## Протестированные модели

1. **bioformers/bioformer-8L** (baseline) - текущая модель
2. **microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext** - рекомендуемая модель
3. **allenai/scibert_scivocab_uncased** - научные статьи
4. **dmis-lab/biobert-v1.1** - PubMed + PMC
5. **microsoft/BioGPT-Large** - генеративная модель (fallback)

## Результаты

Детальные результаты для каждой модели сохраняются в файлах:
- `{model_name}_detailed.json` - полные метрики и предсказания
- `benchmark_summary.json` - сводная таблица всех моделей
- `comparison_table.txt` - читаемое сравнение

## Метрики

Для каждой модели рассчитываются:
- F1-Score (Weighted) - **главная метрика**
- F1-Score (Macro, Micro)
- Accuracy (Weighted)
- Precision, Recall для каждого вопроса
- Confusion Matrix
- Время inference
- Потребление VRAM

## Запуск benchmark

```bash
cd data_d_questions_and_criterias_classifier
python model_benchmark.py
```

Время выполнения: ~20-40 минут для всех моделей.

## Следующие шаги

После выбора лучшей модели:
1. Обновить `config.yaml` с новой моделью
2. Запустить полную классификацию на всех статьях
3. При необходимости рассмотреть дополнительные улучшения:
   - Энсамбль нескольких моделей
   - NLI-подход
   - Section-aware chunking
