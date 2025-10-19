"""
Export Service для Database Viewer
Экспорт таблиц в CSV и Excel форматы
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting tables to CSV and Excel formats"""

    def __init__(self, output_dir: str = "exports"):
        """
        Инициализация Export Service

        Args:
            output_dir: Директория для сохранения экспортированных файлов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Export directory: {self.output_dir.absolute()}")

    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        table_name: str = "table"
    ) -> str:
        """
        Экспортировать данные в CSV

        Args:
            data: Список словарей с данными
            filename: Имя файла (без расширения)
            table_name: Название таблицы для логирования

        Returns:
            Путь к созданному файлу
        """
        try:
            if not data:
                logger.warning(f"No data to export for {table_name}")
                return None

            # Создать DataFrame
            df = pd.DataFrame(data)

            # Генерировать имя файла с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{filename}_{timestamp}.csv"
            csv_path = self.output_dir / csv_filename

            # Сохранить в CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            logger.info(f"Exported {table_name} to CSV: {csv_path} ({len(data)} rows)")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Error exporting {table_name} to CSV: {e}")
            raise

    def export_to_excel(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        table_name: str = "table",
        sheet_name: str = "Sheet1"
    ) -> str:
        """
        Экспортировать данные в Excel

        Args:
            data: Список словарей с данными
            filename: Имя файла (без расширения)
            table_name: Название таблицы для логирования
            sheet_name: Имя листа в Excel

        Returns:
            Путь к созданному файлу
        """
        try:
            if not data:
                logger.warning(f"No data to export for {table_name}")
                return None

            # Создать DataFrame
            df = pd.DataFrame(data)

            # Генерировать имя файла с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"{filename}_{timestamp}.xlsx"
            excel_path = self.output_dir / excel_filename

            # Сохранить в Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Автоматически настроить ширину колонок
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    # Ограничить максимальную ширину
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width

            logger.info(f"Exported {table_name} to Excel: {excel_path} ({len(data)} rows)")
            return str(excel_path)

        except Exception as e:
            logger.error(f"Error exporting {table_name} to Excel: {e}")
            raise

    def export_all_tables_to_excel(
        self,
        table1_data: List[Dict[str, Any]],
        table2_data: List[Dict[str, Any]],
        table3_data: List[Dict[str, Any]],
        filename: str = "aging_theories_all_tables"
    ) -> str:
        """
        Экспортировать все 3 таблицы в один Excel файл с разными листами

        Args:
            table1_data: Данные таблицы 1 (теории)
            table2_data: Данные таблицы 2 (статьи)
            table3_data: Данные таблицы 3 (анализ)
            filename: Имя файла (без расширения)

        Returns:
            Путь к созданному файлу
        """
        try:
            # Генерировать имя файла с timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"{filename}_{timestamp}.xlsx"
            excel_path = self.output_dir / excel_filename

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Таблица 1: Теории
                if table1_data:
                    df1 = pd.DataFrame(table1_data)
                    df1.to_excel(writer, sheet_name="Theories", index=False)
                    self._adjust_column_widths(writer, "Theories", df1)

                # Таблица 2: Статьи
                if table2_data:
                    df2 = pd.DataFrame(table2_data)
                    df2.to_excel(writer, sheet_name="Papers", index=False)
                    self._adjust_column_widths(writer, "Papers", df2)

                # Таблица 3: Анализ
                if table3_data:
                    df3 = pd.DataFrame(table3_data)
                    df3.to_excel(writer, sheet_name="Analysis", index=False)
                    self._adjust_column_widths(writer, "Analysis", df3)

            logger.info(f"Exported all tables to Excel: {excel_path}")
            return str(excel_path)

        except Exception as e:
            logger.error(f"Error exporting all tables to Excel: {e}")
            raise

    def _adjust_column_widths(self, writer, sheet_name: str, df: pd.DataFrame):
        """
        Автоматически настроить ширину колонок в Excel

        Args:
            writer: ExcelWriter объект
            sheet_name: Имя листа
            df: DataFrame
        """
        worksheet = writer.sheets[sheet_name]

        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            )
            # Ограничить максимальную ширину
            adjusted_width = min(max_length + 2, 50)

            # Excel колонки: A, B, C, ... Z, AA, AB, ...
            col_letter = self._get_column_letter(idx)
            worksheet.column_dimensions[col_letter].width = adjusted_width

    def _get_column_letter(self, idx: int) -> str:
        """
        Получить букву колонки Excel по индексу

        Args:
            idx: Индекс колонки (0-based)

        Returns:
            Буква колонки (A, B, ..., Z, AA, AB, ...)
        """
        letter = ''
        idx += 1  # Excel columns are 1-based
        while idx > 0:
            idx, remainder = divmod(idx - 1, 26)
            letter = chr(65 + remainder) + letter
        return letter
