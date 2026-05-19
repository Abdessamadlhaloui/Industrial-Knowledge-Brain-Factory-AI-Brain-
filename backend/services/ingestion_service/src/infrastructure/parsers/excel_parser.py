import io
from typing import Any, Dict

import pandas as pd


class ExcelParser:
    """
    Parser for Excel documents.
    Utilizes pandas and openpyxl to handle multi-sheet workbooks.
    """

    @classmethod
    def parse(cls, file_bytes: bytes) -> Dict[str, Any]:
        """
        Parse an Excel document from raw bytes.
        
        Args:
            file_bytes (bytes): The raw Excel file bytes.
            
        Returns:
            Dict[str, Any]: A dictionary mapping sheet names to their parsed JSON representations.
        """
        # Read all sheets into a dictionary of DataFrames
        # openpyxl engine naturally handles merged cells (usually filling with NaNs for the non-top-left cells)
        excel_stream = io.BytesIO(file_bytes)
        
        try:
            # Read all sheets at once
            dfs = pd.read_excel(excel_stream, sheet_name=None, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")
            
        result = {}
        for sheet_name, df in dfs.items():
            # Clean up the DataFrame: drop empty rows/cols, fill NaNs
            df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
            df_cleaned = df_cleaned.fillna("")
            
            # Convert to a list of dicts (records)
            records = df_cleaned.to_dict(orient="records")
            result[sheet_name] = records
            
        return result
