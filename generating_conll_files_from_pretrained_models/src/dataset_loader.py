"""
Dataset Loader Module
Handles loading of healthcare datasets for NER tasks
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple
import os


class DatasetLoader:
    """Load and prepare healthcare datasets for NER tasks"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DatasetLoader

        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_mtsamples_classifier(self) -> pd.DataFrame:
        """
        Download mtsamples_classifier dataset from Spark NLP workshop

        Returns:
            DataFrame with text data
        """
        url = "https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/mtsamples_classifier.csv"
        file_path = self.data_dir / "mtsamples_classifier.csv"

        if not file_path.exists():
            print(f"Downloading mtsamples_classifier dataset from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Dataset saved to {file_path}")
        else:
            print(f"Dataset already exists at {file_path}")

        df = pd.read_csv(file_path)
        return df

    def download_oncology_notes(self) -> pd.DataFrame:
        """
        Download oncology notes dataset from Spark NLP workshop

        Returns:
            DataFrame with text data
        """
        # Oncology notes are typically in a directory, we'll need to handle multiple files
        base_url = "https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/data/oncology_notes"

        # This is a placeholder - actual implementation would need to list and download files
        # For now, we'll return a message
        print(
            "Oncology notes dataset requires manual download or specific file listing"
        )
        print(f"Base URL: {base_url}")
        return pd.DataFrame()

    def load_local_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from local file

        Args:
            file_path: Path to local dataset file

        Returns:
            DataFrame with data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def prepare_text_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare dataframe for NER pipeline

        Args:
            df: Input dataframe
            text_column: Name of column containing text
            id_column: Name of column containing document IDs (optional)

        Returns:
            Prepared dataframe with 'text_id' and 'text' columns
        """
        result_df = df.copy()

        # Create text_id if not provided
        if id_column is None:
            result_df["text_id"] = range(len(result_df))
        else:
            result_df["text_id"] = result_df[id_column]

        # Ensure text column exists
        if text_column not in result_df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataframe")

        # Select and rename columns
        result_df = result_df[["text_id", text_column]].copy()
        result_df.columns = ["text_id", "text"]

        return result_df

    def get_sample_texts(self, df: pd.DataFrame, n_samples: int = 10) -> pd.DataFrame:
        """
        Get sample texts from dataset

        Args:
            df: Input dataframe
            n_samples: Number of samples to return

        Returns:
            DataFrame with sample texts
        """
        return df.head(n_samples)
