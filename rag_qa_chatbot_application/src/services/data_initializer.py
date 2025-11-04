"""
Data initialization service for PubMed dataset
Downloads and processes the PubMed diabetes dataset into FAISS databases
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
from typing import Optional
import requests
from langchain.schema import Document as LangchainDocument

from ..config import config
from ..utils import app_logger


class DataInitializer:
    """Service for downloading and initializing PubMed dataset"""

    def __init__(self):
        self.logger = app_logger
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.csv_file = self.data_dir / "pubmed_diabetes_1000_meta.csv"
        self.parquet_file = self.data_dir / "pubmed_diabetes_1000_meta.parquet"
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_pubmed_data(self) -> bool:
        """
        Download PubMed diabetes dataset CSV file
        
        Returns:
            True if download successful, False otherwise
        """
        url = "https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/healthcare-nlp/data/pubmed_diabetes_1000_meta.csv"
        
        if self.csv_file.exists():
            self.logger.info(f"PubMed CSV already exists: {self.csv_file}")
            return True
            
        try:
            self.logger.info(f"Downloading PubMed dataset from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save to file
            with open(self.csv_file, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Successfully downloaded PubMed dataset: {self.csv_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download PubMed dataset: {str(e)}")
            return False

    def process_csv_to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Process CSV file into DataFrame (similar to Spark DataFrame)
        
        Returns:
            pandas DataFrame or None if processing failed
        """
        if not self.csv_file.exists():
            self.logger.error(f"CSV file not found: {self.csv_file}")
            return None
            
        try:
            # Read CSV using columns 1-5 (0-indexed: columns 0-4 in pandas)
            # The user's code uses range(1,6) which in pandas means columns 0-5
            self.logger.info(f"Reading CSV file: {self.csv_file}")
            df = pd.read_csv(self.csv_file, usecols=range(1, 6))
            
            # Sort by pubmed_id (similar to orderBy in Spark)
            if 'pubmed_id' in df.columns:
                df = df.sort_values('pubmed_id')
            
            self.logger.info(f"Successfully processed CSV. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to process CSV: {str(e)}")
            return None

    def save_as_parquet(self, df: pd.DataFrame) -> bool:
        """
        Save DataFrame as Parquet file (similar to Spark write.parquet)
        
        Args:
            df: pandas DataFrame to save
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            self.logger.info(f"Saving DataFrame as Parquet: {self.parquet_file}")
            df.to_parquet(self.parquet_file, index=False, engine='pyarrow')
            self.logger.info(f"Successfully saved Parquet file: {self.parquet_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save Parquet file: {str(e)}")
            return False

    def load_parquet(self) -> Optional[pd.DataFrame]:
        """
        Load Parquet file into DataFrame
        
        Returns:
            pandas DataFrame or None if loading failed
        """
        if not self.parquet_file.exists():
            self.logger.warning(f"Parquet file not found: {self.parquet_file}")
            return None
            
        try:
            self.logger.info(f"Loading Parquet file: {self.parquet_file}")
            df = pd.read_parquet(self.parquet_file, engine='pyarrow')
            self.logger.info(f"Successfully loaded Parquet. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load Parquet file: {str(e)}")
            return None

    def dataframe_to_documents(self, df: pd.DataFrame) -> list[LangchainDocument]:
        """
        Convert pandas DataFrame to LangChain documents
        Similar to PySparkDataFrameLoader functionality
        
        Args:
            df: pandas DataFrame with 'abstract' column
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        try:
            if 'abstract' not in df.columns:
                self.logger.error("DataFrame must have 'abstract' column")
                return documents
                
            for idx, row in df.iterrows():
                abstract = str(row.get('abstract', ''))
                
                # Skip empty abstracts
                if not abstract or abstract.strip() == '' or abstract.lower() == 'nan':
                    continue
                
                # Create metadata from other columns
                metadata = {}
                for col in df.columns:
                    if col != 'abstract':
                        value = row.get(col)
                        if pd.notna(value):
                            metadata[col] = str(value)
                
                # Add pubmed_id to source if available
                if 'pubmed_id' in metadata:
                    metadata['source'] = f"PubMed_{metadata['pubmed_id']}"
                
                doc = LangchainDocument(
                    page_content=abstract,
                    metadata=metadata
                )
                documents.append(doc)
                
            self.logger.info(f"Converted DataFrame to {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to convert DataFrame to documents: {str(e)}")
            return documents

    def initialize_pubmed_dataset(self) -> tuple:
        """
        Complete initialization pipeline:
        1. Download CSV if not exists
        2. Process CSV to DataFrame
        3. Save as Parquet
        4. Load Parquet
        5. Convert to LangChain documents
        
        Returns:
            Tuple of (DataFrame, Documents) or (None, []) if failed
        """
        # Step 1: Download CSV
        if not self.download_pubmed_data():
            return None, []
        
        # Step 2: Process CSV
        df = self.process_csv_to_dataframe()
        if df is None:
            return None, []
        
        # Step 3: Save as Parquet (if not exists)
        if not self.parquet_file.exists():
            if not self.save_as_parquet(df):
                return None, []
        else:
            self.logger.info(f"Parquet file already exists, skipping save")
        
        # Step 4: Load Parquet (to ensure consistency)
        df = self.load_parquet()
        if df is None:
            return None, []
        
        # Step 5: Convert to documents
        documents = self.dataframe_to_documents(df)
        
        return df, documents


__all__ = ["DataInitializer"]

