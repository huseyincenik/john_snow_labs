"""
NER Pipeline Module
Creates and executes Spark NLP Healthcare NER pipeline with multiple models
"""

import sparknlp
import sparknlp_jsl
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer
from sparknlp_jsl.annotator import MedicalNerModel, NerConverter
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings("ignore")


class NERPipeline:
    """NER Pipeline with multiple pre-trained models"""

    def __init__(self, spark: SparkSession, license_secret: Optional[str] = None):
        """
        Initialize NER Pipeline

        Args:
            spark: SparkSession instance
            license_secret: Spark NLP Healthcare license secret (if not already configured)
        """
        self.spark = spark
        self.license_secret = license_secret
        self.pipeline = None
        self.models = {}

    def create_pipeline(self, prioritize_posology_deid: bool = True):
        """
        Create NER pipeline with multiple models

        Args:
            prioritize_posology_deid: If True, posology and deid models take priority
        """
        # Document Assembler
        document_assembler = (
            DocumentAssembler()
            .setInputCol("text")
            .setOutputCol("document")
            .setCleanupMode("shrink")
        )

        # Sentence Detector
        sentence_detector = (
            SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentence")
            .setExplodeSentences(True)
        )

        # Tokenizer
        tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

        # NER Models
        # 1. Clinical NER Model
        ner_clinical = (
            MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
            .setInputCols(["sentence", "token"])
            .setOutputCol("ner_clinical")
        )

        # 2. DeID Generic Augmented Model
        ner_deid = (
            MedicalNerModel.pretrained(
                "ner_deid_generic_augmented", "en", "clinical/models"
            )
            .setInputCols(["sentence", "token"])
            .setOutputCol("ner_deid")
        )

        # 3. Posology Model (for Drug and Dosage)
        ner_posology = (
            MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")
            .setInputCols(["sentence", "token"])
            .setOutputCol("ner_posology")
        )

        # Store models
        self.models = {
            "clinical": ner_clinical,
            "deid": ner_deid,
            "posology": ner_posology,
        }

        # Create pipeline stages
        stages = [
            document_assembler,
            sentence_detector,
            tokenizer,
            ner_clinical,
            ner_deid,
            ner_posology,
        ]

        # If prioritizing, we need to merge results
        # For now, we'll run all models and merge in post-processing
        if prioritize_posology_deid:
            # Add NerConverter for each model
            ner_converter_clinical = (
                NerConverter()
                .setInputCols(["document", "token", "ner_clinical"])
                .setOutputCol("chunk_clinical")
            )

            ner_converter_deid = (
                NerConverter()
                .setInputCols(["document", "token", "ner_deid"])
                .setOutputCol("chunk_deid")
            )

            ner_converter_posology = (
                NerConverter()
                .setInputCols(["document", "token", "ner_posology"])
                .setOutputCol("chunk_posology")
            )

            stages.extend(
                [ner_converter_clinical, ner_converter_deid, ner_converter_posology]
            )

        self.pipeline = Pipeline(stages=stages)
        return self.pipeline

    def fit_transform(self, data):
        """
        Fit and transform data through pipeline

        Args:
            data: Spark DataFrame with 'text' column

        Returns:
            Transformed DataFrame with NER results
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first.")

        model = self.pipeline.fit(data)
        result = model.transform(data)
        return result

    def filter_posology_entities(
        self, ner_result, keep_entities: List[str] = ["Drug", "Dosage"]
    ):
        """
        Filter posology entities to keep only Drug and Dosage

        Args:
            ner_result: NER result from posology model
            keep_entities: List of entity types to keep

        Returns:
            Filtered NER results
        """
        # This would be implemented based on the actual structure of NER results
        # For now, this is a placeholder
        return ner_result

    def merge_ner_results(self, result_df, prioritize_posology_deid: bool = True):
        """
        Merge results from multiple NER models with priority

        Priority order (if prioritize_posology_deid=True):
        1. Posology (Drug, Dosage)
        2. DeID (PHI entities)
        3. Clinical (other clinical entities)

        Args:
            result_df: DataFrame with NER results from all models
            prioritize_posology_deid: Whether to prioritize posology and deid

        Returns:
            DataFrame with merged NER results
        """
        # This is a complex operation that requires:
        # 1. Extracting entities from each model
        # 2. Resolving conflicts based on priority
        # 3. Creating a unified entity list

        # For now, return the original dataframe
        # Full implementation would merge chunks with priority logic
        return result_df

    def extract_entities(self, result_df) -> List[Dict]:
        """
        Extract entities from pipeline results

        Args:
            result_df: DataFrame with NER results

        Returns:
            List of entity dictionaries with text_id, begin, end, chunk, entity
        """
        entities = []

        # Extract entities from each model
        # This is a simplified version - actual implementation would need
        # to handle the Spark DataFrame structure properly

        return entities
