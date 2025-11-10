"""
Model Trainer Module
Trains custom NER models using CoNLL format data
Based on: 1.4.Resume_MedicalNer_Model_Training.ipynb
"""

import sparknlp
import sparknlp_jsl
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import WordEmbeddingsModel
from sparknlp_jsl.annotator import (
    MedicalNerApproach,
    MedicalNerDLGraphChecker,
    MedicalNerModel
)
from sparknlp.training import CoNLL
from sparknlp_jsl.eval import NerDLMetrics
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train custom NER models from CoNLL data"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Model Trainer
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        self.clinical_embeddings = None
        self.trained_model = None
    
    def load_embeddings(self, embeddings_model: str = "embeddings_clinical"):
        """
        Load clinical word embeddings
        
        Args:
            embeddings_model: Name of embeddings model to use
        """
        print(f"Loading {embeddings_model} embeddings...")
        self.clinical_embeddings = WordEmbeddingsModel.pretrained(
            embeddings_model, "en", "clinical/models"
        ).setInputCols(["sentence", "token"]).setOutputCol("embeddings")
        print("Embeddings loaded successfully!")
        return self.clinical_embeddings
    
    def load_conll_dataset(self, conll_path: str):
        """
        Load CoNLL format dataset
        
        Args:
            conll_path: Path to CoNLL file
            
        Returns:
            Spark DataFrame with loaded dataset
        """
        print(f"Loading CoNLL dataset from {conll_path}...")
        data = CoNLL().readDataset(self.spark, conll_path)
        print(f"Dataset loaded: {data.count()} sentences")
        return data
    
    def split_dataset(self, data, train_ratio: float = 0.8, seed: int = 100):
        """
        Split dataset into train and validation sets
        
        Args:
            data: Spark DataFrame
            train_ratio: Ratio for training set
            seed: Random seed
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        train_data, validation_data = data.randomSplit(
            [train_ratio, 1 - train_ratio], seed=seed
        )
        print(f"Train set: {train_data.count()} sentences")
        print(f"Validation set: {validation_data.count()} sentences")
        return train_data, validation_data
    
    def create_training_pipeline(self,
                                 max_epochs: int = 10,
                                 lr: float = 0.003,
                                 batch_size: int = 8,
                                 random_seed: int = 0,
                                 verbose: int = 1,
                                 test_dataset: Optional[str] = None,
                                 output_logs_path: str = "./ner_logs",
                                 validation_split: float = 0.2,
                                 use_best_model: bool = True,
                                 early_stopping_criterion: float = 0.04,
                                 early_stopping_patience: int = 3,
                                 pretrained_model_path: Optional[str] = None):
        """
        Create training pipeline
        
        Args:
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            batch_size: Batch size
            random_seed: Random seed
            verbose: Verbosity level
            test_dataset: Path to test dataset (parquet)
            output_logs_path: Path to save training logs
            validation_split: Validation split ratio
            use_best_model: Whether to use best model during training
            early_stopping_criterion: Early stopping criterion
            early_stopping_patience: Early stopping patience
            pretrained_model_path: Path to pretrained model for fine-tuning
            
        Returns:
            Pipeline for training
        """
        if self.clinical_embeddings is None:
            self.load_embeddings()
        
        # Graph checker
        ner_dl_graph_checker = MedicalNerDLGraphChecker()\
            .setInputCols(["sentence", "token"])\
            .setLabelColumn("label")\
            .setEmbeddingsModel(self.clinical_embeddings)
        
        # NER Tagger
        ner_tagger = MedicalNerApproach()\
            .setInputCols(["sentence", "token", "embeddings"])\
            .setLabelColumn("label")\
            .setOutputCol("ner")\
            .setMaxEpochs(max_epochs)\
            .setLr(lr)\
            .setBatchSize(batch_size)\
            .setRandomSeed(random_seed)\
            .setVerbose(verbose)\
            .setEvaluationLogExtended(True)\
            .setEnableOutputLogs(True)\
            .setIncludeConfidence(True)\
            .setValidationSplit(validation_split)\
            .setUseBestModel(use_best_model)\
            .setEarlyStoppingCriterion(early_stopping_criterion)\
            .setEarlyStoppingPatience(early_stopping_patience)\
            .setOutputLogsPath(output_logs_path)
        
        if test_dataset:
            ner_tagger.setTestDataset(test_dataset)
        
        if pretrained_model_path:
            ner_tagger.setPretrainedModelPath(pretrained_model_path)
            ner_tagger.setOverrideExistingTags(True)
        
        pipeline = Pipeline(stages=[
            self.clinical_embeddings,
            ner_dl_graph_checker,
            ner_tagger
        ])
        
        return pipeline
    
    def train_model(self, training_data, pipeline: Pipeline):
        """
        Train the NER model
        
        Args:
            training_data: Training dataset
            pipeline: Training pipeline
            
        Returns:
            Trained model
        """
        print("Starting model training...")
        self.trained_model = pipeline.fit(training_data)
        print("Training completed!")
        return self.trained_model
    
    def evaluate_model(self, test_data, drop_o: bool = True, case_sensitive: bool = True):
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataset
            drop_o: Whether to drop O labels from evaluation
            case_sensitive: Whether evaluation is case sensitive
            
        Returns:
            Evaluation results DataFrame
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.clinical_embeddings is None:
            self.load_embeddings()
        
        print("Evaluating model...")
        pred_df = self.trained_model.stages[2].transform(
            self.clinical_embeddings.transform(test_data)
        )
        
        evaler = NerDLMetrics(mode="full_chunk")
        eval_result = evaler.computeMetricsFromDF(
            pred_df.select("label", "ner"),
            prediction_col="ner",
            label_col="label",
            drop_o=drop_o,
            case_sensitive=case_sensitive
        ).cache()
        
        # Format results
        eval_result_formatted = eval_result.withColumn(
            "precision", F.round(eval_result["precision"], 4)
        ).withColumn(
            "recall", F.round(eval_result["recall"], 4)
        ).withColumn(
            "f1", F.round(eval_result["f1"], 4)
        )
        
        print("\nEvaluation Results:")
        eval_result_formatted.show(100)
        
        # Calculate macro and micro averages
        macro_avg = eval_result.selectExpr("avg(f1) as macro").show()
        micro_avg = eval_result.selectExpr(
            "sum(f1*total) as sumprod", "sum(total) as sumtotal"
        ).selectExpr("sumprod/sumtotal as micro").show()
        
        return eval_result_formatted
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: Path to save model
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {model_path}...")
        self.trained_model.stages[2].write().overwrite().save(str(model_path))
        print("Model saved successfully!")
    
    def load_pretrained_model(self, model_path: str):
        """
        Load a pretrained model
        
        Args:
            model_path: Path to model
        """
        print(f"Loading model from {model_path}...")
        model = MedicalNerModel.load(model_path)
        print("Model loaded successfully!")
        return model

