"""
CoNLL Converter Module
Converts NER predictions to CoNLL format for model training
Based on: 1.3.prepare_CoNLL_from_annotations_for_NER.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer
from pyspark.ml import Pipeline


class CoNLLConverter:
    """Convert NER predictions to CoNLL format"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize CoNLL Converter
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
    
    def make_conll(self, 
                   text_df: pd.DataFrame, 
                   entity_df: pd.DataFrame,
                   save_tag: bool = True,
                   save_conll: bool = True,
                   output_path: str = "data/conll/conll2003_text_file.conll",
                   verbose: bool = None,
                   begin_deviation: int = 0,
                   end_deviation: int = 0) -> str:
        """
        Create CoNLL file from text and entity dataframes
        
        Args:
            text_df: DataFrame with columns ['text_id', 'text']
            entity_df: DataFrame with columns ['text_id', 'begin', 'end', 'chunk', 'entity']
            save_tag: Whether to save tagged text CSV
            save_conll: Whether to save CoNLL file
            output_path: Path to save CoNLL file
            verbose: Whether to print verbose output
            begin_deviation: Deviation to add to begin positions
            end_deviation: Deviation to add to end positions
            
        Returns:
            CoNLL formatted string
        """
        # Prepare dataframes
        df_text = text_df.iloc[:, [0, 1]].copy()
        df_entity = entity_df.iloc[:, [0, 1, 2, 3, 4]].copy()
        df_text.columns = ['text_id', 'text']
        df_entity.columns = ['text_id', 'begin', 'end', 'chunk', 'entity']
        entity_list = list(df_entity.entity.unique())
        
        # Step 1: Tag transformation
        print("Text tagging starting. Applying entities to whole text...\n")
        df = self._apply_tag_ner(df_text, df_entity, save=save_tag, 
                                verbose=verbose, begin_deviation=begin_deviation,
                                end_deviation=end_deviation)
        
        # Step 2: Spark Pipeline for tokenization
        print("\n\nSpark pipeline is running...")
        df_final = self._spark_pipeline(df)
        
        # Step 3: Build CoNLL
        print("Conll file is being created...\n")
        conll_text = self._build_conll(df_final, entity_list, save=save_conll, 
                                       output_path=output_path)
        
        return conll_text
    
    def _transform_text(self, text: str, entities: pd.DataFrame, 
                       verbose: Optional[bool] = None,
                       begin_deviation: int = 0,
                       end_deviation: int = 0) -> str:
        """Transform text by adding entity tags"""
        tag_list = []
        
        # Sort entities by end position (descending) to insert from end to beginning
        entities_sorted = entities.sort_values(by='end', ascending=False)
        
        for _, entity in entities_sorted.iterrows():
            begin = int(entity['begin']) + begin_deviation
            end = int(entity['end']) + end_deviation
            chunk = entity['chunk']
            tag = entity['entity']
            
            # Insert end tag
            text = text[:end] + f' </END_NER:{tag}> ' + text[end:]
            # Insert start tag
            text = text[:begin] + f' <START_NER:{tag}> ' + text[begin:]
            tag_list.append(tag)
        
        sum_of_added_entity = Counter(tag_list)
        sum_of_entity = Counter(entities['entity'].values)
        
        if verbose:
            print(f'Processed text id   : {entities.text_id.values[:1]}')
            print(f'Original Entities   : {sum_of_entity}\nAdded Entities      : {sum_of_added_entity}')
            print(f'Number Equality     : {sum_of_added_entity == sum_of_entity}')
            print("==" * 40)
        
        if not sum_of_entity == sum_of_added_entity:
            print("There is a problem in text id:")
            print(entities.text_id.values[0])
            raise Exception("Check this text!")
        
        return text
    
    def _apply_tag_ner(self, df_text: pd.DataFrame, df_entity: pd.DataFrame,
                      save: bool = False, verbose: Optional[bool] = None,
                      begin_deviation: int = 0, end_deviation: int = 0) -> pd.DataFrame:
        """Apply NER tags to text dataframe"""
        for text_id in tqdm(df_text.text_id):
            text = df_text.loc[df_text['text_id'] == text_id]['text'].values[0]
            entities = df_entity.loc[(df_entity['text_id'] == text_id)].sort_values(
                by='begin', ascending=False
            )
            
            df_text.loc[df_text['text_id'] == text_id, 'text'] = self._transform_text(
                text, entities, verbose=verbose, 
                begin_deviation=begin_deviation, 
                end_deviation=end_deviation
            )
        
        if save:
            output_path = Path("data/processed/text_with_ner_tag.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_text.to_csv(output_path, index=False, encoding='utf8')
        
        return df_text
    
    def _spark_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Spark NLP pipeline for tokenization"""
        spark_df = self.spark.createDataFrame(df)
        
        document_assembler = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")\
            .setCleanupMode("shrink")
        
        sentence_detector = SentenceDetector()\
            .setInputCols(['document'])\
            .setOutputCol('sentences')\
            .setExplodeSentences(True)
        
        tokenizer = Tokenizer()\
            .setInputCols(["sentences"])\
            .setOutputCol("token")
        
        nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer])
        
        empty_df = self.spark.createDataFrame([['']]).toDF("text")
        pipeline_model = nlp_pipeline.fit(empty_df)
        
        result = pipeline_model.transform(spark_df.select(['text']))
        
        return result.select('token.result').toPandas()
    
    def _build_conll(self, df_final: pd.DataFrame, tag_list: List[str],
                    save: bool = False, output_path: str = "data/conll/conll2003_text_file.conll") -> str:
        """Build CoNLL formatted string"""
        header = "-DOCSTART- -X- -X- O\n\n"
        conll_text = ""
        chunks = []
        tag = 'O'  # token tag
        ct = 'B'   # chunk tag part B or I
        
        for sentence_tokens in tqdm(df_final.result[:]):
            for token in sentence_tokens:
                if token.startswith("<START_NER:"):
                    tag = token.split(':')[1][:-1]
                    if tag not in tag_list:
                        tag = 'O'
                        conll_text += f'{token} NN NN {tag}\n'
                    continue
                
                if token.startswith("</END_NER:") and tag != 'O':
                    for i, chunk in enumerate(chunks):
                        ct = 'B' if i == 0 else 'I'
                        conll_text += f'{chunk} NNP NNP {ct}-{tag}\n'
                    chunks = []
                    tag = 'O'
                    continue
                
                if tag != 'O':
                    chunks.append(token)
                    continue
                
                if tag == 'O':
                    conll_text += f'{token} NN NN {tag}\n'
                    continue
            
            conll_text += '\n'
        
        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w+", encoding='utf8') as f:
                f.write(header)
                f.write(conll_text)
        
        print("\nDONE!")
        return conll_text
    
    def convert_from_ner_results(self, result_df, text_id_col: str = "text_id"):
        """
        Alternative method: Convert directly from NER results DataFrame
        This method matches tokens with entities based on position
        
        Args:
            result_df: Spark DataFrame with NER results
            text_id_col: Name of text ID column
            
        Returns:
            CoNLL formatted string
        """
        # This is an alternative implementation based on the notebook
        # It tokenizes first, then matches tokens with entities
        pass

