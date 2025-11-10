"""
Entity Extractor Module
Extracts and merges entities from NER pipeline results with priority
Priority: Posology > DeID > Clinical
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from typing import Dict, List
from collections import defaultdict


def extract_entities_from_ner_results(result_df, text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract entities from NER pipeline results and create entity dataframe.
    Priority: Posology > DeID > Clinical
    
    Args:
        result_df: Spark DataFrame with NER results
        Must contain columns: text_id, chunk_clinical, chunk_deid, chunk_posology
        text_df: Pandas DataFrame with text data (columns: text_id, text)
        
    Returns:
        Pandas DataFrame with merged entities (columns: text_id, begin, end, chunk, entity)
    """
    entities_list = []
    
    # Collect clinical entities
    clinical_results = result_df.select(
        "text_id",
        F.explode(F.arrays_zip(
            result_df["chunk_clinical"].result,
            result_df["chunk_clinical"].begin,
            result_df["chunk_clinical"].end,
            result_df["chunk_clinical"].metadata
        )).alias("clinical_chunk")
    ).select(
        "text_id",
        F.expr("clinical_chunk['0']").alias("chunk"),
        F.expr("clinical_chunk['1']").alias("begin"),
        F.expr("clinical_chunk['2']").alias("end"),
        F.expr("clinical_chunk['3']['entity']").alias("entity")
    ).collect()
    
    # Process clinical entities
    clinical_entities = {}
    for row in clinical_results:
        text_id = row.text_id
        if text_id not in clinical_entities:
            clinical_entities[text_id] = []
        clinical_entities[text_id].append({
            'begin': row.begin,
            'end': row.end,
            'chunk': row.chunk,
            'entity': row.entity,
            'source': 'clinical'
        })
    
    # Collect DeID entities
    deid_results = result_df.select(
        "text_id",
        F.explode(F.arrays_zip(
            result_df["chunk_deid"].result,
            result_df["chunk_deid"].begin,
            result_df["chunk_deid"].end,
            result_df["chunk_deid"].metadata
        )).alias("deid_chunk")
    ).select(
        "text_id",
        F.expr("deid_chunk['0']").alias("chunk"),
        F.expr("deid_chunk['1']").alias("begin"),
        F.expr("deid_chunk['2']").alias("end"),
        F.expr("deid_chunk['3']['entity']").alias("entity")
    ).collect()
    
    # Process DeID entities
    deid_entities = {}
    for row in deid_results:
        text_id = row.text_id
        if text_id not in deid_entities:
            deid_entities[text_id] = []
        deid_entities[text_id].append({
            'begin': row.begin,
            'end': row.end,
            'chunk': row.chunk,
            'entity': row.entity,
            'source': 'deid'
        })
    
    # Collect Posology entities (only Drug and Dosage)
    posology_results = result_df.select(
        "text_id",
        F.explode(F.arrays_zip(
            result_df["chunk_posology"].result,
            result_df["chunk_posology"].begin,
            result_df["chunk_posology"].end,
            result_df["chunk_posology"].metadata
        )).alias("posology_chunk")
    ).select(
        "text_id",
        F.expr("posology_chunk['0']").alias("chunk"),
        F.expr("posology_chunk['1']").alias("begin"),
        F.expr("posology_chunk['2']").alias("end"),
        F.expr("posology_chunk['3']['entity']").alias("entity")
    ).filter(
        F.col("entity").isin(["Drug", "Dosage"])
    ).collect()
    
    # Process Posology entities
    posology_entities = {}
    for row in posology_results:
        text_id = row.text_id
        if text_id not in posology_entities:
            posology_entities[text_id] = []
        posology_entities[text_id].append({
            'begin': row.begin,
            'end': row.end,
            'chunk': row.chunk,
            'entity': row.entity,
            'source': 'posology'
        })
    
    # Merge entities with priority: Posology > DeID > Clinical
    all_entities = []
    for text_id in text_df['text_id'].unique():
        merged = {}  # Key: (begin, end), Value: entity dict
        
        # Add Posology entities first (highest priority)
        if text_id in posology_entities:
            for ent in posology_entities[text_id]:
                key = (ent['begin'], ent['end'])
                merged[key] = ent
        
        # Add DeID entities (medium priority)
        if text_id in deid_entities:
            for ent in deid_entities[text_id]:
                key = (ent['begin'], ent['end'])
                if key not in merged:  # Don't override posology
                    merged[key] = ent
        
        # Add clinical entities (lowest priority)
        if text_id in clinical_entities:
            for ent in clinical_entities[text_id]:
                key = (ent['begin'], ent['end'])
                if key not in merged:  # Don't override posology or deid
                    merged[key] = ent
        
        # Convert to list
        for ent in merged.values():
            all_entities.append({
                'text_id': text_id,
                'begin': ent['begin'],
                'end': ent['end'],
                'chunk': ent['chunk'],
                'entity': ent['entity']
            })
    
    entity_df = pd.DataFrame(all_entities)
    return entity_df


def filter_posology_entities(entity_df: pd.DataFrame, keep_entities: List[str] = ["Drug", "Dosage"]) -> pd.DataFrame:
    """
    Filter posology entities to keep only specified entity types
    
    Args:
        entity_df: DataFrame with entities
        keep_entities: List of entity types to keep
        
    Returns:
        Filtered DataFrame
    """
    return entity_df[entity_df['entity'].isin(keep_entities)]


def get_entity_statistics(entity_df: pd.DataFrame) -> Dict:
    """
    Get statistics about extracted entities
    
    Args:
        entity_df: DataFrame with entities
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_entities': len(entity_df),
        'unique_entity_types': sorted(entity_df['entity'].unique().tolist()),
        'entity_counts': entity_df['entity'].value_counts().to_dict(),
        'texts_with_entities': entity_df['text_id'].nunique(),
        'avg_entities_per_text': len(entity_df) / entity_df['text_id'].nunique() if len(entity_df) > 0 else 0
    }
    return stats

