<div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkzoKPaZIwIrnqHBYP_if-0vLt-hT6h2h-BQ&s" alt="John Snow Labs logo" width="96">
  <h1>Spark NLP Healthcare CoNLL Generation & Custom NER Training</h1>
  <p>From pretrained clinical NER pipelines to labeled CoNLL datasets and fine-tuned models.</p>
  <img alt="Project views" src="https://komarev.com/ghpvc/?username=huseyincenik&color=orange&label=CoNLL+Generator+Views">
  <p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg" alt="Apache Spark" height="40">
    &nbsp;
    <img src="https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp/master/docs/assets/images/logo.png" alt="Spark NLP" height="40">
    &nbsp;
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" height="40">
    &nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1035px-Jupyter_logo.svg.png" alt="Jupyter" height="40">
  </p>
</div>

## 🎯 Project Overview

This project demonstrates an **end-to-end workflow for Healthcare Named Entity Recognition (NER)**:

1. **Data Preparation**: Load and preprocess clinical texts from the MTSamples dataset
2. **Multi-Model NER Pipeline**: Run 3 pretrained NER models in parallel with priority-based merging
3. **CoNLL Generation**: Convert NER annotations to industry-standard CoNLL 2003 format
4. **Custom Model Training**: Fine-tune a domain-specific NER model using the generated data
5. **Inference & Visualization**: Deploy the trained model for predictions with interactive visualization

### Why This Project?

Healthcare NLP requires extracting diverse entity types that no single model covers completely. This project solves this by:

- **Combining multiple specialized models** (Posology, Clinical, De-identification) into a unified pipeline
- **Resolving conflicts** when models disagree on entity boundaries using priority-based merging
- **Generating high-quality training data** that can be used to train custom models for specific use cases
- **Providing a reproducible workflow** from raw clinical text to deployable NER model

---

## 📊 Dataset: MTSamples

The project uses the **MTSamples (Medical Transcription Samples)** dataset, a widely-used resource in Healthcare NLP:

| Property | Value |
|----------|-------|
| **Source** | [mtsamples.com](https://mtsamples.com/) |
| **Total Documents** | 638 clinical texts |
| **Document Types** | Medical transcriptions, surgery notes, clinical reports, discharge summaries |
| **Medical Specialties** | 40+ specialties (Cardiology, Orthopedics, Radiology, Neurology, etc.) |
| **Language** | English |
| **Avg. Document Length** | ~500-2000 words |

**Sample Document Categories:**
- Consultation notes and physical examinations
- Operative reports and surgical procedures
- Diagnostic imaging reports (MRI, CT, X-ray)
- Discharge summaries and progress notes
- Emergency room encounters

---

## 🔗 Unified Entity Schema (3 Models Combined)

This project merges entities from **3 pretrained Spark NLP Healthcare models** into a single unified annotation set:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED ENTITY SCHEMA (11 Entity Types)                      │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  FROM ner_posology (Priority 1 - Medication Entities):                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  DRUG     │ Medication names (Aspirin, Lisinopril, Metformin)               │  │
│  │  DOSAGE   │ Medication amounts (100mg, 2 tablets, 10 units)                 │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  FROM ner_clinical_large (Priority 2 - General Clinical Entities):                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  PROBLEM   │ Diseases, symptoms, conditions (hypertension, chest pain)      │  │
│  │  TREATMENT │ Medical procedures, therapies (surgery, chemotherapy)          │  │
│  │  TEST      │ Medical tests, examinations (MRI, blood pressure, HbA1c)       │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  FROM ner_deid_generic_augmented (Priority 3 - De-identification Entities):       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  NAME       │ Person names (Dr. Smith, John Doe)                            │  │
│  │  DATE       │ Temporal expressions (March 15, 2024, last week)              │  │
│  │  AGE        │ Patient age information (55-year-old, age 42)                 │  │
│  │  LOCATION   │ Places, addresses, facilities (New York, Room 302)            │  │
│  │  ID         │ Medical record numbers, SSN (MRN: 12345)                      │  │
│  │  PROFESSION │ Job titles, occupations (nurse, construction worker)          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  PRIORITY RESOLUTION:                                                             │
│  When entities overlap → Posology > Clinical > DeID                               │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Notebook Summary

| Notebook | Purpose | Key Actions | Output |
|----------|---------|-------------|--------|
| **[`data_prep.ipynb`](notebooks/data_prep.ipynb)** | Data loading & NER pipeline | Load MTSamples → Run 3 NER models → Merge with priority → Convert to CoNLL | `data/conll/*.conll` |
| **[`training.ipynb`](notebooks/training.ipynb)** | Custom model training | Load CoNLL → Configure MedicalNerApproach → Train with early stopping → Evaluate | `models/trained/custom_ner_model` |
| **[`prediction.ipynb`](notebooks/prediction.ipynb)** | Model inference | Load trained model → Run predictions → Visualize with NerVisualizer | Interactive HTML visualizations |

---

## 📋 Table of Contents

- [Project Navigator](#project-navigator)
- [End-to-End Flow](#end-to-end-flow)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Notebook Walkthrough](#notebook-walkthrough)
  - [Part 1: Data Preparation & NER Pipeline](#part-1-data-preparation--ner-pipeline-data_prepipynb)
  - [Part 2: Model Training](#part-2-model-training-trainingipynb)
  - [Part 3: Model Inference & Visualization](#part-3-model-inference--visualization-predictionipynb)
- [Training Features](#training-features)
- [Kaggle Notebook Links](#-kaggle-notebook-links-summary-table)
- [Troubleshooting](#troubleshooting)
- [References & Data Sources](#references--data-sources)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

---

## Project Navigator

- Back to portfolio home → [`../README.md`](../README.md)
- DocETL data curation API → [`../data_curation`](../data_curation)
- RAG QA chatbot → [`../rag_qa_chatbot_application`](../rag_qa_chatbot_application)

---

## End-to-End Flow

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               PROJECT WORKFLOW                                    │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  STEP 1: DATA PREPARATION                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  MTSamples Dataset (638 clinical texts)                                     │  │
│  │  • Medical transcriptions  • Surgery notes  • Clinical reports             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                           │
│                                       ▼                                           │
│  STEP 2: NER PIPELINE                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Multi-NER Pipeline with Priority Merging                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐    │  │
│  │  │ ner_posology (DRUG+DOSAGE)         │ Priority 1 (Highest)           │    │  │
│  │  │ ner_clinical_large                 │ Priority 2                     │    │  │
│  │  │ ner_deid_generic_augmented         │ Priority 3 (Lowest)            │    │  │
│  │  └─────────────────────────────────────────────────────────────────────┘    │  │
│  │                              ↓                                              │  │
│  │  ChunkMergeApproach (Conflict Resolution)                                   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                           │
│                                       ▼                                           │
│  STEP 3: CONLL GENERATION                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  CoNLL 2003 Format                                                          │  │
│  │  Token    POS  POS  BIO-Tag                                                 │  │
│  │  ─────────────────────────────                                              │  │
│  │  Aspirin  NN   NN   B-DRUG                                                  │  │
│  │  100mg    NN   NN   B-DOSAGE                                                │  │
│  │  for      NN   NN   O                                                       │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                           │
│                                       ▼                                           │
│  STEP 4: MODEL TRAINING                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Custom NER Model Training                                                  │  │
│  │  • embeddings_clinical     • MedicalNerApproach                             │  │
│  │  • Early Stopping          • Validation & Test Eval                         │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                           │
│                                       ▼                                           │
│  STEP 5: MODEL INFERENCE                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Predictions & Visualization                                                │  │
│  │  • LightPipeline (fast inference)    • NerVisualizer (HTML output)         │  │
│  │  • Entity extraction & analysis                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Outputs:** 
- `data/conll/*.conll` - CoNLL formatted training data
- `models/trained/custom_ner_model` - Fine-tuned NER model
- `ner_logs/` - Training metrics and evaluation logs

---

## Repository Layout

```
generating_conll_files_from_pretrained_models/
├── notebooks/
│   ├── data_prep.ipynb          # Part 1: Data loading, NER pipeline, CoNLL generation
│   ├── training.ipynb           # Part 2: Custom NER model training
│   └── prediction.ipynb         # Part 3: Model inference & visualization
├── src/
│   ├── dataset_loader.py
│   ├── ner_pipeline.py
│   ├── conll_converter.py
│   └── model_trainer.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── text_data.csv        # Prepared clinical texts
│   │   └── entities.csv         # Extracted entities
│   └── conll/
│       └── conll2003_text_file.conll  # Generated CoNLL file (6.5MB)
├── models/
│   └── trained/
│       └── custom_ner_model/    # Fine-tuned NER model
├── ner_logs/                    # Training logs with metrics
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement              | Minimum       | Notes                      |
| ------------------------ | ------------- | -------------------------- |
| Python                   | 3.8+          | Tested with 3.10           |
| Java                     | 8 or 11       | Required by Spark          |
| RAM                      | 16 GB         | 32 GB recommended          |
| Spark NLP for Healthcare | Valid license | Needed for `spark-nlp-jsl` |

Install Python deps:

```bash
pip install -r requirements.txt
```

Install Spark NLP for Healthcare (replace placeholders with your credentials):

```bash
pip install spark-nlp-jsl==<JSL_VERSION> \
  --extra-index-url https://pypi.johnsnowlabs.com/<SECRET>
```

---

## Quick Start

1. **Clone the repo**
   ```bash
   git clone <repository-url>
   cd generating_conll_files_from_pretrained_models
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure license**
   - Request a Healthcare license from John Snow Labs.
   - Create `spark_jsl.json` or export the provided environment variables.
4. **Launch notebooks**
   ```bash
   jupyter lab
   ```

---

## Notebook Walkthrough

### Part 1: Data Preparation & NER Pipeline ([`data_prep.ipynb`](notebooks/data_prep.ipynb))

This notebook handles the complete data preparation workflow: loading clinical texts, running multi-model NER with priority-based merging, and generating CoNLL-formatted training data.

#### Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-NER PIPELINE WITH PRIORITY MERGING                       │
│              Priority: Posology (DRUG/DOSAGE) > Clinical > DeID                   │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  INPUT TEXT                                                                       │
│  "The patient was prescribed Aspirin 100mg twice daily for hypertension."        │
│                              │                                                    │
│                              ▼                                                    │
│  STAGE 1: BASE NLP COMPONENTS                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  DocumentAssembler → SentenceDetectorDL → Tokenizer → WordEmbeddings        │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                    │
│                              ▼                                                    │
│  STAGE 2: PARALLEL NER MODELS                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ner_clinical_large                                                         │  │
│  │  Labels: PROBLEM, TREATMENT, TEST                                           │  │
│  │  Example: "hypertension" → PROBLEM                                          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ner_deid_generic_augmented                                                 │  │
│  │  Labels: NAME, DATE, AGE, ID, LOCATION, PROFESSION                          │  │
│  │  Example: "Dr. Smith" → NAME, "55-year-old" → AGE                           │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ner_posology                                                               │  │
│  │  Labels: DRUG, DOSAGE, STRENGTH, ROUTE, FREQUENCY, FORM, DURATION           │  │
│  │  Example: "Aspirin" → DRUG, "100mg" → DOSAGE                                │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                    │
│                              ▼                                                    │
│  STAGE 3: CHUNK FILTERING & MERGING                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ChunkFilterer (Posology)                                                   │  │
│  │  WhiteList: ["DRUG", "DOSAGE"] - Only keeps medication entities             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  ChunkMergeApproach                                                         │  │
│  │  InputCols: ["chunk_posology_filtered", "chunk_clinical", "chunk_deid"]     │  │
│  │  Priority: Posology (1st) → Clinical (2nd) → DeID (3rd)                     │  │
│  │  SelectionStrategy: "DiverseLonger" → Longest chunk wins                    │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                    │
│                              ▼                                                    │
│  OUTPUT: merged_ner_chunk                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Aspirin (DRUG) | 100mg (DOSAGE) | hypertension (PROBLEM)                   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### Key Pipeline Components

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| **SentenceDetectorDLModel** | `sentence_detector_dl_healthcare` | Healthcare-specific sentence splitting |
| **WordEmbeddingsModel** | `embeddings_clinical` (1.6GB) | 200-dim clinical word embeddings |
| **MedicalNerModel (Clinical)** | `ner_clinical_large` | PROBLEM, TREATMENT, TEST entities |
| **MedicalNerModel (DeID)** | `ner_deid_generic_augmented` | NAME, DATE, AGE, LOCATION, etc. |
| **MedicalNerModel (Posology)** | `ner_posology` | DRUG, DOSAGE, FREQUENCY, etc. |
| **ChunkFilterer** | WhiteList: ["DRUG", "DOSAGE"] | Filters posology to high-precision entities |
| **ChunkMergeApproach** | DiverseLonger strategy | Resolves conflicts using length priority |

#### WhiteList & Priority System Explained

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      WHY WHITELIST & PRIORITY MERGING?                            │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  PROBLEM: Multiple NER models may tag the SAME text span differently             │
│                                                                                   │
│  Example: "Metformin 500mg"                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ ner_posology:  "Metformin" → DRUG, "500mg" → DOSAGE                         │  │
│  │ ner_clinical:  "Metformin 500mg" → TREATMENT                                │  │
│  │ ner_deid:      (no match - not a protected entity)                          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  SOLUTION 1: WhiteList Filter                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ ChunkFilterer with WhiteList: ["DRUG", "DOSAGE"]                            │  │
│  │                                                                             │  │
│  │ • ner_posology outputs: DRUG, DOSAGE, STRENGTH, ROUTE, FREQUENCY, etc.     │  │
│  │ • After filtering: Only DRUG and DOSAGE remain                              │  │
│  │ • Reason: DRUG and DOSAGE are most critical for medication extraction      │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  SOLUTION 2: Priority-Based Merging                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ InputCols ORDER determines priority:                                        │  │
│  │                                                                             │  │
│  │   1. chunk_posology_filtered (HIGHEST PRIORITY)                             │  │
│  │      → Medication entities from specialized model                           │  │
│  │                                                                             │  │
│  │   2. chunk_clinical (MEDIUM PRIORITY)                                       │  │
│  │      → General clinical entities (PROBLEM, TREATMENT, TEST)                 │  │
│  │                                                                             │  │
│  │   3. chunk_deid (LOWEST PRIORITY)                                           │  │
│  │      → De-identification entities (NAME, DATE, etc.)                        │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  SOLUTION 3: DiverseLonger Strategy                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ When two entities overlap with SAME priority:                               │  │
│  │                                                                             │  │
│  │ OrderingFeatures: ["ChunkLength"]                                           │  │
│  │ SelectionStrategy: "DiverseLonger"                                          │  │
│  │                                                                             │  │
│  │ → The LONGER chunk wins                                                     │  │
│  │ → Preserves more contextual information                                     │  │
│  │ → Example: "severe chest pain" beats "chest pain"                           │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### ChunkMergeApproach Deep Dive: Step-by-Step Example

The `ChunkMergeApproach` is the core component that resolves conflicts when multiple NER models identify overlapping text spans. Here's a detailed walkthrough with a real clinical example:

**Input Text:**
```
"Dr. John Smith prescribed Lisinopril 10mg daily for hypertension on March 15, 2024."
```

**Step 1: Each NER Model Produces Annotations Independently**

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  RAW NER MODEL OUTPUTS (Before Merging)                                           │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ner_posology (Medication-focused):                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  "Lisinopril"  → DRUG       (position: 26-36)                               │  │
│  │  "10mg"        → DOSAGE     (position: 37-41)                               │  │
│  │  "daily"       → FREQUENCY  (position: 42-47)  ⚠️ Filtered by WhiteList    │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ner_clinical_large (General clinical):                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  "Lisinopril 10mg daily"  → TREATMENT (26-47) ⚠️ Overlaps with Posology!   │  │
│  │  "hypertension"           → PROBLEM   (52-64)                               │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ner_deid_generic_augmented (De-identification):                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  "Dr. John Smith"   → NAME  (position: 0-14)                                │  │
│  │  "March 15, 2024"   → DATE  (position: 68-82)                               │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Step 2: ChunkFilterer Applies WhiteList**

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  AFTER WHITELIST FILTERING (Posology only)                                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  WhiteList: ["DRUG", "DOSAGE"]                                                    │
│                                                                                   │
│  ner_posology_filtered:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  "Lisinopril"  → DRUG    (position: 26-36)  ✅ Kept                         │  │
│  │  "10mg"        → DOSAGE  (position: 37-41)  ✅ Kept                         │  │
│  │  "daily"       → FREQUENCY                  ❌ Removed (not in WhiteList)   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  Reason: FREQUENCY, ROUTE, FORM, DURATION are less critical for this use case   │
│          DRUG and DOSAGE provide highest value for medication extraction         │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Step 3: ChunkMergeApproach Resolves Conflicts**

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  CONFLICT DETECTION & RESOLUTION                                                  │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Conflict Zone: Position 26-47                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Text: "... prescribed Lisinopril 10mg daily for hypertension ..."          │  │
│  │  Position:              26──────────────47                                  │  │
│  │                                                                             │  │
│  │  OVERLAPPING ENTITIES:                                                      │  │
│  │  • Posology (Priority 1): [Lisinopril|DRUG] [10mg|DOSAGE] at 26-36, 37-41   │  │
│  │  • Clinical (Priority 2): [Lisinopril 10mg daily|TREATMENT] at 26-47        │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  Resolution Logic:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  1. CHECK PRIORITY ORDER (InputCols sequence)                               │  │
│  │     → First column = Highest Priority (posology_filtered)                   │  │
│  │                                                                             │  │
│  │  2. POSOLOGY HAS HIGHER PRIORITY → Posology entities WIN                    │  │
│  │     • "Lisinopril" (DRUG)              → ✅ SELECTED                        │  │
│  │     • "10mg" (DOSAGE)                  → ✅ SELECTED                        │  │
│  │     • "Lisinopril 10mg daily" (TREAT)  → ❌ DISCARDED (overlaps)            │  │
│  │                                                                             │  │
│  │  3. NON-OVERLAPPING ENTITIES PASS THROUGH                                   │  │
│  │     • "hypertension" (PROBLEM)         → ✅ SELECTED (no conflict)          │  │
│  │     • "Dr. John Smith" (NAME)          → ✅ SELECTED (no conflict)          │  │
│  │     • "March 15, 2024" (DATE)          → ✅ SELECTED (no conflict)          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Step 4: Final Merged Output**

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  FINAL MERGED OUTPUT (merged_ner_chunk)                                           │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Text: "Dr. John Smith prescribed Lisinopril 10mg for hypertension"              │
│                                                                                   │
│  Final Entities:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Dr. John Smith   │ NAME    │ ner_deid (no conflicts)                       │  │
│  │  Lisinopril       │ DRUG    │ ner_posology (priority over clinical)         │  │
│  │  10mg             │ DOSAGE  │ ner_posology (priority over clinical)         │  │
│  │  hypertension     │ PROBLEM │ ner_clinical (no conflicts)                   │  │
│  │  March 15, 2024   │ DATE    │ ner_deid (no conflicts)                       │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  NOTICE: "daily" (FREQUENCY) was not included because:                            │
│  • ChunkFilterer WhiteList only kept DRUG and DOSAGE from ner_posology           │
│  • ner_clinical's "Lisinopril 10mg daily" TREATMENT was discarded                │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**DiverseLonger Strategy: When Priorities are Equal**

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  SAME-PRIORITY CONFLICT EXAMPLE                                                   │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Scenario: Two entities from the SAME model (ner_clinical) overlap               │
│                                                                                   │
│  Text: "Patient has severe chest pain and shortness of breath."                  │
│                                                                                   │
│  ner_clinical outputs:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  "severe chest pain"    → PROBLEM (length: 17 chars)                        │  │
│  │  "chest pain"           → PROBLEM (length: 10 chars) ← Partial overlap      │  │
│  │  "shortness of breath"  → PROBLEM (length: 19 chars) ← No overlap           │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  DiverseLonger Resolution:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                             │  │
│  │  OrderingFeatures: ["ChunkLength"]                                          │  │
│  │  SelectionStrategy: "DiverseLonger"                                         │  │
│  │                                                                             │  │
│  │  Comparison: "severe chest pain" (17) vs "chest pain" (10)                  │  │
│  │              ↑ LONGER → WINS                                                │  │
│  │                                                                             │  │
│  │  Result: "severe chest pain" selected, "chest pain" discarded               │  │
│  │          "shortness of breath" kept (no conflict)                           │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  Rationale: Longer entities typically contain more context/diagnostic value      │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### Generated CoNLL Format Sample

The pipeline generates CoNLL 2003 format with 4 columns: **Token**, **POS**, **POS**, **BIO-Tag**

```
-DOCSTART- -X- -X- O

PROCEDURES NN NN O
PERFORMED NN NN O
: NN NN O
Colonoscopy NNP NNP B-TEST
. NN NN O

INDICATIONS NN NN O
: NN NN O
Renewed NNP NNP B-PROBLEM
symptoms NNP NNP I-PROBLEM
likely NN NN O
consistent NN NN O
with NN NN O
active NN NN O
flare NN NN O
of NN NN O
Inflammatory NNP NNP B-PROBLEM
Bowel NNP NNP I-PROBLEM
Disease NNP NNP I-PROBLEM
, NN NN O
not NN NN O
responsive NN NN O
to NN NN O
conventional NNP NNP B-TREATMENT
therapy NNP NNP I-TREATMENT
including NN NN O
sulfasalazine NNP NNP B-DRUG
, NN NN O
cortisone NNP NNP B-DRUG
, NN NN O
local NNP NNP B-TREATMENT
therapy NNP NNP I-TREATMENT
. NN NN O
```

**BIO Tagging Schema:**
- **B-** (Beginning): First token of an entity
- **I-** (Inside): Continuation tokens of an entity
- **O** (Outside): Non-entity tokens

---

### Part 2: Model Training ([`training.ipynb`](notebooks/training.ipynb))

This notebook fine-tunes a custom NER model using the CoNLL data generated in Part 1.

#### Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CUSTOM NER MODEL TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  INPUT: conll2003_text_file.conll (6.5MB, ~400K lines)                                          │
│         │                                                                                        │
│         ▼                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ CoNLL Data Loading                                                                        │   │
│  │ ┌────────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │ │ • 25,966 sentences (from 638 clinical documents)                                    │   │   │
│  │ │ • 80% Training / 20% Validation split                                               │   │   │
│  │ │ • Labels: 22 unique entity classes                                                  │   │   │
│  │ └────────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                                        │
│         ▼                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Embeddings Layer                                                                          │   │
│  │ ┌────────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Model: embeddings_clinical (John Snow Labs)                                         │   │   │
│  │ │ Size: 1.6 GB                                                                        │   │   │
│  │ │ Dimensions: 200                                                                     │   │   │
│  │ │ Language: English (Clinical Domain)                                                 │   │   │
│  │ └────────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                                        │
│         ▼                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ MedicalNerApproach                                                                        │   │
│  │ ┌────────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Neural Architecture: BiLSTM-CNN-CRF                                                 │   │   │
│  │ │   • Character-level CNN for morphological features                                  │   │   │
│  │ │   • Bidirectional LSTM for context                                                  │   │   │
│  │ │   • CRF layer for sequence labeling                                                 │   │   │
│  │ │                                                                                      │   │   │
│  │ │ Graph: medical-ner-dl/blstm_25_200_128_128.pb                                       │   │   │
│  │ └────────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                                        │
│         ▼                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Training Loop with Early Stopping                                                         │   │
│  │ ┌────────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │ │ Epoch 1 → 13 (Early stopping triggered)                                             │   │   │
│  │ │ Loss: 10416 → 1285 (decreasing)                                                     │   │   │
│  │ │ Validation F1: 0.83 → 0.89 (improving)                                              │   │   │
│  │ └────────────────────────────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                                        │
│         ▼                                                                                        │
│  OUTPUT: models/trained/custom_ner_model                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Training Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **max_epochs** | 35 | Maximum training epochs (early stopping may trigger earlier) |
| **lr** | 0.001 | Initial learning rate |
| **batch_size** | 8 | Samples per training batch |
| **random_seed** | 0 | For reproducibility |
| **validation_split** | 0.2 | 20% of data for validation |
| **early_stopping_criterion** | 0.02 | Minimum improvement threshold |
| **early_stopping_patience** | 6 | Epochs without improvement before stopping |
| **use_best_model** | True | Save best model based on validation metrics |

```python
# Training Pipeline Configuration (from training.ipynb)
training_pipeline = create_training_pipeline(
    clinical_embeddings=clinical_embeddings,
    max_epochs=35,
    lr=0.001,
    batch_size=batch_size,
    random_seed=0,
    verbose=1,
    test_dataset=test_data_path,
    output_logs_path=output_logs_path,
    validation_split=0.2,
    use_best_model=True,
    early_stopping_criterion=0.02,
    early_stopping_patience=6
)
```

#### Model Evaluation Results

The model was evaluated on a held-out test set after training:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL MODEL EVALUATION                                              │
│                        (Early Stopping at Epoch 13/35)                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Entity Performance (Test Set)                                                             │   │
│  ├────────────┬────────┬────────┬─────────┬───────────┬────────┬─────────┐                   │   │
│  │   Entity   │   TP   │   FP   │   FN    │ Precision │ Recall │   F1    │                   │   │
│  ├────────────┼────────┼────────┼─────────┼───────────┼────────┼─────────┤                   │   │
│  │ DATE       │   156  │    8   │   13    │   95.12%  │ 92.31% │ 93.69%  │ ← Best performer  │   │
│  │ AGE        │    76  │    3   │   13    │   96.20%  │ 85.39% │ 90.48%  │                   │   │
│  │ NAME       │    48  │    1   │    9    │   97.96%  │ 84.21% │ 90.57%  │                   │   │
│  │ DRUG       │   335  │   22   │   61    │   93.84%  │ 84.60% │ 88.98%  │                   │   │
│  │ PROBLEM    │  3203  │  445   │  475    │   87.80%  │ 87.09% │ 87.44%  │ ← Most frequent   │   │
│  │ TEST       │  1001  │  196   │  145    │   83.63%  │ 87.35% │ 85.45%  │                   │   │
│  │ TREATMENT  │  1889  │  335   │  427    │   84.94%  │ 81.56% │ 83.22%  │                   │   │
│  │ LOCATION   │    27  │    6   │   11    │   81.82%  │ 71.05% │ 76.06%  │                   │   │
│  │ DOSAGE     │    29  │   10   │   12    │   74.36%  │ 70.73% │ 72.50%  │                   │   │
│  │ ID         │     4  │    2   │    1    │   66.67%  │ 80.00% │ 72.73%  │                   │   │
│  │ PROFESSION │     5  │    1   │    7    │   83.33%  │ 41.67% │ 55.56%  │ ← Lowest support  │   │
│  └────────────┴────────┴────────┴─────────┴───────────┴────────┴─────────┘                   │   │
│                                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Aggregate Metrics                                                                         │   │
│  ├────────────────────────────┬──────────────────────────────────────────────────────────────┤   │
│  │ Macro Average F1           │ 81.51%                                                       │   │
│  │ Micro Average F1           │ 86.00%                                                       │   │
│  ├────────────────────────────┼──────────────────────────────────────────────────────────────┤   │
│  │ Total Training Examples    │ 16,614 sentences                                             │   │
│  │ Total Validation Examples  │ 4,154 sentences                                              │   │
│  │ Epochs Completed           │ 13/35 (Early Stopping)                                       │   │
│  └────────────────────────────┴──────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
│  Analysis:                                                                                       │
│  • DATE, AGE, NAME: High precision (95%+) due to distinctive patterns                          │
│  • DRUG: Strong performance (89% F1) from posology-focused training                            │
│  • PROBLEM: Highest volume (3203 TP), reliable general entity detection                        │
│  • PROFESSION: Low recall (42%) due to sparse training examples                                │
│  • Early stopping prevented overfitting after epoch 13                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Part 3: Model Inference & Visualization ([`prediction.ipynb`](notebooks/prediction.ipynb))

This notebook loads the trained model and generates interactive NER visualizations.

#### Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PREDICTION PIPELINE                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  INPUT: Raw clinical text                                                                        │
│  "The patient was prescribed Aspirin 100mg twice daily for pain management..."                  │
│         │                                                                                        │
│         ▼                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ LightPipeline (Optimized for single-document inference)                                   │   │
│  │                                                                                           │   │
│  │ Stages:                                                                                   │   │
│  │   1. DocumentAssembler → "document"                                                       │   │
│  │   2. SentenceDetector  → "sentence"                                                       │   │
│  │   3. Tokenizer         → "token"                                                          │   │
│  │   4. embeddings_clinical → "embeddings"                                                   │   │
│  │   5. custom_ner_model  → "ner" (Trained in Part 2)                                        │   │
│  │   6. NerConverterInternal → "ner_span"                                                    │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                                        │
│         ▼                                                                                        │
│  OUTPUT: Annotated entities with positions                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ chunk: "Aspirin"                 | begin: 27 | end: 33 | ner_label: DRUG                 │   │
│  │ chunk: "pain management"         | begin: 57 | end: 71 | ner_label: TREATMENT            │   │
│  │ chunk: "monitor blood pressure"  | begin: 92 | end: 113| ner_label: TEST                 │   │
│  │ chunk: "hypertension"            | begin: 122| end: 133| ner_label: PROBLEM              │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### NER Visualization with spark-nlp-display

The notebook uses `NerVisualizer` to create interactive HTML visualizations:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              NER VISUALIZATION OUTPUT                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  The patient was prescribed [Aspirin]      100mg twice daily for [pain management]              │
│                              ┌──────┐                             ┌────────────────┐            │
│                              │ DRUG │                             │   TREATMENT    │            │
│                              └──────┘                             └────────────────┘            │
│                                                                                                  │
│  and was advised to [monitor blood pressure]  due to [hypertension] .                           │
│                      ┌──────────────────────┐        ┌─────────────┐                            │
│                      │        TEST          │        │   PROBLEM   │                            │
│                      └──────────────────────┘        └─────────────┘                            │
│                                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                  │
│  Color Legend:                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │ ██ DRUG (Purple)     ██ PROBLEM (Dark Purple)   ██ TREATMENT (Mauve)                   │    │
│  │ ██ TEST (Blue)       ██ NAME (Green)            ██ DATE (Orange)                       │    │
│  │ ██ AGE (Teal)        ██ LOCATION (Brown)        ██ DOSAGE (Pink)                       │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                  │
│  Visualization Features:                                                                         │
│  • HTML export for sharing                                                                       │
│  • Color-coded entity labels                                                                     │
│  • Inline entity display                                                                         │
│  • Montserrat font for readability                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```python
# Visualization code from prediction.ipynb
from sparknlp_display import NerVisualizer

visualizer = NerVisualizer()

for txt in sample_texts:
    res = light_model.fullAnnotate(txt)[0]
    visualizer.display(
        res,
        label_col='ner_span',
        document_col='document',
        return_html=False
    )
```

---

## Training Features

- **Multi-model NER inference** (clinical, deid, posology) with conflict resolution via `ChunkMergeApproach`
- **Priority-based entity merging**: Posology (DRUG/DOSAGE) > Clinical > DeID
- **WhiteList filtering** for high-precision medication extraction
- **Tokenization aligned** with Spark NLP annotators for consistent CoNLL output
- **CoNLL writer** that enforces BIO tags and preserves evidence spans
- **Training pipeline** with configurable epochs, learning rate, batch size, and early stopping
- **Evaluation summary** (precision, recall, F1, macro/micro averages) logged per epoch
- **LightPipeline** for optimized single-document inference
- **NerVisualizer** for interactive HTML entity visualization

---

## 📑 Kaggle Notebook Links (Summary Table)

| Step | Notebook Name | Kaggle Link |
|------|---------------|-------------|
| **Part 1** | Data Preparation & CoNLL Generation | https://www.kaggle.com/code/huseyincenik/1-data-preparation-ner-pipeline-and-conll |
| **Part 2** | Custom NER Model Training | https://www.kaggle.com/code/huseyincenik/2-model-training-custom-ner-model-training |
| **Part 3** | Trained Model Inference & Predictions | https://www.kaggle.com/code/huseyincenik/3-prediction-trained-model-inference |

---

## Troubleshooting

| Issue                         | Resolution                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| Spark session fails           | Ensure Java is installed, `JAVA_HOME` is set, and increase driver memory (`spark.driver.memory=32G`). |
| License errors                | Double-check `SECRET`/`JSL_VERSION`, ensure internet access for model download.                       |
| Out-of-memory during training | Reduce batch size, sample fewer documents, or scale up cluster resources.                             |
| Model not found error         | Run `data_prep.ipynb` first, then `training.ipynb` before `prediction.ipynb`.                        |
| CoNLL parsing errors          | Verify CoNLL file has 4 columns (Token, POS, POS, BIO) separated by single spaces.                   |

---

## References & Data Sources

- [Spark NLP Workshop Datasets](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/data)
- [MTSamples](https://mtsamples.com/) - Medical transcription samples
- Spark NLP Healthcare certification notebooks:
  - `1.Clinical_Named_Entity_Recognition_Model.ipynb`
  - `1.3.prepare_CoNLL_from_annotations_for_NER.ipynb`
  - `1.4.Resume_MedicalNer_Model_Training.ipynb`

---

## Contribution Guidelines

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-enhancement`.
3. Commit and push your changes.
4. Open a pull request describing datasets, models, and environment details.

---

## License

This project depends on Spark NLP for Healthcare, which is a commercial offering. Ensure you have an active license key before running the notebooks.

---

**Built with ❤️ using Spark NLP for Healthcare**
