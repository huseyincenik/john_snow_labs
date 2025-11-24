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
    <img src="https://raw.githubusercontent.com/jupyter/notebook/master/docs/resources/jupyter.svg" alt="Jupyter" height="40">
  </p>
</div>

This project runs Spark NLP for Healthcare pipelines, converts predictions into CoNLL format, and fine-tunes custom NER models that can be reused by downstream doc-curation or RAG systems.

---

## Project Navigator

- Back to portfolio home → [`../README.md`](../README.md)
- DocETL data curation API → [`../data_curation`](../data_curation)
- RAG QA chatbot → [`../rag_qa_chatbot_application`](../rag_qa_chatbot_application)

---

## End-to-End Flow

```
┌────────────────────────────┐    ┌──────────────────────────────┐
│ Dataset Loader (Spark/CSV) │ -> │ Pretrained Healthcare NERs    │
└──────────────┬────────────┘    │ ner_clinical / ner_deid /     │
               │                 │ ner_posology w/ priority      │
               ▼                 └──────────────┬───────────────┘
       ┌─────────────────────┐                 │
       │ Entity Harmonizer   │                 ▼
       └──────────┬──────────┘         ┌────────────────────────┐
                  │                    │ CoNLL Converter        │
                  ▼                    │ (token-level tagging)  │
       ┌─────────────────────┐         └──────────┬─────────────┘
       │ Custom NER Training │--------------------┘
       │ (Embeddings +       │
       │  training pipeline) │
       └─────────────────────┘
```

Outputs: `data/conll/*.conll` files, model checkpoints in `models/trained/`, and notebook logs that record metrics (precision, recall, F1).

---

## Repository Layout

```
generating_conll_files_from_pretrained_models/
├── notebooks/
│   ├── 1_dataset_loading.ipynb
│   ├── 2_ner_pipeline.ipynb
│   ├── 3_conll_generation.ipynb
│   └── 4_custom_model_training.ipynb
├── src/
│   ├── dataset_loader.py
│   ├── ner_pipeline.py
│   ├── conll_converter.py
│   └── model_trainer.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── conll/
├── models/
│   └── trained/
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

| Notebook                        | Purpose                                                                              | Key Outputs                          |
| ------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------ |
| `1_dataset_loading.ipynb`       | Download/clean datasets (MTSamples, oncology notes, MIMIC)                           | Spark DataFrames in `data/raw`       |
| `2_ner_pipeline.ipynb`          | Run stacked Healthcare NER models with priority merging (Posology > DeID > Clinical) | Harmonized entity dataframe          |
| `3_conll_generation.ipynb`      | Convert tokens + entities into CoNLL 2003 compliant sequences                        | `data/conll/*.conll`                 |
| `4_custom_model_training.ipynb` | Fine-tune NER with embeddings, early stopping, evaluation                            | Saved models under `models/trained/` |

Example usage:

```python
from src.dataset_loader import DatasetLoader
from src.conll_converter import CoNLLConverter

loader = DatasetLoader(data_dir="data/raw")
df = loader.download_mtsamples_classifier()
text_df = loader.prepare_text_dataframe(df, text_column="text")

converter = CoNLLConverter(spark)
conll_text = converter.make_conll(
    text_df=text_df,
    entity_df=entity_df,
    save_conll=True,
    output_path="data/conll/mtsamples.conll"
)
```

---

## Training Features

- Multi-model NER inference (clinical, deid, posology) with conflict resolution.
- Tokenization aligned with Spark NLP annotators.
- CoNLL writer that enforces BIO tags and preserves evidence spans.
- Training pipeline with configurable epochs, LR, batch size, and early stopping.
- Evaluation summary (precision, recall, F1, macro/micro averages) logged per run.

---

## Troubleshooting

| Issue                         | Resolution                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| Spark session fails           | Ensure Java is installed, `JAVA_HOME` is set, and increase driver memory (`spark.driver.memory=32G`). |
| License errors                | Double-check `SECRET`/`JSL_VERSION`, ensure internet access for model download.                       |
| Out-of-memory during training | Reduce batch size, sample fewer documents, or scale up cluster resources.                             |

---

## References & Data Sources

- [Spark NLP Workshop Datasets](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/data)
- [MTSamples](https://mtsamples.com/)
- Spark NLP Healthcare certification notebooks (`1.Clinical_Named_Entity_Recognition_Model.ipynb`, `1.3.prepare_CoNLL_from_annotations_for_NER.ipynb`, `1.4.Resume_MedicalNer_Model_Training.ipynb`)

---

## Contribution Guidelines

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-enhancement`.
3. Commit and push your changes.
4. Open a pull request describing datasets, models, and environment details.

---

## License

This project depends on Spark NLP for Healthcare, which is a commercial offering. Ensure you have an active license key before running the notebooks.
