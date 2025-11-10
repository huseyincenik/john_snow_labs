# Spark NLP Healthcare NER Pipeline ve Custom Model Eğitimi

Bu proje, Spark NLP for Healthcare kullanarak Named Entity Recognition (NER) pipeline'ı çalıştırır, sonuçları CoNLL formatına dönüştürür ve custom NER modeli eğitir.

## Proje Yapısı

```
generating_conll_files_from_pretrained_models/
├── notebooks/              # Jupyter notebook'lar
│   ├── 1_dataset_loading.ipynb
│   ├── 2_ner_pipeline.ipynb
│   ├── 3_conll_generation.ipynb
│   └── 4_custom_model_training.ipynb
├── src/                   # Python modülleri
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── ner_pipeline.py
│   ├── conll_converter.py
│   └── model_trainer.py
├── data/
│   ├── raw/              # Ham dataset'ler
│   ├── processed/        # İşlenmiş veriler
│   └── conll/            # CoNLL formatındaki dosyalar
├── models/
│   └── trained/          # Eğitilmiş modeller
├── requirements.txt
└── README.md
```

## Gereksinimler

### Sistem Gereksinimleri
- Python 3.7+
- Java 8 veya 11
- En az 16GB RAM (önerilen)
- Spark NLP for Healthcare lisansı

### Python Paketleri

```bash
pip install -r requirements.txt
```

**Önemli:** Spark NLP for Healthcare lisanslı bir üründür. Kurulum için:

```bash
pip install spark-nlp-jsl==<JSL_VERSION> --extra-index-url https://pypi.johnsnowlabs.com/<SECRET>
```

`JSL_VERSION` ve `SECRET` değerleri lisans anahtarınızda bulunur.

## Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd generating_conll_files_from_pretrained_models
```

2. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Spark NLP Healthcare lisansını yapılandırın:**
   - John Snow Labs'tan lisans anahtarı alın
   - `spark_jsl.json` dosyası oluşturun veya environment variable'ları ayarlayın

## Kullanım

### 1. Dataset Yükleme

`notebooks/1_dataset_loading.ipynb` notebook'unu çalıştırarak dataset'i yükleyin:

```python
from src.dataset_loader import DatasetLoader

loader = DatasetLoader(data_dir="data/raw")
df = loader.download_mtsamples_classifier()
text_df = loader.prepare_text_dataframe(df, text_column="text")
```

**Desteklenen Dataset'ler:**
- `mtsamples_classifier`: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/data/mtsamples_classifier.csv
- `oncology_notes`: https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/data/oncology_notes
- MIMIC-III (manuel yükleme gerekir)

### 2. NER Pipeline Çalıştırma

`notebooks/2_ner_pipeline.ipynb` notebook'unu kullanarak NER pipeline'ını çalıştırın:

**Kullanılan Modeller:**
- `ner_clinical`: Klinik entity'ler (hastalıklar, prosedürler, vb.)
- `ner_deid_generic_augmented`: PHI (Protected Health Information) entity'ler
- `ner_posology`: İlaç ve dozaj entity'leri (Drug, Dosage öncelikli)

**Önceliklendirme:**
- Posology ve DeID modelleri önceliklidir
- Çakışan entity'lerde posology ve deid sonuçları tercih edilir

### 3. CoNLL Dosyası Oluşturma

`notebooks/3_conll_generation.ipynb` notebook'unu kullanarak NER sonuçlarını CoNLL formatına dönüştürün:

```python
from src.conll_converter import CoNLLConverter

converter = CoNLLConverter(spark)
conll_text = converter.make_conll(
    text_df=text_df,
    entity_df=entity_df,
    save_conll=True,
    output_path="data/conll/conll2003_text_file.conll"
)
```

**CoNLL Formatı:**
```
Token    Label
John     B-PER
Doe      I-PER
is       O
a        O
patient  O
```

### 4. Custom Model Eğitimi

`notebooks/4_custom_model_training.ipynb` notebook'unu kullanarak custom NER modeli eğitin:

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(spark)
trainer.load_embeddings()

# Load CoNLL dataset
training_data = trainer.load_conll_dataset("data/conll/conll2003_text_file.conll")

# Split dataset
train_data, validation_data = trainer.split_dataset(training_data, train_ratio=0.8)

# Create and train pipeline
pipeline = trainer.create_training_pipeline(
    max_epochs=10,
    lr=0.003,
    batch_size=8,
    use_best_model=True,
    early_stopping_patience=3
)

trained_model = trainer.train_model(train_data, pipeline)

# Evaluate
eval_results = trainer.evaluate_model(validation_data)

# Save model
trainer.save_model("models/trained/custom_ner_model")
```

## Özellikler

### NER Pipeline
- ✅ Tokenization ve sentence splitting
- ✅ Multiple NER model execution (clinical, deid, posology)
- ✅ Entity merging with priority (posology > deid > clinical)
- ✅ Posology model filtering (Drug, Dosage only)

### CoNLL Conversion
- ✅ Standard CoNLL format output
- ✅ B-/I- prefix labeling
- ✅ Token-level entity tagging
- ✅ Spark NLP tokenization integration

### Model Training
- ✅ Custom NER model training from CoNLL data
- ✅ Early stopping support
- ✅ Best model selection
- ✅ Evaluation metrics (precision, recall, F1-score)
- ✅ Model checkpointing and resuming

## Değerlendirme Metrikleri

Model değerlendirmesi şu metrikleri içerir:
- **Precision**: Doğru tahmin edilen entity'lerin tüm tahminlere oranı
- **Recall**: Doğru tahmin edilen entity'lerin tüm gerçek entity'lere oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **Macro-average**: Tüm entity tipleri için ortalama
- **Micro-average**: Tüm entity'ler için toplam

## Referans Notebook'lar

Bu proje, aşağıdaki Spark NLP workshop notebook'larından ilham almıştır:

1. **Clinical Named Entity Recognition (NER)**
   - `1.Clinical_Named_Entity_Recognition_Model.ipynb`

2. **Prepare CoNLL file from annotations for NER**
   - `1.3.prepare_CoNLL_from_annotations_for_NER.ipynb`

3. **Resume MedicalNer Model Training**
   - `1.4.Resume_MedicalNer_Model_Training.ipynb`

## Dataset URL'leri

- **mtsamples_classifier**: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/data/mtsamples_classifier.csv
- **oncology_notes**: https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/data/oncology_notes
- **mtsamples**: https://mtsamples.com/

## Sorun Giderme

### Spark Session Hatası
- Java'nın yüklü olduğundan emin olun
- `JAVA_HOME` environment variable'ını ayarlayın
- Spark driver memory'yi artırın

### Model Yükleme Hatası
- Spark NLP Healthcare lisansının geçerli olduğundan emin olun
- `SECRET` değerinin doğru olduğunu kontrol edin
- İnternet bağlantısını kontrol edin (model indirme için gerekli)

### Memory Hatası
- Spark driver memory'yi artırın: `spark.driver.memory=32G`
- Batch size'ı küçültün
- Dataset'i daha küçük parçalara bölün

## Lisans

Bu proje Spark NLP for Healthcare kullanır, bu da lisanslı bir üründür. Kullanım için John Snow Labs'tan lisans almanız gerekir.

## Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add some amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## İletişim

Sorularınız için issue açabilirsiniz.

## Teşekkürler

- John Snow Labs - Spark NLP for Healthcare
- Spark NLP Workshop repository

