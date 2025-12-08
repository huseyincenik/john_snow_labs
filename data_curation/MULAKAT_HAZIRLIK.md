# 🎯 John Snow Labs Data Curation Engineer Mülakat Hazırlık Dokümanı

## 📋 İçindekiler

1. [Projenin Ne Anlama Geldiği](#projenin-ne-anlama-geldiği)
2. [Sorulabilecek Sorular ve Cevaplar](#sorulabilecek-sorular-ve-cevaplar)
3. [Nelere Dikkat Etmek Gerekiyor](#nelere-dikkat-etmek-gerekiyor)
4. [Proje Yapısı ve Dosya Yolları](#proje-yapısı-ve-dosya-yolları)
5. [Teknik Detaylar](#teknik-detaylar)
6. [NAACCR Standartları ve Kanser Kayıt Sistemi](#naaccr-standartları-ve-kanser-kayıt-sistemi)
7. [FastAPI Yapısı ve İstek Akışı](#fastapi-yapısı-ve-istek-akışı)
8. [Mülakat İpuçları](#mülakat-ipuçları)

---

## 🎯 Projenin Ne Anlama Geldiği

### Genel Bakış

Bu proje, **John Snow Labs** için geliştirilmiş bir **Data Curation Service**'tir. Temel amacı, yapılandırılmamış klinik dokümanlardan (metin, PDF, JSON) yapılandırılmış tıbbi veriler çıkarmak ve bunları **NAACCR (North American Association of Central Cancer Registries)** standartlarına uygun şekilde normalize etmektir.

### Projenin Amacı

1. **Doküman Seviyesi Çıkarım (Document-Level Extraction)**: Her dokümandan kanser kayıt alanlarını (diagnosis, staging, performance status vb.) çıkarma
2. **Hasta Seviyesi Konsolidasyon (Patient-Level Consolidation)**: Aynı hastaya ait birden fazla dokümandan gelen bilgileri birleştirme ve normalize etme
3. **Audit Trail (Denetim İzleme)**: Her işlem için tam loglama ve provenance (kaynak) takibi
4. **Multi-LLM Desteği**: Hem bulut (OpenAI) hem de yerel (Qwen/Gemma) LLM modellerini destekleme

### Kullanım Senaryoları

- **Kanser Kayıt Sistemleri**: Hastanelerden gelen yapılandırılmamış dokümanları otomatik olarak yapılandırılmış veriye dönüştürme
- **Klinik Araştırmalar**: Hasta verilerini standart formatta toplama ve analiz etme
- **Veri Kalitesi İyileştirme**: Manuel veri girişini azaltma ve hata oranını düşürme

---

## ❓ Sorulabilecek Sorular ve Cevaplar

### 1. Projeyi Nasıl Açıklarsınız?

**Cevap:**
"Bu proje, **DocETL** framework'ünü kullanarak yapılandırılmamış klinik dokümanlardan yapılandırılmış veri çıkaran bir ETL pipeline'ıdır. Pipeline üç ana aşamadan oluşur:

1. **Tagger**: Dokümanları kronolojik olarak sıralar ve güven skorları atar
2. **Extractor**: Her dokümandan NAACCR standartlarına uygun alanları çıkarır (diagnosis date, TNM staging, performance status vb.)
3. **Consolidator**: Aynı hastaya ait farklı dokümanlardan gelen bilgileri birleştirir ve normalize eder

Tüm işlemler FastAPI üzerinden REST API olarak sunulur ve her session için tam loglama yapılır."

### 2. DocETL Nedir ve Nasıl Kullandınız?

**Cevap:**
"DocETL, UC Berkeley tarafından geliştirilmiş, LLM tabanlı doküman işleme için özel bir ETL framework'üdür. Projede şu operatörleri kullandım:

- **MapOp**: Her dokümandan `cancer_registry_fields.yaml` ontolojisine göre alanları çıkarır
- **UnnestOp**: Çıkarılan alanları düzleştirir (her alan bir satır olur)
- **ResolveOp**: Aynı hasta ve alan için çakışan değerleri çözer (en güvenilir kaynağı seçer)
- **ReduceOp**: Çözümlenmiş alanları hasta seviyesinde özetler

DocETL, memory-based dataset kullanır, yani geçici dosyalar oluşturmadan işlem yapar. Ancak her aşama için checkpoint'ler `data/output/<session_id>/docetl_intermediate/` altında saklanır."

**Detaylı Açıklama:**

Projede DocETL'in kullanımı `src/pipeline/docetl_runner.py` dosyasında gerçekleştirilir. İşte detaylar:

#### DocETL Pipeline Yapısı

**1. Dataset Oluşturma:**
```python
dataset = Dataset(
    type="memory",  # Memory-based, geçici dosya yok
    path=raw_records,  # Python dict listesi
)
```

**2. Pipeline Operatörleri:**

**MapOp (`extract_clinical_fields`):**
- **Görev**: Her dokümandan NAACCR alanlarını çıkarır
- **Input**: Doküman metadata (patient_id, doc_id, content)
- **Output**: `extractions` array'i içeren JSON objesi
- **LLM Prompt**: 800+ satırlık detaylı prompt (confidence score kuralları, field-specific talimatlar)
- **Validation**: Pydantic v2 şemaları ile otomatik validate
- **Özellikler**:
  - Structured output mode (JSON schema validation)
  - Qwen modelleri için özel optimizasyonlar (timeout, retry, reasoning_effort)
  - Confidence score calibration protocol (5-step process)
  - Truncated JSON düzeltme mekanizması

**UnnestOp (`explode_field_records`):**
- **Görev**: Map çıktısındaki `extractions` array'ini düzleştirir
- **Input**: `{doc_id: "...", extractions: [...]}`
- **Output**: Her alan için ayrı satır: `{patient_id, doc_id, field_name, raw_value, ...}`
- **Kullanım**: Resolve işlemi için veriyi hazırlar

**ResolveOp (`resolve_patient_fields`):**
- **Görev**: Aynı `(patient_id, field_name)` için çakışan değerleri çözer
- **Blocking Keys**: `patient_id`, `field_name`
- **Çözümleme Stratejisi**:
  - En yüksek confidence score'a sahip değeri seçer
  - Pathology > Operative > Clinical > Radiology öncelik sırası
  - En yeni doküman tarihine öncelik verir
- **Output**: Her `(patient_id, field_name)` için tek bir `resolved_value`
- **Provenance**: Hangi dokümandan geldiği kaydedilir

**ReduceOp (`reduce_patient_summary`):**
- **Görev**: Resolve edilmiş alanları hasta seviyesinde birleştirir
- **Input**: Resolve çıktısı (her hasta için çözümlenmiş alanlar)
- **Output**: `{patient_id, consolidated_fields: [...], patient_summary: "..."}`
- **LLM Prompt**: Consolidation reasoning üretir
- **Özellikler**:
  - mCODE v4.0.0 formatına uygun yapı
  - Timeline yapıları (cancer_stage, disease_status)
  - Primary cancers array yapısı

**3. Pipeline Execution:**

```python
pipeline = Pipeline(
    name=f"docetl_session_{session_id}",
    datasets={"clinical_docs": dataset},
    operations=[map_op, normalize_op, unnest_op, resolve_op, reduce_op],
    steps=[PipelineStep(...)],
    output=PipelineOutput(
        type="file",
        path="docetl_patient_results.json",
        intermediate_dir="docetl_intermediate/",
    ),
    default_model=self.model_name,
)

pipeline.run(max_threads=200)  # Paralel işleme
```

**4. Checkpoint ve Resume Mekanizması:**
- Her operatör çıktısı `docetl_intermediate/<step>/<op>.json` olarak kaydedilir
- Pipeline hata verirse, mevcut checkpoint'lerden devam eder
- Map, Resolve, Reduce çıktıları ayrı ayrı kontrol edilir

**5. Özel Özellikler:**

**JSON Truncation Handling:**
- LLM çıktısı kesilirse otomatik düzeltme
- Eksik closing brackets/quotes tamamlanır
- Markdown code fence'ler temizlenir

**Model-Specific Optimizations:**
- **Qwen**: Daha kısa timeout (20s), daha az retry (1), reasoning_effort="none"
- **OpenAI**: Daha uzun timeout (40s), daha fazla retry (3)
- Her model için özel `completion_kwargs`

**Error Handling:**
- Transient hatalar için otomatik retry (httpx.ConnectError, LiteLLMAPIError)
- Exponential backoff (3-10 saniye)
- Partial checkpoint'lerden resume

**6. Output Artifacts:**

- **Map Output**: `docetl_intermediate/clinical_registry/extract_clinical_fields.json`
- **Resolve Output**: `docetl_intermediate/clinical_registry/resolve_patient_fields.json`
- **Reduce Output**: `docetl_patient_results.json`
- **Logs**: `logs/<session_id>/stage_extractor_prompts.log`, `stage_consolidator_prompts.log`

**7. Thread Management:**

- Dinamik thread sayısı hesaplama:
  - Base: `max(len(documents), 1)`
  - CPU-based: `cpu_count * 5` (I/O-bound için)
  - Settings-based: `max(max_workers, max_concurrent_requests * 3)`
  - Cap: `min(desired_threads, docetl_max_threads=200)`

**Dosya Yolu:** `src/pipeline/docetl_runner.py` (2142 satır)

### 3. Ontoloji Dosyası Nedir ve Nasıl Kullanıldı?

**Cevap:**
"`cancer_registry_fields.yaml` dosyası, NAACCR standartlarına uygun kanser kayıt alanlarını tanımlar. Bu dosya:

- **12 ana alan** içerir: diagnosis date, cancer site, histology code, TNM staging (clinical ve pathological), summary stage, ECOG/KPS performance status
- Her alan için **detaylı talimatlar** içerir (LLM prompt'larında kullanılır)
- **ICD-O-3** ve **AJCC** kodlama standartlarını takip eder

Extractor, bu ontolojiyi dinamik olarak yükler ve her alan için Pydantic v2 şemaları oluşturur. Bu sayede type-safety sağlanır ve LLM çıktıları otomatik olarak validate edilir."

**Dosya Yolu:** `data/ontology/cancer_registry_fields.yaml` veya `cancer_registry_fields.yaml` (root)

### 4. Multi-LLM Desteği Nasıl Çalışıyor?

**Cevap:**
"Proje, **OpenRouter** üzerinden hem OpenAI hem de Qwen modellerini destekler. OpenRouter, tek bir API key ile 60+ provider ve 300+ model'e erişim sağlar.

- **OpenAI**: `openai/gpt-4o-mini` (varsayılan)
- **Qwen**: `openrouter/qwen/qwen3-8b` veya `qwen-2.5-7b-instruct`

LLM çağrıları `src/utils/llm.py` içinde **LiteLLM** kullanılarak yapılır. LiteLLM, OpenAI-compatible API sağlar, bu yüzden DocETL ile uyumludur.

Konfigürasyon `config/.env` dosyasında yapılır ve `config/settings.py` üzerinden yüklenir."

### 5. Concurrency ve Performance Optimizasyonları Nelerdir?

**Cevap:**
"Proje, yüksek performans için çoklu seviyede paralellik kullanır:

1. **Document-Level Parallelism**: Extractor, semaphore ile maksimum 30 eşzamanlı LLM çağrısı yapar
2. **Patient-Level Parallelism**: Farklı hastalar paralel işlenir (varsayılan 12 hasta)
3. **Thread Pool**: DocETL pipeline'ı için 200 thread kullanılır
4. **Async/Await**: FastAPI background tasks ile non-blocking işlem

Ayarlar `config/settings.py` içinde yapılandırılabilir:
- `max_concurrent_requests: 30`
- `max_parallel_patients: 12`
- `docetl_max_threads: 200`"

### 6. Logging ve Audit Trail Nasıl Çalışıyor?

**Cevap:**
"Her işlem için benzersiz bir `session_id` oluşturulur. Loglar şu yapıda saklanır:

```
logs/
└── <session_id>/
    ├── stage_tagger.log
    ├── stage_extractor.log
    ├── stage_extractor_prompts.log
    ├── stage_consolidator.log
    └── stage_consolidator_prompts.log
```

Her log dosyası:
- Timestamp'li yapılandırılmış loglar içerir
- LLM prompt'larını ve response'larını kaydeder
- Hata durumlarında full stack trace içerir
- Retry mekanizmalarını loglar

Bu sayede her çıkarımın kaynağı (hangi doküman, hangi satır) ve gerekçesi (LLM reasoning) tam olarak takip edilebilir."

### 7. Output Formatları Nelerdir?

**Cevap:**
"İki seviyede output üretilir:

1. **Document-Level**: `data/output/<session_id>/stage_stage_extractor_<session>_extraction.json`
   - Her doküman için çıkarılan alanlar
   - `raw_value`, `normalized_value`, `reasoning_excerpt`, `explanation` içerir

2. **Patient-Level**: `data/output/<session_id>/stage_stage_consolidator_<session>_consolidation.json`
   - Hasta seviyesinde birleştirilmiş ve normalize edilmiş alanlar
   - Provenance bilgisi (hangi dokümandan geldiği)
   - Consolidated reasoning (neden bu değer seçildi)

Ayrıca DocETL intermediate results: `data/output/<session_id>/docetl_intermediate/` altında saklanır."

### 8. Docker ve Deployment Nasıl Yapıldı?

**Cevap:**
"Proje tamamen containerize edilmiştir:

- **Dockerfile**: Python 3.10+ base image, `uv` package manager kullanır
- **docker-compose.yml**: FastAPI servisini ayağa kaldırır
- **run_docker.sh**: Kolay başlatma script'i

Docker container içinde:
- `config/.env` dosyası volume olarak mount edilir
- Tüm bağımlılıklar `uv sync` ile yüklenir
- API `http://localhost:8000` üzerinden erişilebilir

**Not**: LLM çağrıları OpenRouter üzerinden yapıldığı için container içinde model indirmeye gerek yoktur."

### 9. API Endpoints Nelerdir ve FastAPI Nasıl Çalışıyor?

**Cevap:**
"Üç ana endpoint var:

1. **POST `/api/v1/process`**: Dokümanları işleme başlatır
   - `patient_ids`: İşlenecek hasta ID'leri
   - `llm_provider`: "openai" veya "qwen"
   - `llm_model`: Model override (opsiyonel)
   - Response: `session_id` döner

2. **GET `/api/v1/status/{session_id}`**: İşlem durumunu kontrol eder
   - Response: `status`, `tagger_result`, `extraction_result`, `consolidation_result`

3. **POST `/api/v1/upload`**: Yeni doküman yükler
   - `files`: Upload edilecek dosyalar
   - Response: Yüklenen dosya listesi

Tüm işlemler **background task** olarak çalışır, yani API hemen response döner."

**FastAPI Yapısı ve İstek Akışı:**

#### FastAPI Uygulama Yapısı

**Entry Point**: `src/main.py`
```python
app = FastAPI(
    title="Data Curation Service",
    description="DocETL-based medical document processing service",
    version="0.1.0",
)

# CORS middleware eklenir
app.add_middleware(CORSMiddleware, ...)

# Router'lar include edilir
app.include_router(router, prefix="/api/v1", tags=["processing"])
```

**Router Tanımları**: `src/api/routes.py`
- `router = APIRouter()` - API route'ları burada tanımlanır
- Tüm endpoint'ler `/api/v1` prefix'i ile erişilir

#### Endpoint Detayları ve İstek Akışı

**1. POST `/api/v1/process` - İstek Akışı:**

```
Client Request (POST /api/v1/process)
    │
    ├─> FastAPI Route Handler (routes.py:29-66)
    │       │
    │       ├─> Request Validation (Pydantic)
    │       │   └─> ProcessingRequest schema kontrolü
    │       │
    │       ├─> Session ID Oluşturma
    │       │   └─> uuid.uuid4() → session_id
    │       │
    │       ├─> Doküman Yükleme (load_documents)
    │       │   └─> data/input/*.txt dosyaları okunur
    │       │   └─> DocumentMetadata objeleri oluşturulur
    │       │
    │       ├─> Background Task Oluşturma
    │       │   └─> background_tasks.add_task(process_documents_task, ...)
    │       │
    │       └─> Immediate Response
    │           └─> ProcessingResponse(session_id, status="processing")
    │
    └─> Background Task (process_documents_task)
            │
            ├─> Tagger Çalıştırma
            │   └─> tagger.tag_documents(documents, session_id)
            │
            ├─> Extractor Çalıştırma
            │   └─> extractor.extract(tagged_docs, session_id)
            │       └─> DocETL Pipeline (Map → Unnest → Resolve → Reduce)
            │
            └─> Consolidator Çalıştırma
                └─> consolidator.consolidate(extraction_result, ...)
```

**Kod Yolu**: `src/api/routes.py:29-66, 177-265`

**2. GET `/api/v1/status/{session_id}` - İstek Akışı:**

```
Client Request (GET /api/v1/status/{session_id})
    │
    ├─> FastAPI Route Handler (routes.py:69-89)
    │       │
    │       ├─> Session Kontrolü
    │       │   └─> processing_status[session_id] kontrolü
    │       │
    │       ├─> Storage'dan Sonuçları Yükleme
    │       │   ├─> storage.load_tagger(session_id)
    │       │   ├─> storage.load_extraction(session_id)
    │       │   └─> storage.load_consolidation(session_id)
    │       │
    │       └─> Response
    │           └─> ProcessingResponse(status, results)
```

**Kod Yolu**: `src/api/routes.py:69-89`

**3. POST `/api/v1/upload` - İstek Akışı:**

```
Client Request (POST /api/v1/upload, multipart/form-data)
    │
    ├─> FastAPI Route Handler (routes.py:92-111)
    │       │
    │       ├─> File Upload İşleme
    │       │   ├─> Her dosya için:
    │       │   │   ├─> await file.read() (async file reading)
    │       │   │   └─> data/input/ klasörüne yazma
    │       │   │
    │       │   └─> Uploaded files listesi oluşturma
    │       │
    │       └─> Response
    │           └─> {"message": "...", "files": [...]}
```

**Kod Yolu**: `src/api/routes.py:92-111`

#### FastAPI Özellikleri

**1. Async/Await Kullanımı:**
- Tüm endpoint'ler `async def` olarak tanımlanmış
- `aiofiles` ile async file I/O
- `asyncio.gather()` ile paralel doküman parsing

**2. Background Tasks:**
- `BackgroundTasks` ile uzun süren işlemler arka planda çalışır
- API hemen response döner (non-blocking)
- Status polling ile ilerleme takibi

**3. Pydantic Validation:**
- Request/Response modelleri Pydantic v2 ile validate edilir
- `response_model=ProcessingResponse` ile otomatik validation
- Type-safe API garantisi

**4. CORS Middleware:**
- Tüm origin'lere izin verilir (`allow_origins=["*"]`)
- Production'da kısıtlanmalı

**5. Error Handling:**
- `HTTPException` ile standart hata yanıtları
- 400: Bad Request (no documents found)
- 404: Not Found (session not found)
- 500: Internal Server Error (background task hataları)

#### İstek Örnekleri

**Process Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_ids": ["p01"],
    "llm_provider": "openai",
    "llm_model": "openai/gpt-4o-mini"
  }'
```

**Status Check:**
```bash
curl "http://localhost:8000/api/v1/status/{session_id}"
```

**Upload Files:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "files=@document1.txt" \
  -F "files=@document2.txt"
```

#### FastAPI Dosya Yapısı

| Dosya | Açıklama | Önem |
|-------|----------|------|
| `src/main.py` | FastAPI app tanımı, middleware, router include | ⭐⭐⭐ Kritik |
| `src/api/routes.py` | Tüm endpoint tanımları | ⭐⭐⭐ Kritik |
| `src/models/schemas.py` | Pydantic request/response modelleri | ⭐⭐ Önemli |

#### Server Başlatma

**Development:**
```bash
uv run python main.py
# veya
uv run uvicorn src.main:app --reload
```

**Production:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Docker:**
- `docker-compose.yml` içinde uvicorn otomatik başlatılır
- Port: 8000 (configurable via settings)

### 10. Hata Yönetimi ve Retry Mekanizması Nasıl?

**Cevap:**
"Proje, çok katmanlı hata yönetimi kullanır:

1. **LLM Level**: `tenacity` library ile retry:
   - `APIConnectionError`, `APITimeoutError`, `RateLimitError` için otomatik retry
   - Exponential backoff (3-10 saniye arası)
   - Maksimum 3 deneme

2. **DocETL Level**: Schema validation retry:
   - LLM çıktısı şemaya uymazsa, prompt tekrar gönderilir
   - Maksimum 2 validation retry

3. **Application Level**: Exception handling:
   - Tüm hatalar loglanır
   - Session status `failed` olarak işaretlenir
   - Stack trace tam olarak kaydedilir

Hata durumlarında kullanıcı `/status` endpoint'i üzerinden detaylı hata mesajını görebilir."

### 11. LLM Prompt'ları Nerede Tanımlanmış?

**Cevap:**
"Projede **4 farklı prompt** kullanılır ve her biri farklı dosyalarda tanımlanmıştır:

1. **MapOp Prompt (Extractor)**: `src/pipeline/docetl_runner.py` dosyasında `_build_map_operation()` metodunda (satır 604-826). Bu 800+ satırlık en detaylı prompt'tur ve confidence score calibration protocol, field-specific extraction rules, NAACCR/ICD-O-3 format talimatları içerir.

2. **ResolveOp Prompts (Consolidator)**: Aynı dosyada `_build_resolve_operation()` metodunda (satır 1036-1101). İki ayrı prompt var: `comparison_prompt` (iki değeri karşılaştırır) ve `resolution_prompt` (çakışan değerleri çözer).

3. **ReduceOp Prompt (Consolidator)**: `_build_reduce_operation()` metodunda (satır 1194-1283). Patient-level consolidation için kullanılır.

4. **Tagger Prompt**: `src/pipeline/tagger.py` dosyasında `_build_confidence_prompt()` metodunda (satır 235-274). Document type ve date confidence assessment için kullanılır.

Tüm prompt'lar **Jinja2 template engine** kullanarak render edilir ve otomatik olarak loglanır. Log dosyaları `logs/<session_id>/stage_*_prompts.log` formatında saklanır."

---

## ⚠️ Nelere Dikkat Etmek Gerekiyor

### 1. **Ontoloji Dosyasının Önemi**
- `cancer_registry_fields.yaml` dosyası **kritik** bir bileşendir
- Tüm extraction ve consolidation bu dosyaya göre yapılır
- Dosya yolu: `data/ontology/cancer_registry_fields.yaml` veya root'ta `cancer_registry_fields.yaml`
- Bu dosyayı değiştirirseniz, tüm pipeline etkilenir

### 2. **Session ID Yönetimi**
- Her işlem için benzersiz `session_id` oluşturulur
- Loglar ve output'lar bu ID'ye göre organize edilir
- Production'da Redis veya database kullanılmalı (şu an in-memory)

### 3. **LLM API Key Güvenliği**
- `config/.env` dosyası **asla** commit edilmemeli
- `.gitignore` içinde olmalı
- OpenRouter API key formatı: `sk-or-v1-...`

### 4. **Concurrency Limitleri**
- Çok fazla paralel istek API rate limit'lerini tetikleyebilir
- `max_concurrent_requests: 30` ayarı optimize edilmiş
- Qwen modelleri genelde daha hızlı, OpenAI daha yavaş

### 5. **Output Disk Kullanımı**
- Her session için çok sayıda JSON dosyası oluşturulur
- `data/output/` ve `logs/` klasörleri düzenli temizlenmeli
- Production'da S3 veya benzeri object storage kullanılmalı

### 6. **DocETL Memory Kullanımı**
- DocETL memory-based dataset kullanır
- Çok büyük doküman setleri için memory sorunları olabilir
- Batch processing düşünülebilir

### 7. **Type Safety**
- Pydantic v2 şemaları dinamik olarak oluşturulur
- Ontoloji değişirse şemalar da otomatik güncellenir
- LLM çıktıları otomatik validate edilir

### 8. **Error Handling**
- Background task'larda exception'lar yakalanmalı
- Status güncellemeleri her zaman yapılmalı
- Loglama eksiksiz olmalı

---

## 📁 Proje Yapısı ve Dosya Yolları

### Ana Dizin Yapısı

```
data_curation/
├── src/                          # Ana kaynak kod
│   ├── api/                      # FastAPI endpoints
│   │   └── routes.py             # API route tanımları
│   ├── pipeline/                 # ETL pipeline bileşenleri
│   │   ├── tagger.py            # Doküman sıralama ve etiketleme
│   │   ├── extractor.py         # Alan çıkarımı (DocETL Map)
│   │   ├── consolidator.py      # Hasta seviyesi birleştirme (Resolve+Reduce)
│   │   └── docetl_runner.py     # DocETL pipeline orchestration
│   ├── models/                   # Pydantic şemaları
│   │   └── schemas.py           # Request/Response modelleri
│   ├── utils/                    # Yardımcı fonksiyonlar
│   │   ├── llm.py               # LLM provider wrapper (LiteLLM)
│   │   ├── logger.py            # Logging setup
│   │   ├── ontology.py          # Ontoloji yükleme
│   │   └── storage.py            # JSON/PostgreSQL storage
│   └── main.py                   # FastAPI app entry point
├── config/                       # Konfigürasyon
│   ├── settings.py              # Pydantic Settings (env variables)
│   └── .env                      # API keys (gitignore'da)
├── data/                         # Veri dosyaları
│   ├── input/                    # Giriş dokümanları (.txt)
│   ├── output/                   # Çıktı JSON'ları
│   │   └── <session_id>/         # Session bazlı output'lar
│   │       ├── stage_stage_tagger_*.json
│   │       ├── stage_stage_extractor_*.json
│   │       ├── stage_stage_consolidator_*.json
│   │       └── docetl_intermediate/  # DocETL checkpoint'leri
│   └── ontology/                 # Ontoloji dosyası (opsiyonel)
│       └── cancer_registry_fields.yaml
├── logs/                         # Log dosyaları
│   └── <session_id>/             # Session bazlı loglar
│       ├── stage_tagger.log
│       ├── stage_extractor.log
│       ├── stage_extractor_prompts.log
│       ├── stage_consolidator.log
│       └── stage_consolidator_prompts.log
├── scripts/                      # Yardımcı script'ler
│   └── e2e_demo.py              # End-to-end test script'i
├── input_patient_docs/           # Örnek hasta dokümanları
├── cancer_registry_fields.yaml   # NAACCR ontoloji (root)
├── main.py                       # Entry point shim
├── pyproject.toml                # Python proje konfigürasyonu
├── Dockerfile                    # Docker image tanımı
├── docker-compose.yml            # Docker Compose konfigürasyonu
├── run_docker.sh                 # Docker başlatma script'i
└── README.md                     # Proje dokümantasyonu
```

### Kritik Dosya Yolları

| Dosya/Dizin | Açıklama | Önem |
|------------|----------|------|
| `cancer_registry_fields.yaml` | NAACCR ontoloji tanımları | ⭐⭐⭐ Kritik |
| `config/.env` | API keys ve konfigürasyon | ⭐⭐⭐ Kritik (gitignore) |
| `config/settings.py` | Pydantic Settings sınıfı | ⭐⭐ Önemli |
| `src/main.py` | FastAPI app entry point, middleware, router registration | ⭐⭐⭐ Kritik |
| `src/api/routes.py` | FastAPI endpoint tanımları ve business logic | ⭐⭐⭐ Kritik |
| `src/models/schemas.py` | Pydantic request/response modelleri | ⭐⭐ Önemli |
| `src/pipeline/docetl_runner.py` | DocETL pipeline orchestration, **MapOp/ResolveOp/ReduceOp prompt'ları** | ⭐⭐⭐ Kritik |
| `src/pipeline/extractor.py` | Alan çıkarımı mantığı | ⭐⭐⭐ Kritik |
| `src/pipeline/consolidator.py` | Hasta seviyesi birleştirme | ⭐⭐⭐ Kritik |
| `src/pipeline/tagger.py` | Doküman sıralama, **Tagger prompt'u** | ⭐⭐ Önemli |
| `src/utils/llm.py` | LLM provider wrapper | ⭐⭐ Önemli |
| `src/utils/ontology.py` | Ontoloji yükleme | ⭐⭐ Önemli |
| `data/output/<session_id>/` | Session output'ları | ⭐⭐ Önemli |
| `logs/<session_id>/` | Session log'ları | ⭐⭐ Önemli |

---

## 🔧 Teknik Detaylar

### Kullanılan Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.10+ | Ana programlama dili |
| FastAPI | 0.104+ | REST API framework |
| Pydantic | v2 | Type-safe data validation |
| DocETL | 0.2.5+ | LLM-based ETL framework |
| LiteLLM | 1.75+ | Multi-provider LLM wrapper |
| OpenRouter | - | Unified LLM API gateway |
| Uvicorn | 0.24+ | ASGI server |
| uv | latest | Package manager |
| Docker | - | Containerization |

### Pipeline Akışı

```
1. Doküman Yükleme
   └─> data/input/*.txt dosyaları okunur
   └─> DocumentMetadata objeleri oluşturulur

2. Tagger (Opsiyonel)
   └─> LLM ile dokümanlar kronolojik sıralanır
   └─> Güven skorları atanır
   └─> Output: stage_stage_tagger_<session>_sorted.json

3. Extractor (DocETL Map)
   └─> Her doküman için cancer_registry_fields.yaml alanları çıkarılır
   └─> LLM her alan için: raw_value, normalized_value, reasoning_excerpt, explanation üretir
   └─> DocETL MapOp kullanılır
   └─> Output: stage_stage_extractor_<session>_extraction.json

4. DocETL Unnest
   └─> Çıkarılan alanlar düzleştirilir (her alan bir satır)
   └─> Intermediate: docetl_intermediate/unnest/...

5. DocETL Resolve
   └─> Aynı (patient_id, field_name) için çakışan değerler çözülür
   └─> En güvenilir kaynak seçilir
   └─> Intermediate: docetl_intermediate/resolve/...

6. Consolidator (DocETL Reduce)
   └─> Çözümlenmiş alanlar hasta seviyesinde birleştirilir
   └─> Consolidated reasoning üretilir
   └─> Output: stage_stage_consolidator_<session>_consolidation.json
```

### LLM Prompt Yapısı ve Dosya Konumları

Projede **4 farklı LLM prompt'u** kullanılır ve her biri farklı dosyalarda tanımlanmıştır:

#### 1. **MapOp Prompt (Extractor) - En Detaylı Prompt**

**Dosya**: `src/pipeline/docetl_runner.py`  
**Metod**: `_build_map_operation()`  
**Satır Aralığı**: 604-826 (222 satır)  
**Template Değişkeni**: `self.map_prompt_template`

**İçerik**:
- 800+ satırlık detaylı prompt
- Confidence score calibration protocol (5-step)
- Field-specific extraction rules (her NAACCR alanı için)
- NAACCR/ICD-O-3 format talimatları
- Multiple cancer handling
- Verbatim quote requirements
- Jinja2 template variables: `{{ input.patient_id }}`, `{{ input.doc_id }}`, vb.

**Kod Örneği**:
```python
map_prompt = textwrap.dedent(f"""
    You are a certified oncology registrar. Extract every NAACCR field...
    ### Ontology Guidance
    {instructions}
    ### Document Metadata
    - Patient ID: {{{{ input.patient_id }}}}
    - Document ID: {{{{ input.doc_id }}}}
    ...
""")
self.map_prompt_template = map_prompt
```

**Log Dosyası**: `logs/<session_id>/stage_extractor_prompts.log`  
**Logging Metodu**: `_log_map_prompts()` (satır 2040-2078)

#### 2. **ResolveOp Prompts (Consolidator) - İki Ayrı Prompt**

**Dosya**: `src/pipeline/docetl_runner.py`  
**Metod**: `_build_resolve_operation()`  
**Satır Aralığı**: 1036-1101

**a) Comparison Prompt** (Satır 1036-1055):
- İki değeri karşılaştırır
- `is_match` boolean döner
- Jinja2 template: `{{ input1.field_name }}`, `{{ input2.field_name }}`

**b) Resolution Prompt** (Satır 1057-1101):
- Çakışan değerleri çözer
- Confidence score recalculation
- Source priority rules
- Jinja2 template: `{{ inputs[0].patient_id }}`, `{% for item in inputs %}`

**Kod Örneği**:
```python
comparison_prompt = textwrap.dedent("""
    You are comparing two candidate values...
    Field 1 ({{ input1.field_name }}) from {{ input1.doc_id }}:
    ...
""")

resolution_prompt = textwrap.dedent("""
    You are consolidating oncology registry evidence...
    Evidence set:
    {% for item in inputs %}
    - Document {{ item.doc_id }}...
    {% endfor %}
    ...
""")
```

**Log Dosyası**: `logs/<session_id>/stage_consolidator_prompts.log`  
**Logging Metodu**: `_log_reduce_prompts()` (satır 2080-2117) - Not: Resolve prompts reduce ile birlikte loglanır

#### 3. **ReduceOp Prompt (Consolidator) - Patient-Level Summary**

**Dosya**: `src/pipeline/docetl_runner.py`  
**Metod**: `_build_reduce_operation()`  
**Satır Aralığı**: 1194-1283 (89 satır)  
**Template Değişkeni**: `self.reduce_prompt_template`

**İçerik**:
- Patient-level consolidation instructions
- Confidence score calculation rules
- JSON structure requirements
- Jinja2 template: `{{ reduce_key }}`, `{% for item in inputs %}`

**Kod Örneği**:
```python
prompt = textwrap.dedent("""
    You are a certified oncology registrar consolidating patient-level...
    TASK: Generate a patient-level cancer registry row for patient {{ reduce_key }}
    RESOLVED FIELD EXTRACTIONS FOR PATIENT {{ reduce_key }}:
    {% for item in inputs %}
    Field: {{ item.field_name }}...
    {% endfor %}
    ...
""")
self.reduce_prompt_template = prompt
```

**Log Dosyası**: `logs/<session_id>/stage_consolidator_prompts.log`  
**Logging Metodu**: `_log_reduce_prompts()` (satır 2080-2117)

#### 4. **Tagger Prompt (Document Classification)**

**Dosya**: `src/pipeline/tagger.py`  
**Metod**: `_build_confidence_prompt()`  
**Satır Aralığı**: 235-274 (39 satır)

**İçerik**:
- Document type confidence assessment
- Date confidence assessment
- JSON response format requirements
- Content preview (ilk 500 karakter)

**Kod Örneği**:
```python
def _build_confidence_prompt(self, document: DocumentMetadata) -> str:
    content_preview = document.content[:500] if document.content else ""
    return f"""Analyze the following medical document metadata...
    Document ID: {document.doc_id}
    Document Type: {document.doc_type or "unknown"}
    Content Preview (first 500 chars):
    {content_preview}
    ...
    """
```

**System Prompt**: `"You are a medical document classification expert..."` (satır 146)

**Log Dosyası**: `logs/<session_id>/stage_tagger_prompts.log`  
**Logging Metodu**: `_log_prompt()` (satır 310-334)

### Prompt Logging Mekanizması

Tüm prompt'lar otomatik olarak loglanır:

**1. Map Prompts Logging:**
- **Metod**: `_log_map_prompts()` (`docetl_runner.py:2040-2078`)
- **Format**: Jinja2 template render edilir, her doküman için ayrı entry
- **Dosya**: `logs/<session_id>/stage_extractor_prompts.log`
- **Format**: `--- doc_id=... ---\n{prompt_text}`

**2. Reduce Prompts Logging:**
- **Metod**: `_log_reduce_prompts()` (`docetl_runner.py:2080-2117`)
- **Format**: Jinja2 template render edilir, her hasta için ayrı entry
- **Dosya**: `logs/<session_id>/stage_consolidator_prompts.log`
- **Format**: `--- patient_id=... ---\n{prompt_text}`

**3. Tagger Prompts Logging:**
- **Metod**: `_log_prompt()` (`tagger.py:310-334`)
- **Format**: System prompt + user prompt + response
- **Dosya**: `logs/<session_id>/stage_tagger_prompts.log`
- **Format**: Timestamp + document ID + prompts + response

### Prompt Template Rendering

Tüm prompt'lar **Jinja2** template engine kullanarak render edilir:

**Template Variables:**
- MapOp: `{{ input.patient_id }}`, `{{ input.doc_id }}`, `{{ input.content }}`
- ResolveOp: `{{ input1.field_name }}`, `{{ inputs[0].patient_id }}`
- ReduceOp: `{{ reduce_key }}`, `{% for item in inputs %}`

**Rendering:**
```python
from jinja2 import Environment, StrictUndefined
env = Environment(undefined=StrictUndefined)
template = env.from_string(self.map_prompt_template)
prompt_text = template.render(input=record)
```

### Prompt Özellikleri Özeti

| Prompt | Dosya | Satır | Uzunluk | Template Engine | Log Dosyası |
|--------|-------|-------|---------|-----------------|-------------|
| **MapOp** | `docetl_runner.py` | 604-826 | ~222 satır | Jinja2 | `stage_extractor_prompts.log` |
| **Comparison** | `docetl_runner.py` | 1036-1055 | ~19 satır | Jinja2 | `stage_consolidator_prompts.log` |
| **Resolution** | `docetl_runner.py` | 1057-1101 | ~44 satır | Jinja2 | `stage_consolidator_prompts.log` |
| **ReduceOp** | `docetl_runner.py` | 1194-1283 | ~89 satır | Jinja2 | `stage_consolidator_prompts.log` |
| **Tagger** | `tagger.py` | 235-274 | ~39 satır | f-string | `stage_tagger_prompts.log` |

### Prompt İçerik Özeti

**Extractor (MapOp) Prompt İçeriği:**
- Ontoloji dosyasından tüm alan tanımları
- Her alan için talimatlar (instructions)
- Confidence score calibration protocol (5-step)
- Field-specific extraction rules
- Örnek format
- Validation kuralları

**Consolidator (ResolveOp + ReduceOp) Prompt İçeriği:**
- Çakışan değerler listesi
- Her değerin kaynağı (doc_id, timestamp)
- Çözümleme kuralları
- Öncelik sırası (pathology > clinical > radiology)
- Confidence score recalculation rules
- Patient-level summary generation

**Tagger Prompt İçeriği:**
- Document type confidence assessment
- Date confidence assessment
- Content preview analysis
- JSON response format

### FastAPI Yapısı ve İstek Akışı

#### FastAPI Uygulama Mimarisi

**Ana Dosyalar:**
- `src/main.py`: FastAPI app tanımı, middleware, router registration
- `src/api/routes.py`: Tüm endpoint tanımları ve business logic
- `src/models/schemas.py`: Pydantic request/response modelleri

**App Initialization:**
```python
# src/main.py
app = FastAPI(title="Data Curation Service", ...)
app.add_middleware(CORSMiddleware, ...)
app.include_router(router, prefix="/api/v1")
```

**Router Structure:**
```python
# src/api/routes.py
router = APIRouter()
@router.post("/process")
@router.get("/status/{session_id}")
@router.post("/upload")
```

#### İstek Akış Diyagramı

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                           │
│         POST /api/v1/process                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Route Handler                           │
│         (routes.py: process_documents)                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Request Validation (Pydantic)                  │    │
│  │ 2. Session ID Generation (uuid.uuid4())           │    │
│  │ 3. Document Loading (load_documents)              │    │
│  │ 4. Background Task Creation                       │    │
│  │ 5. Immediate Response Return                      │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Background Task (Async)                        │
│         (process_documents_task)                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Tagger.tag_documents()                          │    │
│  │ 2. Extractor.extract() → DocETL Pipeline          │    │
│  │ 3. Consolidator.consolidate()                      │    │
│  │ 4. Status Update (processing_status[session_id])   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

#### Endpoint Detayları

**1. POST `/api/v1/process`**
- **Handler**: `routes.py:29-66`
- **Request Body**: `ProcessingRequest` (Pydantic model)
- **Response**: `ProcessingResponse` (session_id, status)
- **Background Task**: `process_documents_task` (routes.py:177-265)
- **Async Operations**:
  - `load_documents()`: Doküman dosyalarını okur
  - `parse_document_file()`: Her dosyayı parse eder (parallel)
  - `tagger.tag_documents()`: Dokümanları sıralar
  - `extractor.extract()`: DocETL pipeline çalıştırır
  - `consolidator.consolidate()`: Hasta seviyesi birleştirme

**2. GET `/api/v1/status/{session_id}`**
- **Handler**: `routes.py:69-89`
- **Path Parameter**: `session_id` (string)
- **Response**: `ProcessingResponse` (status + results)
- **Storage Operations**:
  - `storage.load_tagger(session_id)`
  - `storage.load_extraction(session_id)`
  - `storage.load_consolidation(session_id)`
- **In-Memory Status**: `processing_status[session_id]` dict'inden okunur

**3. POST `/api/v1/upload`**
- **Handler**: `routes.py:92-111`
- **Request**: `multipart/form-data` (files)
- **Response**: `{"message": "...", "files": [...]}`
- **File Operations**:
  - `await file.read()`: Async file reading
  - `data/input/` klasörüne yazma
  - Uploaded file paths listesi döner

#### FastAPI Özellikleri

**1. Async/Await:**
- Tüm endpoint'ler `async def`
- `aiofiles` ile async file I/O
- `asyncio.gather()` ile paralel işlemler

**2. Background Tasks:**
- `BackgroundTasks.add_task()` ile non-blocking işlemler
- API hemen response döner
- Status polling ile ilerleme takibi

**3. Pydantic Validation:**
- Request/Response otomatik validate edilir
- Type-safe API garantisi
- Error messages otomatik oluşturulur

**4. CORS Middleware:**
- Development: `allow_origins=["*"]`
- Production'da kısıtlanmalı

**5. Error Handling:**
- `HTTPException` ile standart hata yanıtları
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

#### Server Configuration

**Settings (config/settings.py):**
- `api_host: "0.0.0.0"` (tüm interface'ler)
- `api_port: 8000` (default)
- `api_reload: True` (development mode)

**Uvicorn Options:**
- `--reload`: Auto-reload on code changes
- `--workers`: Production'da multiple workers
- `--host`, `--port`: Override settings

### Concurrency Modeli

```
FastAPI Request
    │
    ├─> Background Task (async)
    │       │
    │       ├─> Tagger (parallel document processing)
    │       │
    │       ├─> Extractor (semaphore: 30 concurrent LLM calls)
    │       │       │
    │       │       └─> DocETL MapOp (200 threads)
    │       │
    │       └─> Consolidator (sequential per patient)
    │               │
    │               └─> DocETL ResolveOp + ReduceOp
```

### DocETL Kullanılan Özellikler ve Detaylar

#### 1. **Memory-Based Dataset**
- **Özellik**: DocETL'in `Dataset` sınıfı `type="memory"` modu
- **Kullanım**: Python dict listesi direkt olarak dataset'e verilir
- **Avantaj**: Geçici dosya oluşturmadan işlem yapılır
- **Kod**: `Dataset(type="memory", path=raw_records)`

#### 2. **Structured Output Mode**
- **Özellik**: LLM çıktılarının JSON schema ile validate edilmesi
- **Kullanım**: MapOp ve ReduceOp'da `output_mode="structured_output"`
- **Avantaj**: LLM çıktıları otomatik olarak Pydantic şemalarına uygun olur
- **Validation**: DocETL otomatik olarak şema uyumsuzluğunda retry yapar

#### 3. **Custom Prompt Templates**
- **MapOp Prompt**: 800+ satırlık detaylı prompt
  - Confidence score calibration protocol (5-step)
  - Field-specific extraction rules
  - NAACCR/ICD-O-3 format talimatları
  - Multiple cancer handling
  - Verbatim quote requirements
- **ReduceOp Prompt**: Consolidation reasoning için özel prompt
  - Conflict resolution rules
  - Source priority (pathology > clinical > radiology)
  - Timeline construction

#### 4. **JSON Truncation Handling**
- **Problem**: LLM çıktıları bazen kesilir (token limit)
- **Çözüm**: `_sanitize_json_payload()` fonksiyonu
  - Eksik closing brackets/quotes tamamlanır
  - Markdown code fence'ler temizlenir
  - Balanced pair kontrolü
- **Kod Yolu**: `src/pipeline/docetl_runner.py:183-223`

#### 5. **Checkpoint ve Resume Mekanizması**
- **Özellik**: Her operatör çıktısı ayrı dosyaya kaydedilir
- **Dosya Yapısı**:
  ```
  docetl_intermediate/
  └── clinical_registry/
      ├── extract_clinical_fields.json (Map output)
      ├── explode_field_records.json (Unnest output)
      ├── resolve_patient_fields.json (Resolve output)
      └── reduce_patient_summary.json (Reduce output)
  ```
- **Resume Logic**: Pipeline hata verirse, mevcut checkpoint'lerden devam eder
- **Kod**: `docetl_runner.py:369-400` (checkpoint loading)

#### 6. **Model-Specific Optimizations**
- **Qwen Modelleri**:
  - Timeout: 20s (OpenAI: 40s)
  - Retry: 1 (OpenAI: 3)
  - `reasoning_effort="none"` (hız için)
  - `max_tokens: 10000` (JSON truncation önleme)
  - `top_p: 0.95`, `temperature: 0.1`
- **OpenAI Modelleri**:
  - Timeout: 40s
  - Retry: 3
  - `max_tokens: 8000`
- **Kod**: `docetl_runner.py:833-899`

#### 7. **Error Handling ve Retry**
- **Pipeline Level Retry**:
  - Transient hatalar için otomatik retry
  - Exponential backoff (3-10 saniye)
  - Partial checkpoint'lerden resume
- **Retryable Errors**:
  - `LiteLLMAPIError`
  - `httpx.ConnectError`, `httpx.ReadTimeout`
  - `httpcore.ConnectError`
  - HTTP 500, 502, 503, 504
  - "Bad Gateway", "Service Unavailable"
- **Kod**: `docetl_runner.py:492-600`

#### 8. **Thread Management**
- **Dinamik Thread Hesaplama**:
  ```python
  base_threads = max(len(documents), 1)
  cpu_based_threads = cpu_count * 5  # I/O-bound için
  settings_based_threads = max(max_workers, max_concurrent_requests * 3)
  desired_threads = min(docetl_max_threads, max(base, cpu, settings))
  ```
- **Varsayılan**: 200 thread (ayarlanabilir)
- **Kod**: `docetl_runner.py:404-446`

#### 9. **Blocking Keys (ResolveOp)**
- **Kullanım**: Aynı `(patient_id, field_name)` için çakışan değerleri çözer
- **Blocking Keys**: `["patient_id", "field_name"]`
- **Çözümleme Stratejisi**:
  - En yüksek confidence score
  - En yeni doküman tarihi
  - Source quality (pathology > clinical)
- **Kod**: `docetl_runner.py:1035-1158`

#### 10. **Validation Rules**
- **MapOp Validation**:
  ```python
  validation_rules = [
      "isinstance(output, dict) or isinstance(output, list)",
  ]
  ```
- **Schema Validation**: Pydantic v2 şemaları ile otomatik
- **Retry**: Validation başarısız olursa prompt tekrar gönderilir
- **Max Retries**: 2 (Qwen: 1)

#### 11. **Prompt Logging**
- **Özellik**: Her LLM çağrısının prompt'u loglanır
- **Dosyalar**:
  - `logs/<session_id>/stage_extractor_prompts.log`
  - `logs/<session_id>/stage_consolidator_prompts.log`
- **Format**: Timestamp + prompt content + response
- **Kod**: `docetl_runner.py:362, 480`

#### 12. **Normalize Operation**
- **Görev**: Map çıktısını normalize eder (list formatına çevirir)
- **Kullanım**: Bazı LLM'ler dict, bazıları list döner
- **Kod**: `docetl_runner.py:900-1008`

#### 13. **Confidence Score Calibration**
- **5-Step Protocol**:
  1. Evidence Classification (Explicit/Interpreted/Inferred/Absence)
  2. Source Quality Assessment (pathology > clinical)
  3. Consistency Check (conflicts varsa -0.08)
  4. Numeric Score Mapping (0.30-0.98 arası)
  5. Sanity Clamp (hard caps: Not Reported ≤0.40, inferred ≤0.82)
- **Hard Caps**:
  - "Not Reported": Maximum 0.40
  - `inferred=true`: Maximum 0.82
  - Calculated/interpreted: Maximum 0.94
  - Verbatim: Maximum 0.98 (neredeyse hiç 1.0)
- **Kod**: Map prompt içinde (800+ satır)

#### 14. **Pipeline Output Structure**
- **Intermediate Outputs**: Her operatör çıktısı ayrı dosyada
- **Final Output**: `docetl_patient_results.json`
- **Format**: JSON array of patient records
- **Kod**: `docetl_runner.py:354-360`

#### 15. **CodeMapOp Fallback**
- **Özellik**: `CodeMapOp` sınıfı yoksa dict-based operation kullanılır
- **Fallback**: `_DictOperation` shim class
- **Kod**: `docetl_runner.py:63-71, 994-1008`

#### 16. **Jinja2 Template Rendering**
- **Özellik**: Prompt'larda `{{ input.patient_id }}` gibi template variables
- **Kullanım**: DocETL otomatik olarak render eder
- **Örnek**: `"Patient ID: {{ input.patient_id }}"` → `"Patient ID: p01"`

#### 17. **Parallel Patient Processing**
- **Özellik**: Her hasta için ayrı DocETL pipeline çalıştırılır
- **Concurrency**: Semaphore ile sınırlandırılmış (varsayılan 12 hasta)
- **Session ID**: `{session_id}__{patient_id}` formatı
- **Kod**: `extractor.py:121-206`

#### 18. **Model Failover**
- **Özellik**: Bir model başarısız olursa otomatik fallback
- **Failover Order**:
  1. Seçilen model
  2. OpenAI (varsayılan)
  3. Qwen (fallback)
- **Kod**: `extractor.py:300-328`

#### 19. **Response Parsing**
- **Özellik**: LLM response'ları farklı formatlarda gelebilir
- **Handling**:
  - Markdown code blocks temizlenir
  - JSON string'ler parse edilir
  - Tool call responses işlenir
- **Kod**: `docetl_runner.py:225-246`

#### 20. **Pipeline Step Definition**
- **Özellik**: Pipeline'ın adım adım çalışması
- **Step**: `PipelineStep(name="clinical_registry", input="clinical_docs", operations=[...])`
- **Operations Order**: Map → Normalize → Unnest → Resolve → Reduce
- **Kod**: `docetl_runner.py:343-360`

---

## 💡 Mülakat İpuçları

### 1. **Projeyi Özetleyin**
- 3-5 cümle ile projenin amacını açıklayın
- DocETL'in rolünü vurgulayın
- NAACCR standartlarına uyumdan bahsedin

### 2. **Teknik Derinlik Gösterin**
- DocETL operatörlerini (Map, Unnest, Resolve, Reduce) açıklayın
- Concurrency optimizasyonlarından bahsedin
- Error handling ve retry mekanizmalarını anlatın

### 3. **Zorlukları ve Çözümleri Anlatın**
- LLM API rate limiting ile nasıl başa çıktınız?
- Schema validation retry mekanizması neden gerekli?
- Memory kullanımını nasıl optimize ettiniz?

### 4. **İyileştirme Önerileri**
- Production'da Redis/database kullanımı
- S3 gibi object storage entegrasyonu
- Batch processing için streaming
- Caching mekanizmaları

### 5. **Kod Örnekleri Hazırlayın**
- DocETL pipeline tanımı
- LLM provider wrapper
- Error handling pattern'leri

### 6. **Dosya Yollarını Biliyor Olun**
- `cancer_registry_fields.yaml` nerede?
- Log'lar nereye yazılıyor?
- Output'lar nasıl organize ediliyor?
- **LLM prompt'ları hangi dosyalarda?**
  - MapOp prompt: `src/pipeline/docetl_runner.py:604-826`
  - ResolveOp prompts: `src/pipeline/docetl_runner.py:1036-1101`
  - ReduceOp prompt: `src/pipeline/docetl_runner.py:1194-1283`
  - Tagger prompt: `src/pipeline/tagger.py:235-274`

### 7. **Ontoloji Dosyasını İnceleyin**
- Hangi alanlar var?
- NAACCR standartları neler?
- ICD-O-3 ve AJCC kodlamaları nasıl çalışıyor?

### 8. **API Kullanımını Gösterin**
- curl komutları hazırlayın
- Response formatlarını açıklayın
- Error case'leri anlatın

### 9. **Docker ve Deployment**
- Dockerfile'ı açıklayın
- docker-compose.yml yapısını anlatın
- Environment variable yönetimini gösterin

### 10. **Testing ve Validation**
- E2E demo script'i nasıl çalışıyor?
- Output validation nasıl yapılıyor?
- Log analizi nasıl yapılır?

---

## 🏥 NAACCR Standartları ve Kanser Kayıt Sistemi

### NAACCR Nedir?

**NAACCR (North American Association of Central Cancer Registries)**, Kuzey Amerika'daki kanser kayıt sistemlerini standartlaştıran ve koordine eden bir organizasyondur. Kanser verilerinin toplanması, saklanması ve analiz edilmesi için standartlar belirler.

### NAACCR'ın Amacı

1. **Veri Standardizasyonu**: Tüm kanser kayıt sistemlerinin aynı formatı kullanmasını sağlar
2. **Veri Kalitesi**: Tutarlı ve güvenilir kanser verileri toplama
3. **Araştırma Desteği**: Epidemiyolojik ve klinik araştırmalar için standart veri seti
4. **Halk Sağlığı**: Kanser insidansı, prevalansı ve mortalite trendlerini izleme

### Projede Kullanılan NAACCR Alanları

Projede `cancer_registry_fields.yaml` dosyasında **12 temel NAACCR alanı** tanımlanmıştır:

#### 1. **Diagnosis Domain (Tanı Alanları)**

**naaccr_diagnosis_dt (NAACCR Item #390 - Date of Diagnosis)**
- **Açıklama**: Kanser tanısının konulduğu tarih
- **Format**: YYYY-MM-DD (ISO 8601)
- **Öncelik**: Pathological confirmation date > Clinical diagnosis date
- **Kaynak**: Pathology reports, biopsy results, diagnosis statements
- **Örnek**: "2015-07-15" (July 2015'te tanı konulmuş)

**ca_site (Cancer Site - Anatomical Site)**
- **Açıklama**: Kanserin anatomik yeri (ICD-O-3 Topography Code)
- **Format**: "Site Name (Code)/Behavior"
- **Örnekler**:
  - "Prostate (C61)/Malignant"
  - "Colon, Sigmoid (C18.7)/Malignant"
  - "Breast (C50.9)/Malignant"
- **Behavior Codes**:
  - `/0` = Benign
  - `/1` = Uncertain
  - `/2` = In situ
  - `/3` = Malignant primary (en yaygın)
  - `/6` = Metastatic

**naaccr_histology_cd (ICD-O-3 Histology/Morphology Code)**
- **Açıklama**: Kanserin histolojik tipi (morphology code)
- **Format**: "XXXX/Y - Description"
- **Örnekler**:
  - "8140/3 - Adenocarcinoma, NOS"
  - "8480/3 - Mucinous adenocarcinoma"
  - "8500/3 - Infiltrating duct carcinoma, NOS"
- **Kod Yapısı**:
  - `XXXX` = Morphology code (4 haneli)
  - `Y` = Behavior code (0, 1, 2, 3, 6)
- **Mapping**: Histoloji terimleri ICD-O-3 kodlarına map edilir

#### 2. **Clinical Staging Domain (Klinik Evreleme)**

**AJCC TNM Staging System** kullanılır (American Joint Committee on Cancer):

**ca_clinical_t_stage (Clinical T Stage)**
- **Açıklama**: Tümörün boyutu/yayılımı (klinik değerlendirme)
- **Valid Değerler**: cT0, cTis, cTa, cT1, cT1a, cT1b, cT2, cT2a, cT2b, cT3, cT3a, cT3b, cT4, cT4a, cT4b, cTx
- **Kaynak**: Clinical notes, imaging reports, physical examination
- **Prefix**: `c` = clinical (klinik)
- **Örnek**: "cT2" (klinik olarak T2 evresi)

**ca_clinical_n_stage (Clinical N Stage)**
- **Açıklama**: Lenf nodu tutulumu (klinik değerlendirme)
- **Valid Değerler**: cN0, cN1, cN1a, cN1b, cN1c, cN2, cN2a, cN2b, cN2c, cN3, cN3a, cN3b, cN3c, cNx
- **Kaynak**: Clinical examination, imaging studies, palpation
- **Örnek**: "cN0" (klinik olarak lenf nodu tutulumu yok)

**ca_clinical_m_stage (Clinical M Stage)**
- **Açıklama**: Uzak metastaz (klinik değerlendirme)
- **Valid Değerler**: cM0, cM1, cM1a, cM1b, cM1c, cMx
- **Kaynak**: Imaging studies, clinical assessment, symptoms
- **Örnek**: "cM0" (klinik olarak metastaz yok)

#### 3. **Pathological Staging Domain (Patolojik Evreleme)**

**ca_path_t_stage (Pathological T Stage)**
- **Açıklama**: Tümörün boyutu/yayılımı (cerrahi patoloji)
- **Valid Değerler**: pT0, pTis, pTa, pT1, pT1a, pT1b, pT2, pT2a, pT2b, pT3, pT3a, pT3b, pT4, pT4a, pT4b, pTx
- **Kaynak**: Surgical pathology reports, resection specimens, biopsy reports
- **Prefix**: `p` = pathological (patolojik) - **ZORUNLU**
- **Öncelik**: Pathological staging > Clinical staging (daha güvenilir)
- **Örnek**: "pT3" (patolojik olarak T3 evresi)

**ca_path_n_stage (Pathological N Stage)**
- **Açıklama**: Lenf nodu tutulumu (cerrahi patoloji)
- **Valid Değerler**: pN0, pN1, pN1a, pN1b, pN1c, pN2, pN2a, pN2b, pN2c, pN3, pN3a, pN3b, pN3c, pNx
- **Kaynak**: Lymph node dissection results, pathology reports
- **Örnek**: "pN1" (patolojik olarak 1 lenf nodu pozitif)

**ca_path_m_stage (Pathological M Stage)**
- **Açıklama**: Uzak metastaz (patolojik doğrulama)
- **Valid Değerler**: pM0, pM1, pM1a, pM1b, pM1c, pMx
- **Kaynak**: Pathological confirmation of distant metastases
- **Örnek**: "pM1" (patolojik olarak metastaz var)

#### 4. **Summary Staging Domain (Özet Evreleme)**

**ca_gen_sum_stage_2 (SEER Summary Stage 2000)**
- **Açıklama**: TNM staging'den türetilen özet evre
- **Valid Değerler**:
  - `0` = In situ (yerinde)
  - `1` = Localized (lokalize)
  - `2` = Regional by direct extension (direkt yayılım)
  - `3` = Regional lymph nodes involved (bölgesel lenf nodları)
  - `4` = Distant site/lymph nodes (uzak metastaz)
  - `7` = Distant site/lymph nodes, unknown if extension or mets
  - `9` = Unstaged/Unknown (evrelendirilememiş)
- **Türetme**: TNM değerlerinden otomatik hesaplanır
- **Örnek**: "1" (lokalize kanser)

#### 5. **Performance Status Domain (Performans Durumu)**

**ecog (ECOG Performance Status)**
- **Açıklama**: Hastanın fonksiyonel durumu (0-5)
- **Valid Değerler**:
  - `0` = Fully active (tam aktif)
  - `1` = Restricted in strenuous activity (ağır aktivitede kısıtlı)
  - `2` = Ambulatory, unable to work (yürüyebilir, çalışamaz)
  - `3` = Limited self-care, >50% in bed (sınırlı öz bakım, %50+ yatakta)
  - `4` = Completely disabled, totally confined (tamamen yatalak)
  - `5` = Dead (ölü)
- **KPS Mapping**: KPS varsa ECOG'ye çevrilebilir:
  - KPS 90-100 → ECOG 0
  - KPS 70-80 → ECOG 1
  - KPS 50-60 → ECOG 2
  - KPS 30-40 → ECOG 3
  - KPS ≤20 → ECOG 4

**kps (Karnofsky Performance Score)**
- **Açıklama**: Hastanın performans skoru (0-100, 10'ar artışlarla)
- **Valid Değerler**: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- **Anlamlar**:
  - `100` = Normal (normal)
  - `90` = Minor symptoms (hafif semptomlar)
  - `80` = Some symptoms (bazı semptomlar)
  - `70` = Cares for self (kendine bakabilir)
  - `60` = Occasional assistance (ara sıra yardım)
  - `50` = Considerable assistance (önemli yardım)
  - `40` = Disabled (engelli)
  - `30` = Severely disabled (ağır engelli)
  - `20` = Very sick (çok hasta)
  - `10` = Moribund (ölüm döşeğinde)
  - `0` = Dead (ölü)
- **ECOG Mapping**: ECOG varsa KPS'ye çevrilebilir (tersine)

### ICD-O-3 Kodlama Sistemi

**ICD-O-3 (International Classification of Diseases for Oncology, 3rd Edition)**, WHO tarafından geliştirilmiş kanser kodlama sistemidir.

#### Topography Codes (C Kodları)
- **Format**: CXX.X (ör: C61.9)
- **Örnekler**:
  - C61.9 = Prostate
  - C50.9 = Breast
  - C18.7 = Colon, Sigmoid
  - C20.9 = Rectum
- **Kullanım**: `ca_site` alanında kullanılır

#### Morphology Codes (Histology Codes)
- **Format**: XXXX/Y (ör: 8140/3)
- **Örnekler**:
  - 8140/3 = Adenocarcinoma, NOS
  - 8480/3 = Mucinous adenocarcinoma
  - 8500/3 = Infiltrating duct carcinoma
- **Kullanım**: `naaccr_histology_cd` alanında kullanılır

### AJCC TNM Staging System

**AJCC (American Joint Committee on Cancer)** tarafından geliştirilmiş evreleme sistemi.

#### TNM Kısaltmaları
- **T (Tumor)**: Tümörün boyutu ve yayılımı
- **N (Nodes)**: Lenf nodu tutulumu
- **M (Metastasis)**: Uzak metastaz varlığı

#### Prefix'ler
- **c** = Clinical (klinik değerlendirme)
- **p** = Pathological (patolojik değerlendirme)
- **y** = Post-treatment (tedavi sonrası)
- **r** = Recurrent (tekrarlayan)

#### Stage Grouping
TNM değerleri birleştirilerek Stage Group oluşturulur:
- **Stage 0**: Tis, N0, M0
- **Stage I**: T1-T2, N0, M0
- **Stage II**: T2-T3, N0, M0
- **Stage III**: T1-T4, N1-N3, M0
- **Stage IV**: Herhangi bir T, Herhangi bir N, M1

### SEER Summary Stage 2000

**SEER (Surveillance, Epidemiology, and End Results)** programı tarafından geliştirilmiş özet evreleme sistemi.

#### Evre Kategorileri
1. **In Situ (0)**: Kanser sadece başladığı yerde
2. **Localized (1)**: Kanser sadece orijinal organında
3. **Regional (2-3)**: Kanser yakın dokulara/lenf nodlarına yayılmış
4. **Distant (4)**: Kanser uzak organlara metastaz yapmış
5. **Unstaged (9)**: Yeterli bilgi yok

### Projede NAACCR Uyumluluğu

#### 1. **Ontoloji Dosyası**
- `cancer_registry_fields.yaml` dosyası NAACCR standartlarına uygun
- Her alan için NAACCR Item numarası veya vocabulary referansı var
- ICD-O-3 ve AJCC kodlama standartları takip ediliyor

#### 2. **Extraction Rules**
- LLM prompt'larında NAACCR format talimatları var
- Field-specific extraction rules NAACCR standartlarına göre
- Date format: YYYY-MM-DD (ISO 8601)
- Code format: ICD-O-3 ve AJCC standartlarına uygun

#### 3. **Validation**
- Pydantic şemaları NAACCR formatlarını validate eder
- Invalid değerler (ör: "Rx") reddedilir
- Required fields NAACCR core elements'e göre

#### 4. **Consolidation**
- Hasta seviyesi birleştirme NAACCR öncelik kurallarına göre:
  - Pathology > Operative > Clinical > Radiology
  - En yeni tarih öncelikli
  - En yüksek confidence score öncelikli

### NAACCR Data Quality Standards

#### 1. **Completeness (Tamlık)**
- Tüm required fields doldurulmalı
- "Not Reported" sadece gerçekten bilgi yoksa kullanılmalı

#### 2. **Accuracy (Doğruluk)**
- Kodlar standartlara uygun olmalı
- Tarihler doğru formatlanmalı
- Staging değerleri geçerli olmalı

#### 3. **Consistency (Tutarlılık)**
- Aynı hasta için farklı dokümanlarda tutarlı değerler
- Çakışmalar çözülmeli (ResolveOp)

#### 4. **Timeliness (Zamanında)**
- Diagnosis date en erken tarih olmalı
- Staging timeline'ı kronolojik olmalı

### Mülakat İçin NAACCR Bilgileri

#### Sorulabilecek Sorular ve Cevaplar

**S: NAACCR nedir ve neden önemlidir?**
**C:** "NAACCR, Kuzey Amerika'daki kanser kayıt sistemlerini standartlaştıran bir organizasyondur. Projede NAACCR standartlarına uygun 12 temel alan çıkarıyoruz: diagnosis date, cancer site (ICD-O-3), histology code, TNM staging (clinical ve pathological), summary stage, ve performance status (ECOG/KPS). Bu standartlar sayesinde farklı kaynaklardan gelen veriler tutarlı bir formatta toplanabilir ve analiz edilebilir."

**S: ICD-O-3 kodlama sistemini açıklayın.**
**C:** "ICD-O-3, WHO tarafından geliştirilmiş kanser kodlama sistemidir. İki ana bileşeni var: Topography codes (C kodları, anatomik yer) ve Morphology codes (XXXX/Y formatı, histoloji tipi). Örneğin, 'Prostate (C61.9)/Malignant' topografiyi, '8140/3 - Adenocarcinoma' ise morfolojiyi gösterir. Behavior code (/3) malignant primary'i belirtir."

**S: AJCC TNM staging sistemini nasıl kullanıyorsunuz?**
**C:** "AJCC TNM sistemi, T (tümör), N (lenf nodları), M (metastaz) değerlerini kullanır. Projede hem clinical (cT, cN, cM) hem de pathological (pT, pN, pM) staging çıkarıyoruz. Pathological staging daha güvenilir olduğu için consolidation'da öncelik veriyoruz. Prefix'ler zorunlu - 'p' olmadan pathological staging kabul edilmez."

**S: SEER Summary Stage nasıl türetiliyor?**
**C:** "SEER Summary Stage, TNM değerlerinden otomatik olarak türetilir. 0 (In situ), 1 (Localized), 2-3 (Regional), 4 (Distant), 9 (Unstaged) değerlerini alır. LLM, TNM değerlerine bakarak uygun summary stage'i belirler veya dokümanlarda açıkça belirtilmişse onu kullanır."

**S: Performance status alanlarını nasıl işliyorsunuz?**
**C:** "ECOG (0-5) ve KPS (0-100) olmak üzere iki performance status alanı var. Eğer sadece biri varsa, diğerine çevrilebilir. Örneğin, KPS 80 varsa ECOG 1 olarak infer edilir. Bu mapping kuralları ontoloji dosyasında tanımlı ve LLM prompt'larında belirtilmiş."

### NAACCR Kaynakları

- **NAACCR Website**: https://www.naaccr.org/
- **NAACCR Standards**: https://www.naaccr.org/standards-and-data-operations/
- **ICD-O-3**: https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology
- **AJCC Staging**: https://www.cancerstaging.org/
- **SEER Summary Stage**: https://seer.cancer.gov/tools/ssm/

---

## 📚 Ek Kaynaklar

### DocETL Dokümantasyonu
- [DocETL GitHub](https://github.com/ucbepic/docetl)
- [DocETL Documentation](https://ucbepic.github.io/docetl/)
- [DocWrangler Blog](https://data-people-group.github.io/blogs/2025/01/13/docwrangler/)

### NAACCR Standartları
- [NAACCR Website](https://www.naaccr.org/)
- [ICD-O-3](https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology)
- [AJCC Staging](https://www.cancerstaging.org/)

### Teknoloji Dokümantasyonu
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic v2](https://docs.pydantic.dev/)
- [LiteLLM](https://docs.litellm.ai/)
- [OpenRouter](https://openrouter.ai/)

---

## ✅ Son Kontrol Listesi

Mülakattan önce şunları kontrol edin:

- [ ] Proje yapısını ezbere biliyorum
- [ ] Kritik dosya yollarını biliyorum
- [ ] DocETL operatörlerini açıklayabilirim
- [ ] Ontoloji dosyasının içeriğini biliyorum
- [ ] API endpoint'lerini ve kullanımlarını biliyorum
- [ ] Concurrency modelini açıklayabilirim
- [ ] Error handling stratejilerini anlatabilirim
- [ ] Docker setup'ını açıklayabilirim
- [ ] Logging yapısını biliyorum
- [ ] Output formatlarını anlatabilirim
- [ ] **NAACCR standartlarını ve 12 alanı biliyorum**
- [ ] **ICD-O-3 kodlama sistemini açıklayabilirim**
- [ ] **AJCC TNM staging sistemini anlatabilirim**
- [ ] **SEER Summary Stage türetme mantığını biliyorum**
- [ ] **FastAPI endpoint'lerini ve istek akışını biliyorum**
- [ ] **Background tasks ve async/await kullanımını anlatabilirim**
- [ ] **Pydantic validation ve error handling'i açıklayabilirim**
- [ ] **LLM prompt'larının hangi dosyalarda olduğunu biliyorum**
- [ ] **Prompt logging mekanizmasını anlatabilirim**
- [ ] İyileştirme önerileri hazırladım
- [ ] Kod örnekleri hazırladım

---

**Başarılar! 🚀**

*Bu doküman, mülakat öncesi hazırlık için hazırlanmıştır. Proje detaylarını ve kod yapısını gözden geçirmeniz önerilir.*

