#  Medical AI API — 7 Models in One FastAPI

##  Project Structure

```
medical_api/
├── main.py              ← FastAPI app (كل الـ endpoints)
├── model_lung.py        ← PyTorch architecture للـ Lung model
├── requirements.txt     ← كل الـ dependencies
├── README.md
└── weights/            
    ├── ImranIsicModel.keras
    ├── Final Final Breast Cancer Segmentation.h5
    ├── bes__model.h5
    ├── eye_model.keras
    ├── brain_model.h5
    ├── heart_segmentation_model.h5
    ├── final_ChestX6_hybrid_model.pth
    └── kidney_model1.h5
```

---

##  Setup

```bash
# 1. إنشاء virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. تثبيت الـ dependencies
pip install -r requirements.txt

# 3. ضع ملفات الأوزان في مجلد weights/

# 4. تشغيل الـ API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

##  Endpoints

| Method | Endpoint          | Model  | Task                        | Input Size        |
|--------|-------------------|--------|-----------------------------|-------------------|
| GET    | `/`               | —      | API info + loaded models    | —                 |
| GET    | `/health`         | —      | Health check                | —                 |
| POST   | `/predict/skin`   | Skin   | 8-class classification      | 28×28 RGB         |
| POST   | `/predict/breast` | Breast | Seg + 3-class classification| 256/224 RGB       |
| POST   | `/predict/eye`    | Eye    | 4-class classification      | 224×224 RGB       |
| POST   | `/predict/brain`  | Brain  | 4-class classification      | 224×224 RGB       |
| POST   | `/predict/heart`  | Heart  | Segmentation (U-Net)        | 128×128 Grayscale |
| POST   | `/predict/lung`   | Lung   | 6-class classification      | 224×224 RGB       |
| POST   | `/predict/kidney` | Kidney | 4-class classification      | 200×200 Grayscale |

---

## Request Format

كل endpoint بيقبل `multipart/form-data` مع field اسمه `file`:

```bash
# مثال — Skin
curl -X POST "http://localhost:8000/predict/skin" \
     -F "file=@skin_image.jpg"

# مثال — Brain
curl -X POST "http://localhost:8000/predict/brain" \
     -F "file=@mri_scan.jpg"
```

---

##  Response Format

### Classification endpoints (skin / eye / brain / lung / kidney):
```json
{
  "model": "brain",
  "predicted_class": "glioma",
  "confidence": 94.32,
  "all_probabilities": {
    "glioma": 94.32,
    "meningioma": 3.11,
    "notumor": 1.88,
    "pituitary": 0.69
  }
}
```

### Breast (Seg + Classification):
```json
{
  "model": "breast",
  "predicted_class": "malignant",
  "confidence": 87.5,
  "all_probabilities": {
    "benign": 8.2,
    "malignant": 87.5,
    "normal": 4.3
  },
  "segmentation": {
    "mask_mean_activation": 0.3241,
    "lesion_coverage_percent": 18.45
  }
}
```

### Heart (Segmentation):
```json
{
  "model": "heart",
  "task": "segmentation",
  "heart_area_ratio_percent": 12.34,
  "assessment": "normal",
  "mask_stats": {
    "mean_activation": 0.2341,
    "max_activation": 0.9871,
    "detected_pixels": 2031
  }
}
```

### Skin (with severity):
```json
{
  "model": "skin",
  "predicted_class": "MEL",
  "predicted_name": "Melanoma",
  "confidence": 91.2,
  "severity": "high",
  "all_probabilities": { ... }
}
```

---

##  Class Labels

| Model  | Classes |
|--------|---------|
| Skin   | AK, BCC, BKL, DF, MEL, NV, SCC, VASC |
| Breast | benign, malignant, normal |
| Eye    | Cataract, Diabetic Retinopathy, Glaucoma, Normal |
| Brain  | glioma, meningioma, notumor, pituitary |
| Heart  | normal, slightly_large, abnormally_large, not_detected |
| Lung   | Tuberculosis, Pneumonia-Viral, Pneumonia-Bacterial, Normal, Emphysema, Covid-19 |
| Kidney | Cyst, Normal, Stone, Tumor |

---

##  Interactive Docs
بعد تشغيل الـ server:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**       http://localhost:8000/redoc

---

##  Notes

- الـ API بيعمل حتى لو بعض الأوزان مش موجودة — هيرجع خطأ `503` للـ endpoint ده بس
- الـ Lung model بيستخدم **PyTorch** وباقي الموديلز **TensorFlow**
- ملفات الأوزان لازم تكون في مجلد `weights/` جنب `main.py`
