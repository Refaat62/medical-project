"""
Medical AI FastAPI — 7 Models in One API
Models:
  1. Skin Cancer Classification      → /predict/skin
  2. Breast Cancer Seg + Class       → /predict/breast
  3. Eye Disease Classification      → /predict/eye
  4. Brain Tumor Classification      → /predict/brain
  5. Heart CT Segmentation           → /predict/heart
  6. Lung/Chest X-Ray Classification → /predict/lung
  7. Kidney Disease Classification   → /predict/kidney
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
import io, os, logging
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _mask_to_base64(mask: np.ndarray, colormap: str = "plasma") -> str:
    """Convert a 2-D float mask → PNG base64 string."""
    import io
    norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    cmap = cm.get_cmap(colormap)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    img  = Image.fromarray(rgba)
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _overlay_to_base64(orig: Image.Image, mask: np.ndarray,
                        size: tuple, alpha: float = 0.45) -> str:
    """Blend original image with coloured mask → PNG base64 string."""
    import io
    img_rgb   = np.array(orig.convert("RGB").resize(size))
    norm      = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    cmap      = cm.get_cmap("plasma")
    colored   = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    overlay   = (img_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
    buf       = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model store

MODELS: dict = {}


def _build_brain_cnn():
    """VGG-like CNN — same architecture used during brain tumor training."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64,  activation="relu"),
        tf.keras.layers.Dense(4,   activation="softmax"),
    ])
    return model


def _build_kidney_cnn():
    """CNN architecture used during kidney disease training."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(200,200,1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(4,   activation="softmax"),
    ])
    return model


HF_REPO = "morefaat69/medical-ai-models"
WEIGHTS_DIR = "weights"

def _download_weight(filename: str) -> str:
    """Download a model file from Hugging Face if not already cached locally."""
    from huggingface_hub import hf_hub_download
    local_path = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(local_path):
        logger.info(f"  Downloading {filename} from Hugging Face...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=WEIGHTS_DIR,
        )
        logger.info(f"✅ {filename} downloaded")
        return downloaded
    logger.info(f" {filename} already cached locally")
    return local_path


def load_all_models():
    """Download weights from Hugging Face (if needed) then load all models."""
    import tensorflow as tf

    # 1 ─ Skin
    try:
        path = _download_weight("ImranIsicModel.keras")
        MODELS["skin"] = tf.keras.models.load_model(path)
        logger.info("✅ Skin model loaded")
    except Exception as e:
        logger.warning(f"  Skin model failed: {e}")

    # 2 ─ Breast (two sub-models)
    try:
        seg_path = _download_weight("Final Final Breast Cancer Segmentation.h5")
        cls_path  = _download_weight("bes__model.h5")
        MODELS["breast_seg"] = tf.keras.models.load_model(seg_path)
        MODELS["breast_cls"] = tf.keras.models.load_model(cls_path)
        logger.info("✅ Breast models loaded")
    except Exception as e:
        logger.warning(f"  Breast models failed: {e}")

    # 3 ─ Eye
    try:
        path = _download_weight("eye_model.keras")
        MODELS["eye"] = tf.keras.models.load_model(path)
        logger.info("✅ Eye model loaded")
    except Exception as e:
        logger.warning(f"  Eye model failed: {e}")

    # 4 ─ Brain
    try:
        path = _download_weight("brain_model.h5")
        try:
            MODELS["brain"] = tf.keras.models.load_model(path, compile=False)
        except TypeError:
            brain_model = _build_brain_cnn()
            brain_model.load_weights(path)
            MODELS["brain"] = brain_model
        logger.info("✅ Brain model loaded")
    except Exception as e:
        logger.warning(f"  Brain model failed: {e}")

    # 5 ─ Heart
    try:
        path = _download_weight("heart_segmentation_model.h5")
        MODELS["heart"] = _build_unet()
        MODELS["heart"].load_weights(path)
        logger.info("✅ Heart model loaded")
    except Exception as e:
        logger.warning(f"  Heart model failed: {e}")

    # 6 ─ Lung (PyTorch)
    try:
        import torch
        from model_lung import HybridModel
        path = _download_weight("final_ChestX6_hybrid_model.pth")
        m = HybridModel(num_classes=6)
        state = torch.load(path, map_location="cpu")
        clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        m.load_state_dict(clean)
        m.eval()
        MODELS["lung"] = m
        logger.info("✅ Lung model loaded")
    except Exception as e:
        logger.warning(f" Lung model failed: {e}")

    # 7 ─ Kidney
    try:
        path = _download_weight("kidney_model1.h5")
        try:
            MODELS["kidney"] = tf.keras.models.load_model(path, compile=False)
        except TypeError:
            kidney_model = _build_kidney_cnn()
            kidney_model.load_weights(path)
            MODELS["kidney"] = kidney_model
        logger.info("✅ Kidney model loaded")
    except Exception as e:
        logger.warning(f"  Kidney model failed: {e}")



# U-Net builder (Heart)
# ──────────────────────────────────────────────────────────────────────────────
def _build_unet(input_shape=(128, 128, 1)):
    import tensorflow as tf
    L = tf.keras.layers
    inputs = L.Input(shape=input_shape)
    c1 = L.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = L.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = L.MaxPooling2D()(c1)
    c2 = L.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = L.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = L.MaxPooling2D()(c2)
    c3 = L.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = L.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = L.MaxPooling2D()(c3)
    c4 = L.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = L.Conv2D(128, 3, activation="relu", padding="same")(c4)
    p4 = L.MaxPooling2D()(c4)
    c5 = L.Conv2D(256, 3, activation="relu", padding="same")(p4)
    c5 = L.Conv2D(256, 3, activation="relu", padding="same")(c5)
    u6 = L.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u6 = L.concatenate([u6, c4])
    c6 = L.Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = L.Conv2D(128, 3, activation="relu", padding="same")(c6)
    u7 = L.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u7 = L.concatenate([u7, c3])
    c7 = L.Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = L.Conv2D(64, 3, activation="relu", padding="same")(c7)
    u8 = L.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u8 = L.concatenate([u8, c2])
    c8 = L.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = L.Conv2D(32, 3, activation="relu", padding="same")(c8)
    u9 = L.Conv2DTranspose(16, 2, strides=2, padding="same")(c8)
    u9 = L.concatenate([u9, c1])
    c9 = L.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = L.Conv2D(16, 3, activation="relu", padding="same")(c9)
    outputs = L.Conv2D(1, 1, activation="sigmoid")(c9)
    return tf.keras.models.Model(inputs, outputs)



# App lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    MODELS.clear()


app = FastAPI(
    title="Medical AI API",
    description="7 medical AI models in one unified API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Utility

async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    return Image.open(io.BytesIO(data))


def _require(key: str):
    if key not in MODELS:
        raise HTTPException(503, f"Model '{key}' is not loaded. Check weight file.")



# Root


# RECOMMENDATIONS — Arabic medical info for every disease

RECOMMENDATIONS = {

    # 1 Skin 
    "AK": {
        "disease_name": "AK — Actinic Keratosis",
        "description": "آفة جلدية تظهر نتيجة التعرض المفرط للشمس وتُعدّ مرحلة ما قبل سرطانية، قابلة للتحول لسرطان إذا تُركت دون علاج.",
        "recommendations": [
            "استشر طبيب جلدية لتقييم المنطقة وتحديد العلاج المناسب.",
            "من خيارات العلاج: التجميد بالنيتروجين أو كريمات طبية موضعية.",
            "الحد من التعرض للشمس المباشرة أمر بالغ الأهمية.",
            "استخدم واقي الشمس يومياً وأعد وضعه كل ساعتين عند الخروج.",
            "متابعة دورية مع الطبيب كل 6-12 شهراً.",
        ],
    },
    "BCC": {
        "disease_name": "BCC — Basal Cell Carcinoma",
        "description": "أكثر أنواع سرطان الجلد شيوعاً، ينمو ببطء ونادراً ما ينتشر، لكنه يحتاج علاجاً طبياً سريعاً لمنع تلف الأنسجة المحيطة.",
        "recommendations": [
            "راجع طبيب جلدية في أقرب وقت لتقييم المنطقة.",
            "الطبيب على الأرجح سيقترح إزالة جراحية للمنطقة المصابة.",
            "ابتعد عن التعرض المباشر للشمس خصوصاً بين 10 صباحاً و4 عصراً.",
            "ضع واقي شمس يومياً حتى في الأيام الغائمة.",
            "راقب أي تغير في الحجم أو اللون أو النزيف.",
        ],
    },
    "BKL": {
        "disease_name": "BKL — Benign Keratosis-like Lesion",
        "description": "نتوءات جلدية حميدة شائعة خاصةً مع التقدم في السن. ليست خطرة ولا تنتشر، لكنها قد تكون مزعجة من الناحية التجميلية.",
        "recommendations": [
            "الحالة حميدة في الغالب ولا تستدعي قلقاً فورياً.",
            "راقب المنطقة — إذا تغير الشكل أو اللون أو نزّت دماً راجع الطبيب.",
            "يمكن إزالتها جمالياً عند طبيب جلدية إذا أزعجتك.",
            "اعتنِ بنظافة المنطقة وتجنب حكّها.",
            "زيارة روتينية للطبيب سنوياً للاطمئنان.",
        ],
    },
    "DF": {
        "disease_name": "DF — Dermatofibroma",
        "description": "ورم جلدي حميد صلب يظهر عادةً في الساقين، ويُعتقد أنه يتكون بعد إصابة بسيطة كلسعة حشرة. نادراً ما يسبب مشاكل.",
        "recommendations": [
            "الورم الليفي حميد تماماً ولا يحتاج علاجاً في معظم الحالات.",
            "تجنب الضغط عليه أو العبث به.",
            "إذا كبر بسرعة أو سبّب ألماً اعرضه على الطبيب.",
            "يمكن إزالته جراحياً إذا أزعجك مظهره.",
            "مراجعة الطبيب سنوياً للاطمئنان.",
        ],
    },
    "MEL": {
        "disease_name": "MEL — Melanoma",
        "description": "نوع خطير من سرطان الجلد ينشأ من الخلايا المنتجة للميلانين. يُعدّ من أشد أنواع سرطان الجلد خطورةً إذا لم يُكتشف مبكراً.",
        "recommendations": [
            "اذهب لطبيب جلدية أو أورام فوراً — لا تتأخر.",
            "لا تتعرض لأشعة الشمس المباشرة على المنطقة المصابة.",
            "التقط صوراً دورية للمنطقة لمتابعة أي تغير في الشكل أو الحجم.",
            "استخدم واقي الشمس بعامل حماية SPF 50+ يومياً.",
            "قد يحتاج الطبيب لأخذ خزعة (biopsy) للتأكد من التشخيص.",
        ],
    },
    "NV": {
        "disease_name": "NV — Melanocytic Nevi",
        "description": "الشامات الشائعة الحميدة الناتجة عن تجمّع الخلايا المنتجة للميلانين. معظمها غير ضار لكن يجب مراقبتها باستمرار.",
        "recommendations": [
            "الشامة الطبيعية غير ضارة — لكن المراقبة المستمرة مهمة.",
            "اتبع قاعدة ABCDE: تغير في التماثل والحدود واللون والحجم والشكل العام.",
            "التقط صورة شهرية لمتابعة أي تغيير.",
            "واقي الشمس يومياً يقلل خطر تحوّلها.",
            "إذا لاحظت أي تغيير مفاجئ راجع الطبيب فوراً.",
        ],
    },
    "SCC": {
        "disease_name": "SCC — Squamous Cell Carcinoma",
        "description": "نوع من سرطان الجلد ينشأ من الطبقات الخارجية للجلد، وقد ينتشر للعقد الليمفاوية إذا أُهمل.",
        "recommendations": [
            "زيارة طبيب جلدية عاجلة ضرورية جداً.",
            "الطبيب سيحدد ما إذا كانت الإزالة الجراحية أو العلاج الإشعاعي مناسباً.",
            "تجنّب تماماً أجهزة تسمير الجلد (سولاريوم).",
            "واقي شمس SPF 50+ يومياً إلزامي.",
            "أخبر الطبيب بأي تاريخ مرضي سابق لأمراض الجلد.",
        ],
    },
    "VASC": {
        "disease_name": "VASC — Vascular Lesion",
        "description": "آفات تنشأ من الأوعية الدموية تحت الجلد كالأورام الوعائية والشعيرات المتوسعة. معظمها حميد تماماً.",
        "recommendations": [
            "معظم الآفات الوعائية حميدة ولا تحتاج تدخلاً طبياً عاجلاً.",
            "راقب المنطقة — أي نزيف أو ألم يستدعي زيارة الطبيب.",
            "يمكن علاجها بالليزر لأغراض تجميلية.",
            "حافظ على نظافة المنطقة وتجنب احتكاكها بالملابس.",
            "زيارة طبيب جلدية للتأكد من التشخيص.",
        ],
    },

    # 2 Breast 
    "benign": {
        "disease_name": "Benign — ورم حميد",
        "description": "الورم الحميد لا ينتشر ولا يشكّل خطراً مباشراً، لكن يجب متابعته طبياً بانتظام للتأكد من عدم تغيّره.",
        "recommendations": [
            "راجع طبيب مختص لتأكيد التشخيص بالخزعة إذا لزم.",
            "متابعة دورية بالموجات فوق الصوتية كل 6 أشهر.",
            "تجنّبي الضغط على المنطقة أو العبث بها.",
            "أبلغي الطبيب فوراً إذا تغيّر الحجم أو ظهر ألم.",
            "الفحص الذاتي الشهري للثدي مهم جداً للمتابعة.",
        ],
    },
    "malignant": {
        "disease_name": "Malignant — ورم خبيث",
        "description": "الورم الخبيث (السرطاني) يحتاج تدخلاً طبياً عاجلاً. الاكتشاف المبكر يرفع نسبة الشفاء بشكل كبير.",
        "recommendations": [
            "توجّهي لطبيب أورام فوراً — لا تتأخري أبداً.",
            "ستحتاجين لفحوصات إضافية: خزعة، تصوير بالرنين، سكان عظام.",
            "خيارات العلاج تشمل: جراحة، كيماوي، إشعاع، علاج هرموني.",
            "احرصي على دعم نفسي وعائلي خلال فترة العلاج.",
            "لا تعتمدي على أي علاج شعبي بديل دون استشارة الطبيب.",
        ],
    },
    "normal": {
        "disease_name": "Normal — طبيعي",
        "description": "لم تظهر علامات مرضية في الصورة. الثدي يبدو بصحة جيدة.",
        "recommendations": [
            "استمري في الفحص الذاتي الشهري للثدي.",
            "احرصي على فحص دوري سنوي عند الطبيب بعد سن الـ 40.",
            "الماموغرام مهم كل عامين للنساء فوق 40 سنة.",
            "نظام غذائي صحي وممارسة الرياضة تقلل خطر سرطان الثدي.",
            "أبلغي الطبيبة فوراً عند ملاحظة أي تغيير غير طبيعي.",
        ],
    },

    # 3 Eye 
    "Cataract": {
        "disease_name": "Cataract — إعتام عدسة العين",
        "description": "تعتّم في عدسة العين الطبيعية يؤدي إلى ضبابية في الرؤية. شائع مع التقدم في السن وقابل للعلاج الجراحي.",
        "recommendations": [
            "راجع طبيب عيون لتقييم مدى تأثيره على رؤيتك.",
            "العملية الجراحية بسيطة وآمنة وتعيد الرؤية بشكل كامل في معظم الحالات.",
            "استخدم نظارات شمسية لحماية العينين من الأشعة فوق البنفسجية.",
            "تجنب قيادة السيارة ليلاً إذا كانت رؤيتك ضعيفة.",
            "نظام غذائي غني بالفيتامينات C وE يحمي العينين.",
        ],
    },
    "Diabetic Retinopathy": {
        "disease_name": "Diabetic Retinopathy — اعتلال الشبكية السكري",
        "description": "مضاعفة لمرض السكري تصيب الأوعية الدموية في شبكية العين وقد تؤدي لفقدان البصر إذا لم تُعالج.",
        "recommendations": [
            "تحكّم في مستوى السكر في الدم بشكل صارم.",
            "راجع طبيب شبكية متخصص في أقرب وقت.",
            "قد تحتاج ليزر أو حقن مضادة للـ VEGF حسب الحالة.",
            "افحص عينيك سنوياً على الأقل إذا كنت مصاباً بالسكري.",
            "التحكم في ضغط الدم والكوليسترول يبطئ تطور المرض.",
        ],
    },
    "Glaucoma": {
        "disease_name": "Glaucoma — المياه الزرقاء",
        "description": "مجموعة من أمراض العين تصيب العصب البصري غالباً بسبب ارتفاع ضغط العين. قد تؤدي لفقدان البصر التدريجي.",
        "recommendations": [
            "راجع طبيب عيون فوراً لقياس ضغط العين.",
            "العلاج المبكر بالقطرات أو الجراحة يوقف تطور المرض.",
            "لا تتوقف عن استخدام قطرات الضغط دون استشارة الطبيب.",
            "افحص عينيك دورياً خاصة إذا كان هناك تاريخ عائلي للمرض.",
            "تجنب الأنشطة التي ترفع ضغط العين كرفع الأثقال الشديد.",
        ],
    },
    "Normal": {
        "disease_name": "Normal — طبيعي",
        "description": "لم تظهر علامات لأمراض العين في الصورة. العين تبدو بصحة جيدة.",
        "recommendations": [
            "احرص على فحص العينين مرة كل عام أو عامين.",
            "نظام غذائي غني بالفيتامينات A وC وE مفيد للعيون.",
            "استخدم نظارات شمسية ذات حماية UV عند الخروج.",
            "خذ استراحة كل 20 دقيقة عند استخدام الشاشات.",
            "تجنب التدخين فإنه يزيد خطر أمراض العيون.",
        ],
    },

    # 4 Brain 
    "glioma": {
        "disease_name": "Glioma — ورم الغليوما",
        "description": "نوع من أورام المخ ينشأ من الخلايا الدبقية. يتراوح بين درجات منخفضة الخطورة وأخرى عالية الخطورة.",
        "recommendations": [
            "توجّه لطبيب أعصاب أو جراح مخ فوراً.",
            "ستحتاج لتصوير بالرنين المغناطيسي المعزز لتحديد دقة الورم.",
            "خيارات العلاج تشمل: جراحة، إشعاع، كيماوي حسب درجة الورم.",
            "لا تتوقف عن أدوية الصرع إذا وُصفت لك.",
            "الدعم النفسي والعائلي جزء أساسي من رحلة العلاج.",
        ],
    },
    "meningioma": {
        "disease_name": "Meningioma — ورم السحايا",
        "description": "ورم ينشأ من أغشية المخ (السحايا). معظمه حميد وبطيء النمو، لكن قد يسبب ضغطاً على المخ.",
        "recommendations": [
            "راجع طبيب أعصاب لتقييم الورم ومتابعته.",
            "الأورام الصغيرة غير المسببة لأعراض قد تحتاج مراقبة فقط.",
            "الأورام الكبيرة أو المسببة لأعراض قد تحتاج جراحة أو إشعاع.",
            "متابعة بالرنين المغناطيسي كل 6-12 شهراً حسب توجيه الطبيب.",
            "أبلغ الطبيب فوراً عند ظهور صداع شديد أو تغيرات في الرؤية.",
        ],
    },
    "notumor": {
        "disease_name": "No Tumor — لا يوجد ورم",
        "description": "لم تظهر علامات لوجود ورم في الصورة. الدماغ يبدو طبيعياً.",
        "recommendations": [
            "إذا كانت لديك أعراض استشر طبيب أعصاب للتأكد.",
            "فحص دوري سنوي مهم خاصة عند وجود تاريخ عائلي.",
            "نوم كافٍ وتجنب الضغط النفسي الشديد يحمي صحة الدماغ.",
            "مارس الرياضة بانتظام لتحسين الدورة الدموية للمخ.",
            "تجنب التدخين والكحول للحفاظ على صحة الجهاز العصبي.",
        ],
    },
    "pituitary": {
        "disease_name": "Pituitary — ورم الغدة النخامية",
        "description": "ورم ينشأ في الغدة النخامية بقاعدة الدماغ. معظمه حميد لكن قد يؤثر على الهرمونات والرؤية.",
        "recommendations": [
            "راجع طبيب غدد صماء وطبيب أعصاب معاً.",
            "ستحتاج لتحليل هرمونات شاملة لتقييم تأثير الورم.",
            "بعض الأورام تُعالج بالدواء فقط دون جراحة.",
            "تابع رؤيتك — ورم الغدة قد يضغط على العصب البصري.",
            "الفحص الدوري بالرنين المغناطيسي ضروري لمتابعة حجم الورم.",
        ],
    },

    # 5 Heart 
    "normal": {
        "disease_name": "Normal — حجم القلب طبيعي",
        "description": "منطقة القلب المكتشفة في الصورة تقع ضمن النطاق الطبيعي المتوقع.",
        "recommendations": [
            "استمر في ممارسة الرياضة بانتظام للحفاظ على صحة القلب.",
            "نظام غذائي صحي قليل الدهون المشبعة مفيد للقلب.",
            "راقب ضغط الدم والكوليسترول بشكل دوري.",
            "تجنب التدخين وتقليل التوتر النفسي.",
            "فحص قلب سنوي مهم خاصة بعد سن الـ 40.",
        ],
    },
    "slightly_large": {
        "disease_name": "Slightly Large — تضخم بسيط في القلب",
        "description": "تم اكتشاف منطقة قلب أكبر قليلاً من المعدل الطبيعي في الصورة. يستدعي مراجعة طبية.",
        "recommendations": [
            "راجع طبيب قلب لإجراء تخطيط قلب وإيكو.",
            "تجنب المجهود الشديد حتى تأكيد التشخيص.",
            "راقب أي أعراض: ضيق تنفس، خفقان، تورم في القدمين.",
            "قلّل الملح في طعامك لتخفيف الضغط على القلب.",
            "الالتزام بالأدوية إذا وصفها الطبيب أمر ضروري.",
        ],
    },
    "abnormally_large": {
        "disease_name": "Abnormally Large — تضخم غير طبيعي في القلب",
        "description": "تم اكتشاف منطقة قلب أكبر بكثير من المعدل الطبيعي. يستدعي تدخلاً طبياً عاجلاً.",
        "recommendations": [
            "اذهب لطبيب قلب أو طوارئ فوراً.",
            "أجرِ تخطيط قلب (ECG) وإيكو قلب (Echocardiogram) عاجلاً.",
            "تجنب أي مجهود بدني حتى تحصل على تقييم طبي.",
            "أخبر الطبيب بكل الأدوية التي تتناولها.",
            "التضخم القلبي قابل للعلاج إذا اكتُشف مبكراً.",
        ],
    },
    "not_detected": {
        "disease_name": "Not Detected — لم يتم اكتشاف القلب",
        "description": "لم يتمكن الموديل من تحديد منطقة القلب بوضوح في الصورة. قد يكون بسبب جودة الصورة.",
        "recommendations": [
            "تأكد من جودة الصورة وأنها CT scan واضحة.",
            "أعد رفع صورة بجودة أعلى وزاوية مناسبة.",
            "راجع طبيب متخصص لإجراء الفحص اللازم.",
        ],
    },

    # 6 Lung 
    "Tuberculosis": {
        "disease_name": "Tuberculosis — السل الرئوي",
        "description": "عدوى بكتيرية تصيب الرئتين وتنتشر عن طريق الهواء. من أعراضها السعال المزمن ونزول الدم والتعرق الليلي وفقدان الوزن.",
        "recommendations": [
            "راجع طبيب أمراض صدر فوراً.",
            "يحتاج علاج مضادات حيوية لمدة 6 أشهر على الأقل.",
            "تجنب الاختلاط بالآخرين حتى تأكيد التشخيص.",
            "أجرِ تحليل بلغم وأشعة مقطعية للتأكد.",
            "الالتزام بجرعات الدواء كاملة بدون توقف أمر بالغ الأهمية.",
        ],
    },
    "Pneumonia-Viral": {
        "disease_name": "Pneumonia-Viral — التهاب رئوي فيروسي",
        "description": "التهاب في أنسجة الرئة بسبب فيروس. يسبب حمى وسعالاً وضيق تنفس وإرهاداً شديداً.",
        "recommendations": [
            "راجع طبيب في أقرب وقت.",
            "الراحة التامة وشرب السوائل بكثرة ضروريان.",
            "المضادات الحيوية لا تفيد (فيروسي) — الطبيب سيحدد العلاج المناسب.",
            "راقب مستوى الأكسجين في الدم.",
            "اذهب للطوارئ إذا نزل الأكسجين عن 94% أو اشتد ضيق التنفس.",
        ],
    },
    "Pneumonia-Bacterial": {
        "disease_name": "Pneumonia-Bacterial — التهاب رئوي بكتيري",
        "description": "التهاب رئوي بسبب بكتيريا. أعراضه أشد من الفيروسي وتشمل حمى عالية وسعال بلغمي وألم في الصدر.",
        "recommendations": [
            "اذهب للطبيب فوراً — يحتاج مضادات حيوية.",
            "لا تأخذ مضادات حيوية بدون وصفة طبية.",
            "الراحة التامة والسوائل الكافية ضرورية للتعافي.",
            "في الحالات الشديدة قد تحتاج دخول مستشفى.",
            "أكمل دورة المضادات الحيوية كاملة حتى لو تحسّنت.",
        ],
    },
    "Normal": {
        "disease_name": "Normal — رئة طبيعية",
        "description": "الأشعة لا تظهر أي تغيرات مرضية. الرئتان تبدوان بصحة جيدة.",
        "recommendations": [
            "استمر في الاهتمام بصحتك.",
            "ابتعد عن التدخين والأماكن الملوثة.",
            "مارس الرياضة بانتظام لتقوية الرئتين.",
            "إذا كانت عندك أعراض رغم النتيجة استشر طبيب.",
            "فحص سنوي مهم خاصة لمن يعملون في بيئات ملوثة.",
        ],
    },
    "Emphysema": {
        "disease_name": "Emphysema — انتفاخ الرئة",
        "description": "تلف في الحويصلات الهوائية بالرئة يجعل التنفس صعباً. غالباً مرتبط بالتدخين لفترات طويلة.",
        "recommendations": [
            "راجع طبيب أمراض صدر.",
            "التوقف عن التدخين فوراً هو أهم خطوة يمكنك اتخاذها.",
            "قد تحتاج موسّع للشعب الهوائية (بخاخ).",
            "تمارين التنفس مع متخصص تساعد كثيراً.",
            "تجنب الأماكن الملوثة والغبار والدخان.",
        ],
    },
    "Covid-19": {
        "disease_name": "Covid-19 — كوفيد-19",
        "description": "عدوى فيروسية تسببها فيروس كورونا. تظهر في الأشعة كبقع زجاجية في الرئتين. أعراضها تتراوح من خفيفة لشديدة.",
        "recommendations": [
            "أجرِ تحليل PCR لتأكيد الإصابة.",
            "عزل فوري لمنع انتشار العدوى.",
            "راجع طبيب لتقييم الحالة وتحديد العلاج.",
            "راقب مستوى الأكسجين — إذا نزل عن 94% اذهب للطوارئ.",
            "الراحة والسوائل والتغذية الجيدة تسرّع التعافي.",
        ],
    },

    # 7 Kidney 
    "Cyst": {
        "disease_name": "Cyst — كيس في الكلى",
        "description": "تجمّع سائل في الكلى يشكّل كيساً. معظم الأكياس حميدة ولا تسبب أعراضاً، لكن تحتاج متابعة.",
        "recommendations": [
            "راجع طبيب كلى لتقييم حجم الكيس وطبيعته.",
            "معظم الأكياس لا تحتاج علاجاً — متابعة بالأشعة كافية.",
            "اشرب كميات كافية من الماء يومياً (2-3 لتر).",
            "راقب أي أعراض: ألم في الجانب، دم في البول، ارتفاع ضغط.",
            "متابعة بالأشعة فوق الصوتية كل 6-12 شهراً.",
        ],
    },
    "Normal": {
        "disease_name": "Normal — كلية طبيعية",
        "description": "لم تظهر علامات مرضية في صورة الكلى. الكلية تبدو بصحة جيدة.",
        "recommendations": [
            "اشرب كميات كافية من الماء يومياً للحفاظ على صحة الكلى.",
            "قلّل الملح والبروتين الزائد في نظامك الغذائي.",
            "راقب ضغط الدم والسكر لأنهما يؤثران على الكلى.",
            "تجنب الإفراط في مسكنات الألم (مثل الإيبوبروفين).",
            "فحص وظائف الكلى سنوياً مهم خاصة بعد سن الـ 40.",
        ],
    },
    "Stone": {
        "disease_name": "Stone — حصى في الكلى",
        "description": "تراكم معادن وأملاح داخل الكلى يشكّل حصوات. قد تسبب ألماً شديداً عند تحرّكها.",
        "recommendations": [
            "راجع طبيب كلى أو مسالك بولية لتحديد حجم الحصى.",
            "اشرب 3 لتر ماء يومياً على الأقل لمساعدة الحصى على الخروج.",
            "تجنب الأطعمة الغنية بالأوكسالات كالسبانخ والشوكولاتة.",
            "الحصى الصغيرة تخرج وحدها — الكبيرة تحتاج تفتيت أو جراحة.",
            "اذهب للطوارئ إذا اشتد الألم أو ظهر دم في البول مع حمى.",
        ],
    },
    "Tumor": {
        "disease_name": "Tumor — ورم في الكلى",
        "description": "تم اكتشاف كتلة غير طبيعية في الكلى. يحتاج تقييماً طبياً عاجلاً لتحديد طبيعة الورم.",
        "recommendations": [
            "توجّه لطبيب أورام أو كلى فوراً — لا تتأخر.",
            "ستحتاج لأشعة مقطعية (CT) بصبغة لتحديد طبيعة الورم.",
            "خيارات العلاج تشمل: جراحة، استئصال جزئي أو كلي للكلى، أو علاج موضعي.",
            "الاكتشاف المبكر يرفع نسبة الشفاء بشكل كبير.",
            "لا تتجاهل أي أعراض: دم في البول، ألم في الجانب، فقدان وزن.",
        ],
    },
}


@app.get("/")
def root():
    return {
        "message": "Medical AI API — 7 models",
        "loaded_models": list(MODELS.keys()),
        "endpoints": [
            "/predict/skin",
            "/predict/breast",
            "/predict/eye",
            "/predict/brain",
            "/predict/heart",
            "/predict/lung",
            "/predict/kidney",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS.keys())}



# 1 ─ SKIN

SKIN_LABELS = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
SKIN_NAMES  = {
    "AK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "SCC": "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesion",
}
SKIN_SEVERITY = {
    "MEL": "high", "BCC": "high", "SCC": "high",
    "AK": "medium",
    "BKL": "low", "DF": "low", "NV": "low", "VASC": "low",
}


@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    """
    Classify skin lesion from a dermatoscopy image.

    Returns: predicted class, confidence, severity, all probabilities.
    """
    _require("skin")
    img = await read_image(file)
    arr = np.array(img.convert("RGB").resize((28, 28)), dtype=np.float32)
    arr = np.expand_dims(arr, 0)

    probs     = MODELS["skin"].predict(arr, verbose=0)[0]
    idx       = int(np.argmax(probs))
    label     = SKIN_LABELS[idx]

    rec = RECOMMENDATIONS.get(label, {})
    return {
        "model": "skin",
        "predicted_class": label,
        "predicted_name": SKIN_NAMES[label],
        "confidence": round(float(probs[idx]) * 100, 2),
        "severity": SKIN_SEVERITY[label],
        "all_probabilities": {
            SKIN_LABELS[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 2 ─ BREAST
# ................................................................
BREAST_CLASSES = ["benign", "malignant", "normal"]


@app.post("/predict/breast")
async def predict_breast(file: UploadFile = File(...)):
    """
    Breast ultrasound — runs segmentation + classification.

    Returns:
      - classification result (benign / malignant / normal)
      - segmentation mask stats (mean activation, coverage %)
    """
    _require("breast_seg")
    _require("breast_cls")

    img = await read_image(file)

    # ── segmentation ──
    seg_in  = np.array(img.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    seg_in  = np.expand_dims(seg_in, 0)
    mask    = MODELS["breast_seg"].predict(seg_in, verbose=0)[0, :, :, 0]
    coverage = round(float(np.mean(mask > 0.5)) * 100, 2)

    # ── classification ──
    cls_in  = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    cls_in  = np.expand_dims(cls_in, 0)
    probs   = MODELS["breast_cls"].predict(cls_in, verbose=0)[0]
    idx     = int(np.argmax(probs))

    mask_b64    = _mask_to_base64(mask, colormap="plasma")
    overlay_b64 = _overlay_to_base64(img, mask, size=(256, 256))
    rec = RECOMMENDATIONS.get(BREAST_CLASSES[idx], {})
    return {
        "model": "breast",
        "predicted_class": BREAST_CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {
            BREAST_CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "segmentation": {
            "mask_mean_activation": round(float(np.mean(mask)), 4),
            "lesion_coverage_percent": coverage,
            "mask_image_base64": mask_b64,
            "overlay_image_base64": overlay_b64,
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 3 ─ EYE
# ................................................................
EYE_CLASSES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


@app.post("/predict/eye")
async def predict_eye(file: UploadFile = File(...)):
    """
    Classify eye disease from a fundus / eye image.

    Returns: predicted class, confidence, all probabilities.
    """
    _require("eye")
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img  = await read_image(file)
    arr  = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    arr  = preprocess_input(arr)
    arr  = np.expand_dims(arr, 0)

    probs = MODELS["eye"].predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))

    rec = RECOMMENDATIONS.get(EYE_CLASSES[idx], {})
    return {
        "model": "eye",
        "predicted_class": EYE_CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {
            EYE_CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 4 ─ BRAIN
# ................................................................
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


@app.post("/predict/brain")
async def predict_brain(file: UploadFile = File(...)):
    """
    Classify brain tumor type from an MRI image.

    Returns: predicted class, confidence, all probabilities.
    """
    _require("brain")
    import cv2

    img_pil = await read_image(file)
    img_np  = np.array(img_pil.convert("RGB"))
    img_np  = cv2.resize(img_np, (224, 224))
    arr     = img_np.astype(np.float32) / 255.0
    arr     = np.expand_dims(arr, 0)

    probs = MODELS["brain"].predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))

    rec = RECOMMENDATIONS.get(BRAIN_CLASSES[idx], {})
    return {
        "model": "brain",
        "predicted_class": BRAIN_CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {
            BRAIN_CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 5 ─ HEART
# ................................................................
@app.post("/predict/heart")
async def predict_heart(file: UploadFile = File(...)):
    """
    Heart segmentation from a CT scan image.

    Returns:
      - heart_area_ratio (%)
      - assessment (normal / slightly_large / abnormally_large / not_detected)
      - mask stats
    """
    _require("heart")

    img = await read_image(file)
    arr = np.array(img.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, (0, 3))

    mask     = MODELS["heart"].predict(arr, verbose=0)[0, :, :, 0]
    mask_bin = (mask > 0.5).astype(np.uint8)
    ratio    = float(np.sum(mask_bin)) / (128 * 128)

    if ratio < 0.01:
        assessment = "not_detected"
    elif ratio < 0.15:
        assessment = "normal"
    elif ratio < 0.35:
        assessment = "slightly_large"
    else:
        assessment = "abnormally_large"

    mask_b64    = _mask_to_base64(mask, colormap="hot")
    overlay_b64 = _overlay_to_base64(img, mask, size=(128, 128))
    rec = RECOMMENDATIONS.get(assessment, {})
    return {
        "model": "heart",
        "task": "segmentation",
        "heart_area_ratio_percent": round(ratio * 100, 2),
        "assessment": assessment,
        "mask_stats": {
            "mean_activation": round(float(np.mean(mask)), 4),
            "max_activation":  round(float(np.max(mask)), 4),
            "detected_pixels": int(np.sum(mask_bin)),
        },
        "segmentation": {
            "mask_image_base64": mask_b64,
            "overlay_image_base64": overlay_b64,
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 6 ─ LUNG (PyTorch)
# ................................................................
LUNG_CLASSES = [
    "Tuberculosis", "Pneumonia-Viral", "Pneumonia-Bacterial",
    "Normal", "Emphysema", "Covid-19",
]


@app.post("/predict/lung")
async def predict_lung(file: UploadFile = File(...)):
    """
    Classify chest X-ray disease (6 classes).

    Returns: predicted class, confidence, all probabilities.
    """
    _require("lung")
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    img  = await read_image(file)
    arr  = np.array(img.convert("RGB"))
    t    = transform(image=arr)["image"].unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(MODELS["lung"](t), dim=1).squeeze().numpy()

    idx = int(np.argmax(probs))

    rec = RECOMMENDATIONS.get(LUNG_CLASSES[idx], {})
    return {
        "model": "lung",
        "predicted_class": LUNG_CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {
            LUNG_CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },
    }



# 7 ─ KIDNEY
# ................................................................
KIDNEY_CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]


@app.post("/predict/kidney")
async def predict_kidney(file: UploadFile = File(...)):
    """
    Classify kidney disease from a CT scan image.

    Returns: predicted class, confidence, all probabilities.
    """
    _require("kidney")

    img  = await read_image(file)
    arr  = np.array(img.convert("L").resize((200, 200)), dtype=np.float32) / 255.0
    arr  = arr.reshape(1, 200, 200, 1)

    probs = MODELS["kidney"].predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))

    rec = RECOMMENDATIONS.get(KIDNEY_CLASSES[idx], {})
    return {
        "model": "kidney",
        "predicted_class": KIDNEY_CLASSES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "all_probabilities": {
            KIDNEY_CLASSES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "disease_info": {
            "disease_name": rec.get("disease_name", ""),
            "description": rec.get("description", ""),
            "recommendations": rec.get("recommendations", []),
        },

    }
