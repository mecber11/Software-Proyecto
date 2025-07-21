"""
Configuracion global del proyecto SIPaKMeD (Compatible Windows)
"""
import os
from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURACION SIPAKMED
# ============================================================================
SIPAKMED_USERNAME = "juliusdta"
SIPAKMED_TOKEN = "3611de84f34951034eef4a9ab943c79e"

# ============================================================================
# CONFIGURACION HÍBRIDA AVANZADA
# ============================================================================

# Arquitecturas híbridas disponibles
HYBRID_ARCHITECTURES = {
    'ensemble': {
        'name': 'Ensemble Hybrid CNN',
        'description': 'Fusión de MobileNetV2, ResNet50 y EfficientNetB0 con atención',
        'base_models': ['MobileNetV2', 'ResNet50', 'EfficientNetB0'],
        'fusion_method': 'attention_weighted',
        'trainable_layers': {'MobileNetV2': 20, 'ResNet50': 30, 'EfficientNetB0': 25}
    },
    'multiscale': {
        'name': 'Multi-Scale Hybrid CNN',
        'description': 'CNN personalizada con bloques multi-escala y atención espacial',
        'base_models': ['Custom Multi-Scale'],
        'fusion_method': 'multi_scale_attention',
        'stages': [64, 128, 256, 512],
        'blocks_per_stage': [2, 3, 4, 3]
    }
}

# Configuración de preprocesamiento avanzado
ADVANCED_PREPROCESSING = {
    'clahe': {
        'enabled': True,
        'clip_limit': 3.0,
        'tile_grid_size': (8, 8),
        'adaptive': True,
        'multi_scale': True,
        'scales': [0.5, 1.0, 1.5]
    },
    'segmentation': {
        'enabled': True,
        'method': 'watershed',  # 'watershed' o 'kmeans'
        'enhance_boundaries': True,
        'kmeans_k': 3
    },
    'augmentation': {
        'geometric': {
            'rotation_range': 15,
            'zoom_range': 0.2,
            'shift_range': 0.1,
            'flip_horizontal': True,
            'flip_vertical': True,
            'elastic_transform': True
        },
        'intensity': {
            'brightness_range': 0.2,
            'contrast_range': 0.2,
            'gamma_range': (0.8, 1.2),
            'hue_saturation': True,
            'gaussian_noise': True,
            'blur_variants': True
        },
        'advanced': {
            'clahe_augment': True,
            'sharpen': True,
            'cutout': True,
            'coarse_dropout': True,
            'grid_distortion': True,
            'optical_distortion': True
        }
    }
}

# Configuración de entrenamiento híbrido
HYBRID_TRAINING = {
    'epochs': 25,
    'batch_size': 16,
    'learning_rate': 0.001,
    'image_size': (224, 224),
    'validation_split': 0.2,
    'early_stopping_patience': 8,
    'reduce_lr_patience': 5,
    'mixed_precision': True,
    'use_class_weights': True,
    'save_best_only': True,
    'callbacks': {
        'tensorboard': True,
        'csv_logger': True,
        'model_checkpoint': True,
        'early_stopping': True,
        'reduce_lr': True,
        'custom_lr_scheduler': True
    }
}

# ============================================================================
# CLASES DEL DATASET
# ============================================================================
REAL_CLASSES = {
    "dyskeratotic": 0,
    "koilocytotic": 1,
    "metaplastic": 2,
    "parabasal": 3,
    "superficial_intermediate": 4
}

CLASS_NAMES_FRIENDLY = {
    "dyskeratotic": "Celulas Displasicas",
    "koilocytotic": "Celulas Koilocitoticas",
    "metaplastic": "Celulas Metaplasicas", 
    "parabasal": "Celulas Parabasales",
    "superficial_intermediate": "Celulas Superficiales-Intermedias"
}

# ============================================================================
# HIPERPARAMETROS DE ENTRENAMIENTO
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Data augmentation
ROTATION_RANGE = 180
ZOOM_RANGE = 0.3
SHEAR_RANGE = 0.2
BRIGHTNESS_RANGE = (0.7, 1.3)
SHIFT_RANGE = 0.2

# Early stopping
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7

# ============================================================================
# CONFIGURACION DE HARDWARE
# ============================================================================
# Configurar GPU si esta disponible
GPU_MEMORY_GROWTH = True
MIXED_PRECISION = False  # Cambiar a True para entrenar mas rapido en GPU

# ============================================================================
# CONFIGURACION DE LOGGING
# ============================================================================
VERBOSE = 1  # 0: silencioso, 1: informacion basica, 2: informacion detallada

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def get_class_names():
    """Retorna lista de nombres de clases"""
    return list(REAL_CLASSES.keys())

def get_friendly_names():
    """Retorna lista de nombres amigables"""
    return list(CLASS_NAMES_FRIENDLY.values())

def get_model_path(model_name):
    """Retorna path completo para guardar modelo"""
    return MODELS_DIR / f"sipakmed_{model_name}.h5"

def get_report_path(model_name, extension="txt"):
    """Retorna path completo para guardar reporte"""
    return METRICS_DIR / f"sipakmed_{model_name}_report.{extension}"

def get_figure_path(filename):
    """Retorna path completo para guardar figura"""
    return FIGURES_DIR / filename