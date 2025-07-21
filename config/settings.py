"""
Configuración centralizada para la aplicación SIPaKMeD
"""
import os
from pathlib import Path
import streamlit as st

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
STATIC_DIR = PROJECT_ROOT / "static"
TRANSLATIONS_DIR = PROJECT_ROOT / "translations"

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================
APP_CONFIG = {
    "title": "🔬 Clasificador de Células Cervicales",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================
MODEL_CONFIG = {
    "model_files": {
        "MobileNetV2": "sipakmed_MobileNetV2.h5",
        "ResNet50": "sipakmed_ResNet50.h5",
        "EfficientNetB0": "sipakmed_EfficientNetB0.h5"
    },
    "input_size": (224, 224),
    "num_classes": 5,
    "class_names": [
        "dyskeratotic", 
        "koilocytotic", 
        "metaplastic", 
        "parabasal", 
        "superficial_intermediate"
    ]
}

# ============================================================================
# CONFIGURACIÓN DE UI
# ============================================================================
UI_CONFIG = {
    "supported_formats": ['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    "max_upload_size": 200,  # MB
    "default_language": "es",
    "available_languages": {
        "es": "🇪🇸 Español",
        "en": "🇺🇸 English", 
        "pt": "🇧🇷 Português",
        "fr": "🇫🇷 Français"
    }
}

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS
# ============================================================================
ANALYSIS_CONFIG = {
    "confidence_threshold": 0.5,
    "consensus_threshold": 0.67,
    "clahe_clip_limit": 3.0,
    "clahe_tile_grid_size": (8, 8)
}

# ============================================================================
# CONFIGURACIÓN DE REPORTES
# ============================================================================
REPORT_CONFIG = {
    "pdf_page_size": "A4",
    "include_charts": True,
    "chart_dpi": 300,
    "chart_format": "PNG"
}

# ============================================================================
# FUNCIONES DE CONFIGURACIÓN
# ============================================================================
def setup_streamlit_config():
    """Configura Streamlit con los parámetros de la aplicación"""
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
    )

def get_model_path(model_name):
    """Retorna la ruta completa del modelo"""
    filename = MODEL_CONFIG["model_files"].get(model_name)
    if filename:
        return MODELS_DIR / filename
    return None

def get_static_file_path(filename):
    """Retorna la ruta completa de un archivo estático"""
    return STATIC_DIR / filename

def get_translation_file_path(language):
    """Retorna la ruta completa del archivo de traducción"""
    return TRANSLATIONS_DIR / f"{language}.json"

def get_figure_path(filename):
    """Retorna la ruta completa de una figura"""
    return FIGURES_DIR / filename

# ============================================================================
# VARIABLES DE ENTORNO
# ============================================================================
def get_env_var(var_name, default_value=None):
    """Obtiene una variable de entorno con valor por defecto"""
    return os.getenv(var_name, default_value)

# Configuraciones desde variables de entorno (opcional)
DEBUG_MODE = get_env_var("SIPAKMED_DEBUG", "False").lower() == "true"
LOG_LEVEL = get_env_var("SIPAKMED_LOG_LEVEL", "INFO")

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
LOGGING_CONFIG = {
    "level": LOG_LEVEL,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "disable_tensorflow_warnings": True
}

# ============================================================================
# CONFIGURACIÓN DE GPU/CPU
# ============================================================================
COMPUTE_CONFIG = {
    "use_gpu": True,
    "gpu_memory_growth": True,
    "mixed_precision": False
}

def setup_tensorflow():
    """Configura TensorFlow según las preferencias"""
    try:
        import tensorflow as tf
        
        # Configurar GPU si está disponible
        if COMPUTE_CONFIG["use_gpu"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Permitir crecimiento de memoria
                    if COMPUTE_CONFIG["gpu_memory_growth"]:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Configurar precisión mixta si está habilitada
                    if COMPUTE_CONFIG["mixed_precision"]:
                        tf.config.optimizer.set_experimental_options(
                            {'auto_mixed_precision': True}
                        )
                        
                except RuntimeError as e:
                    print(f"Error configurando GPU: {e}")
        
        # Suprimir warnings de TensorFlow si está configurado
        if LOGGING_CONFIG["disable_tensorflow_warnings"]:
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
    except ImportError:
        print("TensorFlow no está disponible")

# ============================================================================
# INICIALIZACIÓN
# ============================================================================
def initialize_app():
    """Inicializa la configuración completa de la aplicación"""
    setup_streamlit_config()
    setup_tensorflow()
    
    # Crear directorios si no existen
    directories = [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, STATIC_DIR, TRANSLATIONS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)