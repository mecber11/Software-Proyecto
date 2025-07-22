"""
Módulo para carga de datos, modelos y recursos para SIPaKMeD
"""
import os
import json
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rutas de datos - Corregidas para la estructura actual
import os
# Desde utils/data_loader.py ir hacia el directorio ISO-Final
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # utils/
BASE_DIR = os.path.dirname(CURRENT_DIR)  # ISO-Final/
MODEL_PATH = os.path.join(BASE_DIR, "data", "models")
FIGURES_PATH = os.path.join(BASE_DIR, "reports", "figures") 
TRANSLATIONS_PATH = os.path.join(BASE_DIR, "translations")

@st.cache_resource
def load_models():
    """Carga los modelos entrenados de SIPaKMeD - OPTIMIZADA CON CACHE"""
    models = {}
    model_files = {
        "MobileNetV2": "sipakmed_MobileNetV2.h5",
        "ResNet50": "sipakmed_ResNet50.h5", 
        "EfficientNetB0": "sipakmed_EfficientNetB0.h5"
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, filename) in enumerate(model_files.items()):
        model_path = os.path.join(MODEL_PATH, filename)
        status_text.text(f'🔄 Cargando {name}...')
        
        if os.path.exists(model_path):
            try:
                models[name] = load_model(model_path)
                progress_bar.progress((i + 1) / len(model_files))
                logger.info(f"Modelo {name} cargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error cargando {name}: {str(e)}")
                logger.error(f"Error cargando {name}: {str(e)}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {model_path}")
            logger.warning(f"Archivo no encontrado: {model_path}")
    
    progress_bar.empty()
    status_text.empty()
    
    return models

@st.cache_data
def load_translations(language="es"):
    """Carga las traducciones desde archivos JSON"""
    try:
        translation_file = os.path.join(TRANSLATIONS_PATH, f"{language}.json")
        logger.info(f"Intentando cargar traducciones desde: {translation_file}")
        
        if os.path.exists(translation_file):
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                logger.info(f"Traducciones cargadas exitosamente: {len(translations)} claves")
                return translations
        else:
            logger.warning(f"Archivo no encontrado: {translation_file}")
    except Exception as e:
        logger.error(f"Error cargando traducciones desde {translation_file}: {e}")
    
    # Fallback a español
    try:
        es_file = os.path.join(TRANSLATIONS_PATH, "es.json")
        logger.info(f"Intentando fallback a español: {es_file}")
        
        if os.path.exists(es_file):
            with open(es_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                logger.info(f"Traducciones ES cargadas: {len(translations)} claves")
                return translations
        else:
            logger.error(f"Archivo de español no encontrado: {es_file}")
    except Exception as e:
        logger.error(f"Error cargando español: {e}")
    
    # Últimos fallbacks
    logger.error("No se pudieron cargar traducciones, usando valores por defecto")
    return {
        'models': 'modelos',
        'models_loaded': 'Cargados',
        'processing_mode': 'Procesamiento',
        'accuracy_range': 'Rango',
        'cell_types_count': 'Tipos de células',
        'main_title': 'Clasificador de Células Cervicales',
        'subtitle': 'Sistema de análisis automatizado basado en Deep Learning'
    }

@st.cache_data
def load_training_images():
    """Carga las imágenes generadas durante el entrenamiento - OPTIMIZADA"""
    training_images = {}
    
    # Definir las imágenes esperadas (clásicas + híbridas)
    image_files = {
        "confusion_matrices": [
            "confusion_matrix_MobileNetV2.png",
            "confusion_matrix_ResNet50.png", 
            "confusion_matrix_EfficientNetB0.png",
            "confusion_matrix_HybridEnsemble.png",
            "confusion_matrix_HybridMultiScale.png"
        ],
        "training_histories": [
            "training_history_MobileNetV2.png",
            "training_history_ResNet50.png",
            "training_history_EfficientNetB0.png",
            "training_history_HybridEnsemble.png",
            "training_history_HybridMultiScale.png"
        ],
        "model_comparison": "models_comparison.png",
        "hybrid_roc_curves": "roc_curves_hybrid_models.png",
        "hybrid_comparison": "models_comparison_hybrid.png"
    }
    
    # Verificar y cargar imágenes (buscar en múltiples ubicaciones)
    for category, files in image_files.items():
        if isinstance(files, list):
            training_images[category] = []
            for file in files:
                # Buscar primero en reports/figures
                file_path = os.path.join(FIGURES_PATH, file)
                
                # Si no está, buscar en hybrid_training_results/figures
                if not os.path.exists(file_path):
                    hybrid_figures_path = os.path.join(BASE_DIR, "hybrid_training_results", "figures")
                    file_path = os.path.join(hybrid_figures_path, file)
                
                # Si tampoco está, buscar en directorios alternativos
                if not os.path.exists(file_path):
                    alt_hybrid_path = os.path.join(os.path.dirname(BASE_DIR), "hybrid_training_results", "figures")
                    file_path = os.path.join(alt_hybrid_path, file)
                
                if os.path.exists(file_path):
                    training_images[category].append({
                        'path': file_path,
                        'name': file.replace('.png', '').replace('_', ' ').title(),
                        'model': file.split('_')[-1].replace('.png', '')
                    })
        else:
            # Buscar archivo individual en múltiples ubicaciones
            file_path = os.path.join(FIGURES_PATH, files)
            
            if not os.path.exists(file_path):
                hybrid_figures_path = os.path.join(BASE_DIR, "hybrid_training_results", "figures")
                file_path = os.path.join(hybrid_figures_path, files)
            
            if not os.path.exists(file_path):
                alt_hybrid_path = os.path.join(os.path.dirname(BASE_DIR), "hybrid_training_results", "figures")
                file_path = os.path.join(alt_hybrid_path, files)
            
            if os.path.exists(file_path):
                training_images[category] = {
                    'path': file_path,
                    'name': files.replace('.png', '').replace('_', ' ').title()
                }
    
    return training_images


@st.cache_data
def load_hybrid_training_results():
    """Carga los resultados del entrenamiento híbrido si existen"""
    try:
        # Buscar en el directorio híbrido
        hybrid_report_path = os.path.join(BASE_DIR, "hybrid_training_results", "metrics", "hybrid_training_report.json")
        if os.path.exists(hybrid_report_path):
            with open(hybrid_report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Buscar en directorios alternativos
        alt_paths = [
            os.path.join(os.path.dirname(BASE_DIR), "hybrid_training_results", "metrics", "hybrid_training_report.json"),
            os.path.join(BASE_DIR, "..", "hybrid_training_results", "metrics", "hybrid_training_report.json")
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                with open(alt_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
    except Exception as e:
        logger.error(f"Error cargando resultados híbridos: {e}")
    return None

def get_available_languages():
    """Obtiene la lista de idiomas disponibles"""
    languages = {
        "es": "🇪🇸 Español",
        "en": "🇺🇸 English", 
        "pt": "🇧🇷 Português",
        "fr": "🇫🇷 Français",
        "zh": "🇨🇳 中文"
    }
    
    available_languages = {}
    for lang_code, lang_name in languages.items():
        translation_file = os.path.join(TRANSLATIONS_PATH, f"{lang_code}.json")
        if os.path.exists(translation_file):
            available_languages[lang_code] = lang_name
    
    return available_languages

def get_language():
    """Obtiene el idioma seleccionado de la sesión"""
    try:
        return st.session_state.get('selected_language', 'es')
    except Exception:
        # Fallback si session_state no está disponible
        logger.warning("session_state no disponible, usando español por defecto")
        return 'es'

def set_language(language):
    """Establece el idioma en la sesión"""
    try:
        st.session_state.selected_language = language
    except Exception:
        # Si session_state no está disponible, simplemente ignorar
        logger.warning(f"No se pudo establecer idioma {language}, session_state no disponible")
        pass

def get_class_names_friendly():
    """Retorna nombres amigables de las clases"""
    return {
        "dyskeratotic": "Células Displásicas",
        "koilocytotic": "Células Koilocitóticas", 
        "metaplastic": "Células Metaplásicas",
        "parabasal": "Células Parabasales",
        "superficial_intermediate": "Células Superficiales-Intermedias"
    }

def get_clinical_info():
    """Retorna información clínica de las clases"""
    # Usar traducciones dinámicas basadas en el idioma seleccionado
    translations = load_translations(get_language())
    
    def t(key):
        return translations.get(key, key)
    
    return {
        "dyskeratotic": {
            "descripcion": t("dyskeratotic_desc"),
            "significado": t("dyskeratotic_meaning"),
            "color": "#FC424A",
            "riesgo": t("high_risk"),
            "icon": "🔴"
        },
        "koilocytotic": {
            "descripcion": t("koilocytotic_desc"),
            "significado": t("koilocytotic_meaning"),
            "color": "#FFAB00",
            "riesgo": t("moderate_risk"),
            "icon": "🟡"
        },
        "metaplastic": {
            "descripcion": t("metaplastic_desc"),
            "significado": t("metaplastic_meaning"),
            "color": "#0066CC",
            "riesgo": t("low_risk"),
            "icon": "🟡"
        },
        "parabasal": {
            "descripcion": t("parabasal_desc"),
            "significado": t("parabasal_meaning"),
            "color": "#00D25B",
            "riesgo": t("normal_risk"),
            "icon": "🟢"
        },
        "superficial_intermediate": {
            "descripcion": t("superficial_intermediate_desc"),
            "significado": t("superficial_intermediate_meaning"),
            "color": "#00D25B", 
            "riesgo": t("normal_risk"),
            "icon": "🟢"
        }
    }

def load_hybrid_comparison_results():
    """Carga los resultados de comparación entre modelos clásicos e híbridos"""
    try:
        comparison_file = os.path.join(BASE_DIR, "reports", "hybrid_statistical_analysis", "hybrid_comparison_complete.json")
        
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
                logger.info("Resultados de comparación híbrida cargados exitosamente")
                return comparison_data
        else:
            logger.info("No se encontraron resultados de comparación híbrida")
            return None
    except Exception as e:
        logger.error(f"Error cargando comparación híbrida: {e}")
        return None

def load_project_data():
    """Carga todos los datos del proyecto de una vez"""
    return {
        'models': load_models(),
        'training_images': load_training_images(),
        'hybrid_comparison': load_hybrid_comparison_results(),
        'available_languages': get_available_languages()
    }