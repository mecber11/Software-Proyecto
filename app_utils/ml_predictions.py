"""
Módulo de predicciones y preprocesamiento ML para SIPaKMeD
"""
import numpy as np
from PIL import Image

# Importación condicional de OpenCV para evitar conflictos
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import warnings
    warnings.warn("OpenCV no disponible. Algunas funciones de mejora de imagen estarán deshabilitadas.")
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_cervical_cell_image(image):
    """
    Mejora la imagen de células cervicales usando CLAHE y técnicas de procesamiento
    """
    try:
        # Convertir PIL a array numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if not CV2_AVAILABLE:
            # Mejora básica sin OpenCV
            logger.warning("OpenCV no disponible, usando mejora básica")
            # Normalizar contraste básico
            img_float = img_array.astype(np.float32) / 255.0
            # Aumentar contraste
            enhanced = np.clip((img_float - 0.5) * 1.2 + 0.5, 0, 1)
            return (enhanced * 255).astype(np.uint8)
        
        # Convertir a escala de grises si es necesario
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Aplicar filtro bilateral para reducir ruido manteniendo bordes
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convertir de vuelta a RGB
        if len(img_array.shape) == 3:
            enhanced_rgb = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2RGB)
        else:
            enhanced_rgb = np.stack([bilateral] * 3, axis=-1)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.error(f"Error en mejora de imagen: {e}")
        # Retornar imagen original en caso de error
        return np.array(image) if isinstance(image, Image.Image) else image

def preprocess_image(image, model_name):
    """
    Preprocesa la imagen para un modelo específico
    
    Args:
        image: PIL Image o numpy array
        model_name: Nombre del modelo (MobileNetV2, ResNet50, EfficientNetB0)
    
    Returns:
        numpy array preprocessed
    """
    try:
        # Convertir a array si es PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Asegurar que es RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        elif len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        else:
            img_rgb = img_array[:,:,:3]
        
        # Redimensionar a 224x224
        if CV2_AVAILABLE:
            img_resized = cv2.resize(img_rgb, (224, 224))
        else:
            # Fallback usando PIL
            pil_img = Image.fromarray(img_rgb.astype('uint8'))
            pil_resized = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            img_resized = np.array(pil_resized)
        
        # Expandir dimensiones para batch
        img_batch = np.expand_dims(img_resized, axis=0)
        
        # Aplicar preprocesamiento específico del modelo
        if model_name == "MobileNetV2":
            processed = mobilenet_preprocess(img_batch.astype(np.float32))
        elif model_name == "ResNet50":
            processed = resnet_preprocess(img_batch.astype(np.float32))
        elif model_name == "EfficientNetB0":
            processed = efficientnet_preprocess(img_batch.astype(np.float32))
        else:
            # Preprocesamiento por defecto
            processed = img_batch.astype(np.float32) / 255.0
        
        return processed
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento para {model_name}: {e}")
        # Preprocesamiento básico en caso de error
        img_array = np.array(image) if isinstance(image, Image.Image) else image
        if CV2_AVAILABLE:
            img_resized = cv2.resize(img_array, (224, 224))
        else:
            # Fallback usando PIL
            if isinstance(image, Image.Image):
                img_resized = np.array(image.resize((224, 224), Image.Resampling.LANCZOS))
            else:
                pil_img = Image.fromarray(img_array.astype('uint8'))
                img_resized = np.array(pil_img.resize((224, 224), Image.Resampling.LANCZOS))
        return np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

def predict_cervical_cells(image, models):
    """
    Realiza predicciones usando todos los modelos disponibles
    
    Args:
        image: PIL Image
        models: Diccionario con los modelos cargados
    
    Returns:
        Diccionario con predicciones de cada modelo
    """
    try:
        if not models:
            logger.error("No hay modelos disponibles para predicción")
            return {}
        
        # Mapeo de clases
        class_names = ["dyskeratotic", "koilocytotic", "metaplastic", "parabasal", "superficial_intermediate"]
        
        predictions = {}
        
        for model_name, model in models.items():
            try:
                # Preprocesar imagen para el modelo específico
                processed_image = preprocess_image(image, model_name)
                
                # Realizar predicción
                prediction = model.predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_idx])
                predicted_class = class_names[predicted_class_idx]
                
                # Obtener información clínica
                clinical_info = get_clinical_info_for_class(predicted_class)
                
                predictions[model_name] = {
                    'class': predicted_class,
                    'class_friendly': get_friendly_name(predicted_class),
                    'confidence': confidence,
                    'probabilities': prediction[0].tolist(),
                    'clinical_info': clinical_info
                }
                
                logger.info(f"{model_name}: {predicted_class} ({confidence:.2%})")
                
            except Exception as e:
                logger.error(f"Error en predicción con {model_name}: {e}")
                continue
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error general en predicciones: {e}")
        return {}

def get_clinical_info_for_class(cell_class):
    """
    Obtiene información clínica para una clase de célula
    """
    clinical_data = {
        "dyskeratotic": {
            "color": "#FC424A",
            "riesgo": "Alto",
            "icon": "🔴"
        },
        "koilocytotic": {
            "color": "#FFAB00", 
            "riesgo": "Moderado",
            "icon": "🟡"
        },
        "metaplastic": {
            "color": "#0066CC",
            "riesgo": "Bajo",
            "icon": "🟡"
        },
        "parabasal": {
            "color": "#00D25B",
            "riesgo": "Normal",
            "icon": "🟢"
        },
        "superficial_intermediate": {
            "color": "#00D25B",
            "riesgo": "Normal", 
            "icon": "🟢"
        }
    }
    
    return clinical_data.get(cell_class, {
        "color": "#6C63FF",
        "riesgo": "Desconocido",
        "icon": "⚪"
    })

def get_friendly_name(cell_class):
    """
    Obtiene el nombre amigable para una clase de célula
    """
    friendly_names = {
        "dyskeratotic": "Células Displásicas",
        "koilocytotic": "Células Koilocitóticas",
        "metaplastic": "Células Metaplásicas",
        "parabasal": "Células Parabasales",
        "superficial_intermediate": "Células Superficiales-Intermedias"
    }
    
    return friendly_names.get(cell_class, cell_class)

def calculate_consensus(predictions):
    """
    Calcula el consenso entre múltiples modelos
    
    Args:
        predictions: Diccionario con predicciones de cada modelo
    
    Returns:
        Diccionario con información del consenso
    """
    if not predictions:
        return None
    
    # Contar votos por clase
    class_votes = {}
    total_confidence = {}
    
    for model_name, pred in predictions.items():
        cell_class = pred['class']
        confidence = pred['confidence']
        
        if cell_class not in class_votes:
            class_votes[cell_class] = 0
            total_confidence[cell_class] = 0
        
        class_votes[cell_class] += 1
        total_confidence[cell_class] += confidence
    
    # Encontrar la clase con más votos
    consensus_class = max(class_votes, key=class_votes.get)
    consensus_votes = class_votes[consensus_class]
    avg_confidence = total_confidence[consensus_class] / consensus_votes
    
    # Determinar nivel de acuerdo
    total_models = len(predictions)
    agreement_ratio = consensus_votes / total_models
    
    if agreement_ratio == 1.0:
        agreement_level = "Unánime"
    elif agreement_ratio >= 0.67:
        agreement_level = "Mayoría fuerte"
    elif agreement_ratio >= 0.5:
        agreement_level = "Mayoría simple"
    else:
        agreement_level = "Sin consenso"
    
    return {
        'class': consensus_class,
        'class_friendly': get_friendly_name(consensus_class),
        'votes': consensus_votes,
        'total_models': total_models,
        'agreement_ratio': agreement_ratio,
        'agreement_level': agreement_level,
        'average_confidence': avg_confidence,
        'clinical_info': get_clinical_info_for_class(consensus_class)
    }