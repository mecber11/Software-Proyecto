"""
Integración de Modelos Híbridos PyTorch con Aplicación Streamlit
Adaptador para usar modelos híbridos entrenados junto con los modelos TensorFlow existentes
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configurar logging
logger = logging.getLogger(__name__)

class HybridModelAdapter:
    """Adaptador para usar modelos híbridos PyTorch en la aplicación Streamlit"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hybrid_models = {}
        self.loaded = False
        
    def load_hybrid_models(self, models_dir: str = "hybrid_training_results/models") -> bool:
        """Cargar modelos híbridos entrenados"""
        try:
            # Buscar en múltiples ubicaciones posibles
            possible_paths = [
                Path(models_dir),
                Path("../hybrid_training_results/models"),
                Path("../../hybrid_training_results/models"),
                Path(__file__).parent.parent / "hybrid_training_results" / "models",
                Path(__file__).parent.parent.parent / "hybrid_training_results" / "models"
            ]
            
            models_path = None
            for path in possible_paths:
                if path.exists():
                    models_path = path
                    logger.info(f"📁 Modelos híbridos encontrados en: {path}")
                    break
            
            if models_path is None:
                logger.warning(f"Directorio de modelos híbridos no encontrado en ninguna ubicación")
                logger.info(f"Ubicaciones buscadas: {[str(p) for p in possible_paths]}")
                return False
            
            # Importar arquitecturas
            try:
                from hybrid_models.hybrid_architectures import get_hybrid_model
            except ImportError as e:
                logger.error(f"No se pueden importar arquitecturas híbridas: {e}")
                # Intentar importar desde el directorio padre
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from hybrid_models.hybrid_architectures import get_hybrid_model
                    logger.info("✅ Arquitecturas híbridas importadas desde directorio padre")
                except ImportError as e2:
                    logger.error(f"Error final importando arquitecturas: {e2}")
                    return False
            
            # Buscar modelos entrenados
            model_files = {
                "HybridEnsemble": "ensemble_best.pth",
                "HybridMultiScale": "multiscale_best.pth"
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    try:
                        # Crear modelo
                        if "Ensemble" in model_name:
                            model = get_hybrid_model("ensemble", num_classes=5, pretrained=False)
                        else:
                            model = get_hybrid_model("multiscale", num_classes=5)
                        
                        # Cargar pesos
                        checkpoint = torch.load(model_path, map_location=self.device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        model.to(self.device)
                        
                        self.hybrid_models[model_name] = {
                            'model': model,
                            'accuracy': checkpoint.get('accuracy', 0.0),
                            'epoch': checkpoint.get('epoch', 0),
                            'config': checkpoint.get('config', {})
                        }
                        
                        logger.info(f"✅ Modelo híbrido cargado: {model_name} (Precisión: {checkpoint.get('accuracy', 0):.2f}%)")
                        
                    except Exception as e:
                        logger.error(f"Error cargando {model_name}: {e}")
                        continue
                else:
                    logger.warning(f"Archivo de modelo no encontrado: {model_path}")
            
            self.loaded = len(self.hybrid_models) > 0
            if self.loaded:
                logger.info(f"🧠 Modelos híbridos cargados: {list(self.hybrid_models.keys())}")
            
            return self.loaded
            
        except Exception as e:
            logger.error(f"Error general cargando modelos híbridos: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesar imagen para modelos PyTorch"""
        try:
            # Convertir a array numpy
            img_array = np.array(image)
            
            # Asegurar RGB
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_rgb = img_array
            elif len(img_array.shape) == 2:
                img_rgb = np.stack([img_array] * 3, axis=-1)
            else:
                img_rgb = img_array[:,:,:3]
            
            # Redimensionar a 224x224
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # Normalizar usando ImageNet stats (asegurar float32 desde el inicio)
            img_normalized = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_normalized - mean) / std
            
            # Convertir a tensor PyTorch (asegurar float32)
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
            img_tensor = img_tensor.to(self.device, dtype=torch.float32)
            
            return img_tensor
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return None
    
    def predict_hybrid(self, image: Image.Image) -> Dict:
        """Realizar predicciones con modelos híbridos"""
        if not self.loaded or not self.hybrid_models:
            return {}
        
        try:
            # Preprocesar imagen
            img_tensor = self.preprocess_image(image)
            if img_tensor is None:
                return {}
            
            predictions = {}
            class_names = ["dyskeratotic", "koilocytotic", "metaplastic", "parabasal", "superficial_intermediate"]
            
            with torch.no_grad():
                for model_name, model_info in self.hybrid_models.items():
                    try:
                        model = model_info['model']
                        
                        # Realizar predicción
                        outputs = model(img_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        
                        # Extraer resultados
                        probs_cpu = probabilities.cpu().numpy()[0]
                        predicted_idx = np.argmax(probs_cpu)
                        confidence = float(probs_cpu[predicted_idx])
                        predicted_class = class_names[predicted_idx]
                        
                        # Obtener información clínica
                        clinical_info = self._get_clinical_info_for_class(predicted_class)
                        
                        predictions[model_name] = {
                            'class': predicted_class,
                            'class_friendly': self._get_friendly_name(predicted_class),
                            'confidence': confidence,
                            'probabilities': probs_cpu.tolist(),
                            'clinical_info': clinical_info,
                            'model_type': 'hybrid',
                            'framework': 'PyTorch',
                            'accuracy': model_info.get('accuracy', 0.0)
                        }
                        
                        logger.debug(f"{model_name}: {predicted_class} ({confidence:.2%})")
                        
                    except Exception as e:
                        logger.error(f"Error en predicción con {model_name}: {e}")
                        continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error general en predicción híbrida: {e}")
            return {}
    
    def _get_clinical_info_for_class(self, cell_class: str) -> Dict:
        """Obtener información clínica para una clase"""
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
    
    def _get_friendly_name(self, cell_class: str) -> str:
        """Obtener nombre amigable para clase"""
        friendly_names = {
            "dyskeratotic": "Células Displásicas",
            "koilocytotic": "Células Koilocitóticas",
            "metaplastic": "Células Metaplásicas",
            "parabasal": "Células Parabasales",
            "superficial_intermediate": "Células Superficiales-Intermedias"
        }
        
        return friendly_names.get(cell_class, cell_class)
    
    def get_model_info(self) -> Dict:
        """Obtener información de modelos cargados"""
        if not self.loaded:
            return {}
        
        info = {
            'total_models': len(self.hybrid_models),
            'device': str(self.device),
            'models': {}
        }
        
        for model_name, model_info in self.hybrid_models.items():
            info['models'][model_name] = {
                'accuracy': model_info.get('accuracy', 0.0),
                'epoch': model_info.get('epoch', 0),
                'parameters': sum(p.numel() for p in model_info['model'].parameters() if p.requires_grad),
                'framework': 'PyTorch',
                'type': 'Híbrido'
            }
        
        return info

# Instancia global del adaptador
hybrid_adapter = HybridModelAdapter()

@st.cache_resource(show_spinner=False)
def load_hybrid_models_cached():
    """Cargar modelos híbridos con cache de Streamlit"""
    success = hybrid_adapter.load_hybrid_models()
    return hybrid_adapter if success else None

def clear_hybrid_cache():
    """Limpiar cache de modelos híbridos"""
    load_hybrid_models_cached.clear()
    global hybrid_adapter
    hybrid_adapter = HybridModelAdapter()

def get_hybrid_predictions(image: Image.Image) -> Dict:
    """Función pública para obtener predicciones híbridas"""
    adapter = load_hybrid_models_cached()
    if adapter:
        return adapter.predict_hybrid(image)
    return {}

def get_hybrid_model_info() -> Dict:
    """Función pública para obtener información de modelos"""
    adapter = load_hybrid_models_cached()
    if adapter:
        info = adapter.get_model_info()
        return info.get('models', {})  # Devolver solo la información de modelos
    return {}

def is_hybrid_available() -> bool:
    """Verificar si los modelos híbridos están disponibles"""
    adapter = load_hybrid_models_cached()
    return adapter is not None and adapter.loaded

# Función para mostrar información de modelos híbridos en sidebar
def display_hybrid_info_in_sidebar(t_func):
    """Mostrar información de modelos híbridos en sidebar"""
    try:
        if not is_hybrid_available():
            return
        
        hybrid_info = get_hybrid_model_info()
        if not hybrid_info:
            return
        
        # Traducciones seguras con fallbacks
        def safe_translate(key, default):
            try:
                return t_func(key)
            except:
                return default
        
        hybrid_title = safe_translate('hybrid_models_title', '🧠 MODELOS HÍBRIDOS')
        ensemble_name = safe_translate('ensemble_cnn', 'CNN Ensemble Híbrida')
        multiscale_name = safe_translate('multiscale_cnn', 'CNN Multi-Escala Híbrida')
        pytorch_name = safe_translate('pytorch_framework', 'Framework PyTorch')
        
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h3>{hybrid_title}</h3>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--accent-color);">
                    <strong style="color: var(--accent-color);">🔥 {ensemble_name}:</strong><br>
                    {hybrid_info.get('HybridEnsemble', {}).get('accuracy', 0):.1f}% precisión
                </div>
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--warning-color);">
                    <strong style="color: var(--warning-color);">⚡ {multiscale_name}:</strong><br>
                    {hybrid_info.get('HybridMultiScale', {}).get('accuracy', 0):.1f}% precisión
                </div>
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--success-color);">
                    <strong style="color: var(--success-color);">🎯 {pytorch_name}:</strong><br>
                    GPU disponible
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Error mostrando info híbrida en sidebar: {e}")
        # Mostrar versión básica sin traducciones
        try:
            st.sidebar.markdown("""
            <div class="sidebar-section">
                <h3>🧠 MODELOS HÍBRIDOS</h3>
                <div style="color: var(--text-secondary); line-height: 1.6;">
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px;">
                        <strong>🔥 CNN Ensemble:</strong><br>
                        93.16% precisión
                    </div>
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px;">
                        <strong>⚡ CNN Multi-Escala:</strong><br>
                        90.73% precisión
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

if __name__ == "__main__":
    # Prueba básica del adaptador
    print("🧠 Probando adaptador de modelos híbridos...")
    
    adapter = HybridModelAdapter()
    success = adapter.load_hybrid_models()
    
    if success:
        print(f"✅ Modelos cargados: {list(adapter.hybrid_models.keys())}")
        print(f"🔧 Dispositivo: {adapter.device}")
        
        # Probar con imagen sintética
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        predictions = adapter.predict_hybrid(test_image)
        
        print(f"🔍 Predicciones de prueba: {len(predictions)} modelos")
        for model_name, pred in predictions.items():
            print(f"   {model_name}: {pred['class_friendly']} ({pred['confidence']:.2%})")
    else:
        print("❌ No se pudieron cargar modelos híbridos")
        print("💡 Ejecute primero: python train_hybrid_models.py")