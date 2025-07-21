"""
Descargador automático de modelos para GitHub Codespaces y despliegues remotos
"""
import os
import requests
import streamlit as st
from pathlib import Path
import logging
import zipfile
import json
from typing import Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)

# URLs de modelos - TODOS los 5 modelos (3 clásicos + 2 híbridos)
MODEL_URLS = {
    # Modelos Clásicos TensorFlow (.h5)
    "MobileNetV2": {
        "url": "https://drive.google.com/file/d/1hTw08J2jBp_uKczIdgNF5UNKQPHh7f4g/view?usp=drive_link",
        "filename": "sipakmed_MobileNetV2.h5",
        "size_mb": 18,
        "type": "tensorflow",
        "framework": "TensorFlow"
    },
    "ResNet50": {
        "url": "https://drive.google.com/file/d/15TdMz4Pk7M37SjeaYujQryLljLvmhmfG/view?usp=drive_link",
        "filename": "sipakmed_ResNet50.h5", 
        "size_mb": 104,
        "type": "tensorflow",
        "framework": "TensorFlow"
    },
    "EfficientNetB0": {
        "url": "https://drive.google.com/file/d/1pYvOmxNWHN2x8XYce_OK4C39z-fxTfo7/view?usp=drive_link",
        "filename": "sipakmed_EfficientNetB0.h5",
        "size_mb": 25,
        "type": "tensorflow",
        "framework": "TensorFlow"
    },
    
    # Modelos Híbridos PyTorch (.pth)
    "HybridEnsemble": {
        "url": "https://drive.google.com/file/d/1LQczwM8EuVArccE6UgInoe-FjYfY6D7I/view?usp=drive_link",
        "filename": "ensemble_best.pth",
        "size_mb": 173,  # Tamaño aproximado
        "type": "pytorch",
        "framework": "PyTorch",
        "architecture": "HybridEnsemble",
        "description": "Fusión inteligente ResNet50+MobileNetV2+EfficientNet con atención"
    },
    "HybridMultiScale": {
        "url": "https://drive.google.com/file/d/172JYXrEPl2cT6msR_QIj1TUArwf2U5YQ/view?usp=drive_link", 
        "filename": "multiscale_best.pth",
        "size_mb": 65,   # Tamaño aproximado
        "type": "pytorch", 
        "framework": "PyTorch",
        "architecture": "HybridMultiScale",
        "description": "Arquitectura multi-escala con atención espacial"
    }
}

# URLs alternativos (Hugging Face Hub - recomendado)
HUGGINGFACE_MODELS = {
    # Modelos Clásicos
    "MobileNetV2": "JDAVIDT97/sipakmed-mobilenetv2",
    "ResNet50": "JDAVIDT97/sipakmed-resnet50", 
    "EfficientNetB0": "JDAVIDT97/sipakmed-efficientnetb0",
    
    # Modelos Híbridos  
    "HybridEnsemble": "JDAVIDT97/sipakmed-hybrid-ensemble",
    "HybridMultiScale": "JDAVIDT97/sipakmed-hybrid-multiscale"
}

def get_models_directory(model_type="tensorflow"):
    """Obtener directorio donde guardar modelos según el tipo"""
    # Directorio base
    if os.path.exists("/workspaces"):
        # Estamos en Codespaces
        base_dir = Path("/workspaces/sipakmed-web")
    else:
        # Local o otro ambiente
        base_dir = Path(".")
    
    # Subdirectorios según tipo de modelo
    if model_type == "tensorflow":
        models_dir = base_dir / "data" / "models"
    elif model_type == "pytorch": 
        models_dir = base_dir / "hybrid_training_results" / "models"
    else:
        models_dir = base_dir / "models"  # Fallback
    
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def calculate_md5(file_path: str) -> str:
    """Calcular MD5 hash de un archivo"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file_with_progress(url: str, destination: Path, filename: str, size_mb: int) -> bool:
    """Descargar archivo con barra de progreso"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Crear barra de progreso en Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(destination / filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Descargando {filename}: {downloaded//1024//1024:.1f}/{size_mb} MB")
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ {filename} descargado exitosamente")
        
        return True
        
    except Exception as e:
        logger.error(f"Error descargando {filename}: {e}")
        st.error(f"Error descargando {filename}: {e}")
        return False

def download_from_huggingface(model_name: str, destination: Path) -> bool:
    """Descargar modelo desde Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        repo_id = HUGGINGFACE_MODELS.get(model_name)
        if not repo_id:
            return False
        
        filename = MODEL_URLS[model_name]["filename"]
        
        st.info(f"Descargando {model_name} desde Hugging Face...")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(destination.parent),
            force_download=False  # Usar cache si existe
        )
        
        # Copiar a nuestro directorio
        import shutil
        shutil.copy2(downloaded_path, destination / filename)
        
        st.success(f"✅ {model_name} descargado desde Hugging Face")
        return True
        
    except ImportError:
        logger.warning("huggingface_hub no instalado, usando descarga directa")
        return False
    except Exception as e:
        logger.error(f"Error descargando de Hugging Face: {e}")
        return False

def check_model_integrity(model_path: Path, expected_md5: str) -> bool:
    """Verificar integridad del modelo usando MD5"""
    if not model_path.exists():
        return False
    
    try:
        actual_md5 = calculate_md5(str(model_path))
        return actual_md5 == expected_md5
    except Exception as e:
        logger.error(f"Error verificando integridad: {e}")
        return False

def download_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """Descargar un modelo específico (TensorFlow o PyTorch)"""
    if model_name not in MODEL_URLS:
        logger.error(f"Modelo desconocido: {model_name}")
        return None
    
    model_info = MODEL_URLS[model_name]
    model_type = model_info.get("type", "tensorflow")
    models_dir = get_models_directory(model_type)
    model_path = models_dir / model_info["filename"]
    
    # Verificar si el modelo ya existe y es válido
    if model_path.exists() and not force_download:
        # Verificar integridad si tenemos MD5
        if model_info.get("md5"):
            if check_model_integrity(model_path, model_info["md5"]):
                logger.info(f"Modelo {model_name} ya existe y es válido")
                return model_path
            else:
                logger.warning(f"Modelo {model_name} corrupto, re-descargando...")
        else:
            logger.info(f"Modelo {model_name} existe (sin verificación MD5)")
            return model_path
    
    # Intentar descargar desde Hugging Face primero (más confiable)
    if download_from_huggingface(model_name, models_dir):
        return model_path
    
    # Fallback: descarga directa
    st.warning(f"Descargando {model_name} ({model_info['size_mb']} MB)...")
    
    if download_file_with_progress(
        model_info["url"], 
        models_dir, 
        model_info["filename"], 
        model_info["size_mb"]
    ):
        return model_path
    
    return None

def download_all_models(models_to_download: List[str] = None) -> Dict[str, Optional[Path]]:
    """Descargar múltiples modelos"""
    if models_to_download is None:
        models_to_download = list(MODEL_URLS.keys())
    
    results = {}
    
    st.info("🔄 Descargando modelos necesarios...")
    
    for model_name in models_to_download:
        with st.expander(f"Descarga {model_name}", expanded=True):
            model_path = download_model(model_name)
            results[model_name] = model_path
            
            if model_path:
                st.success(f"✅ {model_name} listo")
            else:
                st.error(f"❌ Error descargando {model_name}")
    
    return results

def setup_models_for_codespaces():
    """Configuración automática para GitHub Codespaces"""
    st.markdown("## 🔄 Configuración de Modelos")
    
    # Detectar si estamos en Codespaces
    is_codespaces = os.path.exists("/workspaces") or os.getenv("CODESPACES") == "true"
    
    if is_codespaces:
        st.info("🚀 GitHub Codespaces detectado. Configurando modelos automáticamente...")
    else:
        st.info("💻 Ambiente local detectado.")
    
    models_dir = get_models_directory()
    
    # Verificar modelos existentes
    existing_models = []
    missing_models = []
    
    for model_name, info in MODEL_URLS.items():
        model_path = models_dir / info["filename"]
        if model_path.exists():
            existing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    if existing_models:
        st.success(f"✅ Modelos encontrados: {', '.join(existing_models)}")
    
    if missing_models:
        st.warning(f"⚠️ Modelos faltantes: {', '.join(missing_models)}")
        
        if st.button("📥 Descargar Modelos Faltantes", type="primary"):
            with st.spinner("Descargando modelos..."):
                results = download_all_models(missing_models)
                
                success_count = sum(1 for path in results.values() if path is not None)
                total_count = len(results)
                
                if success_count == total_count:
                    st.success(f"🎉 ¡Todos los modelos descargados exitosamente!")
                    st.balloons()
                    return True
                else:
                    st.error(f"⚠️ Descargados {success_count}/{total_count} modelos")
                    return False
    else:
        st.success("🎉 ¡Todos los modelos están listos!")
        return True
    
    return len(existing_models) > 0

def create_model_config_file():
    """Crear archivo de configuración con URLs actualizables"""
    config = {
        "model_urls": MODEL_URLS,
        "huggingface_models": HUGGINGFACE_MODELS,
        "version": "1.0",
        "last_updated": "2025-07-21"
    }
    
    config_path = Path("model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuración de modelos guardada en {config_path}")

def load_model_config():
    """Cargar configuración de modelos desde archivo"""
    config_path = Path("model_config.json")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Actualizar URLs globales
            global MODEL_URLS, HUGGINGFACE_MODELS
            MODEL_URLS.update(config.get("model_urls", {}))
            HUGGINGFACE_MODELS.update(config.get("huggingface_models", {}))
            
            logger.info("Configuración de modelos cargada")
            return True
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
    
    return False

# Inicializar configuración al importar
load_model_config()
