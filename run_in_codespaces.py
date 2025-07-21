#!/usr/bin/env python3
"""
Script optimizado para ejecutar SIPaKMeD en GitHub Codespaces
Maneja descarga automática de modelos y configuración del ambiente
"""

import streamlit as st
import os
import sys
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar utils al path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def detect_environment():
    """Detectar el ambiente de ejecución"""
    if os.path.exists("/workspaces") or os.getenv("CODESPACES") == "true":
        return "codespaces"
    elif os.getenv("STREAMLIT_SHARING") == "1":
        return "streamlit_cloud"
    else:
        return "local"

def setup_codespaces_environment():
    """Configurar ambiente específico para Codespaces"""
    
    st.markdown("# 🚀 SIPaKMeD en GitHub Codespaces")
    st.markdown("### Configurando ambiente para análisis de células cervicales...")
    
    # Importar herramientas de descarga
    try:
        from app_utils.model_downloader import setup_models_for_codespaces
        
        # Configurar modelos
        models_ready = setup_models_for_codespaces()
        
        if models_ready:
            st.success("✅ Ambiente configurado correctamente")
            st.markdown("### 🎯 La aplicación está lista para usar")
            st.markdown("**Para iniciar la aplicación, ejecuta este comando en el terminal:**")
            st.code("streamlit run SIPaKMeD-Web/app_optimized.py", language="bash")
            st.markdown("O si estás en la carpeta SIPaKMeD-Web:")
            st.code("streamlit run app_optimized.py", language="bash")
            
            st.info("💡 **Tip:** Detén este script (Ctrl+C) y ejecuta el comando de arriba para abrir la aplicación principal.")
        else:
            st.warning("⚠️ Es necesario descargar los modelos para continuar")
            
    except ImportError as e:
        st.error(f"Error importando módulos: {e}")
        st.info("Asegúrate de instalar las dependencias con: `pip install -r requirements.txt`")

def main():
    """Función principal"""
    
    # Configurar página
    st.set_page_config(
        page_title="SIPaKMeD Setup - GitHub Codespaces",
        page_icon="🔬",
        layout="wide"
    )
    
    # Detectar ambiente
    env = detect_environment()
    
    if env == "codespaces":
        setup_codespaces_environment()
    elif env == "streamlit_cloud":
        st.error("❌ Streamlit Cloud no soporta modelos pesados. Usa GitHub Codespaces.")
        st.markdown("### Alternativas:")
        st.markdown("- 🚀 Usa [GitHub Codespaces](https://github.com/codespaces)")
        st.markdown("- 💻 Ejecuta localmente con la versión SIPaKMeD-Training")
    else:
        # Ambiente local - redirigir directamente a la app
        st.info("💻 Ambiente local detectado. Ejecutando aplicación principal...")
        
        # Importar y ejecutar app principal
        try:
            exec(open("app_optimized.py").read())
        except FileNotFoundError:
            st.error("app_optimized.py no encontrado. Asegúrate de estar en el directorio correcto.")

if __name__ == "__main__":
    main()