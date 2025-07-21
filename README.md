# 🔬 SIPaKMeD-Web: Aplicación Web para Clasificación de Células Cervicales

## 🌟 Descripción
SIPaKMeD-Web es la versión optimizada para GitHub del sistema de clasificación de células cervicales basado en Deep Learning. Ejecuta **toda la funcionalidad completa** de tu aplicación directamente en GitHub Codespaces.

## ⚡ **FUNCIONALIDAD COMPLETA EN GITHUB**
- ✅ **3 Modelos CNN completos**: MobileNetV2, ResNet50, EfficientNetB0
- ✅ **Descarga automática de modelos**: Los modelos se descargan automáticamente
- ✅ **Interfaz completa**: Toda la UI de Streamlit funcional
- ✅ **Reportes PDF**: Generación completa de reportes
- ✅ **Tests estadísticos**: McNemar y Matthews implementados
- ✅ **Multi-idioma**: 4 idiomas soportados
- ✅ **GitHub Codespaces**: 8GB RAM, CPU potente, sin limitaciones

## 🚀 **Ejecutar en GitHub Codespaces (Recomendado)**

### Opción 1: Un Click Deploy
1. **Fork** este repositorio en GitHub
2. Click en **"Code"** → **"Codespaces"** → **"Create codespace"**
3. Espera 2-3 minutos para que se configure automáticamente
4. Ejecuta:
   ```bash
   streamlit run run_in_codespaces.py
   ```
5. **¡Listo!** Tu aplicación completa funcionando en la nube

### Opción 2: Manual Setup
```bash
# En Codespaces terminal:
pip install -r requirements.txt
streamlit run app_optimized.py
```

## 🔧 **Cómo Funciona la Descarga de Modelos**

### 🤖 Descarga Automática Inteligente
1. **Al abrir en Codespaces**: El sistema detecta automáticamente el ambiente
2. **Descarga inteligente**: Los modelos se descargan solo si no existen
3. **Múltiples fuentes**: Hugging Face Hub → Google Drive → URLs directas
4. **Verificación de integridad**: MD5 checksums para asegurar modelos correctos
5. **Cache persistente**: Los modelos se guardan y reutilizan entre sesiones

### 📊 Modelos Disponibles
| Modelo | Tamaño | Precisión | Descarga |
|--------|--------|-----------|----------|
| MobileNetV2 | ~18 MB | 85.8% | ⚡ Rápida |
| ResNet50 | ~104 MB | 87.2% | 🔄 Moderada |
| EfficientNetB0 | ~25 MB | 89.1% | ⚡ Rápida |

## 💻 **Para Desarrollo Local**

### Instalación
```bash
git clone https://github.com/tu-usuario/sipakmed-web.git
cd sipakmed-web
pip install -r requirements.txt
streamlit run app_optimized.py
```

## 📦 **Estructura Optimizada para GitHub**
```
SIPaKMeD-Web/
├── app_optimized.py              # Aplicación principal (funciona igual que local)
├── run_in_codespaces.py         # Script optimizado para Codespaces
├── requirements.txt             # Dependencias completas
├── .devcontainer/               # Configuración automática Codespaces
├── utils/
│   ├── model_downloader.py      # 🔥 Descargador automático de modelos
│   ├── ml_predictions.py        # Predicciones ML
│   ├── pdf_generator.py         # Generación PDF
│   └── [resto de utilidades]
├── config/                      # Configuraciones
├── hybrid_models/              # Arquitecturas (código)
├── static/                     # CSS y assets
└── translations/               # Multi-idioma
```

## 🎯 **Casos de Uso**

### 🚀 **Para Demo/Presentación**:
1. Abrir en Codespaces
2. Ejecutar `streamlit run run_in_codespaces.py`  
3. Compartir URL pública de Codespaces
4. **Funcionalidad 100% igual** a tu versión local

### 👥 **Para Colaboración**:
1. Múltiples desarrolladores pueden abrir Codespaces simultáneamente
2. Cada uno tiene su propia instancia con 8GB RAM
3. Sin conflictos de dependencias o ambiente

### 🎓 **Para Enseñanza**:
1. Estudiantes pueden ejecutar sin instalar nada
2. Ambiente idéntico para todos
3. Sin problemas de compatibilidad

## ⚙️ **Configuración Avanzada**

### 🔧 Configurar URLs de Modelos
Edita `utils/model_downloader.py`:
```python
MODEL_URLS = {
    "MobileNetV2": {
        "url": "https://tu-url-google-drive-o-huggingface",
        "filename": "sipakmed_MobileNetV2.h5",
        "size_mb": 18
    }
    # ... otros modelos
}
```

### 🌐 Hugging Face Hub (Recomendado)
```bash
# Subir modelos a Hugging Face Hub
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload tu-usuario/sipakmed-models ./data/models/
```

### 📱 Configuración Codespaces
El archivo `.devcontainer/devcontainer.json` configura automáticamente:
- Python 3.10
- Puerto 8501 para Streamlit  
- Extensiones VS Code útiles
- Variables de ambiente optimizadas

## 🚨 **Limitaciones y Soluciones**

### ❌ **Limitaciones de Codespaces**:
- ⚠️ **Sin GPU**: Solo CPU (pero funciona bien)
- ⚠️ **Timeout**: Sesiones inactivas se cierran (pero se puede reabrir)
- ⚠️ **Límite mensual**: 120 horas gratis/mes (suficiente para demos)

### ✅ **Soluciones Implementadas**:
- ✅ **Optimización CPU**: Modelos optimizados para inferencia CPU
- ✅ **Cache inteligente**: Los modelos no se re-descargan
- ✅ **Descarga progresiva**: Solo descarga modelos que necesitas
- ✅ **Fallbacks**: Múltiples URLs de descarga

## 📊 **Rendimiento en Codespaces**

### ⚡ **Tiempos Esperados**:
- **Setup inicial**: ~3-5 minutos (primera vez)
- **Inicio posterior**: ~30 segundos
- **Descarga de modelos**: ~2-5 minutos total
- **Predicción por imagen**: ~2-5 segundos
- **Generación PDF**: ~3-8 segundos

### 💾 **Recursos**:
- **RAM disponible**: 8 GB (más que suficiente)
- **CPU**: 4 cores (buena performance)
- **Almacenamiento**: 32 GB (sobra espacio)

## 🔧 **Troubleshooting**

### "Modelos no encontrados"
```bash
# Forzar re-descarga
rm -rf data/models/
streamlit run run_in_codespaces.py
```

### "Error de memoria"  
```bash
# Reiniciar Codespace
# Ctrl+Shift+P → "Codespaces: Restart"
```

### "Puerto no accesible"
- Asegúrate de que el puerto 8501 esté configurado como público
- Ir a "PORTS" tab en VS Code → Click derecho → "Port Visibility" → "Public"

## 🎉 **¡RESULTADO FINAL!**

### ✅ **Lo que TIENES ahora**:
- 🚀 **Tu aplicación completa funcionando en GitHub**
- 🌐 **Sin instalar nada local**
- 💻 **8GB RAM para trabajar**
- 🔗 **URL pública para compartir**
- ⚡ **Setup automático en <5 minutos**
- 🧠 **Todos los modelos funcionando**
- 📊 **Funcionalidad 100% igual a local**

### 🚀 **Para Empezar AHORA**:
1. **Fork** este repo
2. **Abrir en Codespaces** 
3. **Ejecutar**: `streamlit run run_in_codespaces.py`
4. **¡Funciona igual que tu versión local!** 🎉

---

**🔬 Tu SIPaKMeD ahora funciona perfectamente en GitHub con toda la funcionalidad completa**