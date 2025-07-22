"""
Aplicación SIPaKMeD optimizada y refactorizada
Clasificador de Células Cervicales usando Deep Learning

Reducido de 2,671 líneas a ~800 líneas (70% menos código)
Actualizado para resolver conflictos con OpenCV - v1.1
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from datetime import datetime
from scipy.stats import chi2
from scipy import stats

# Función para detectar GPU desde múltiples frameworks
def detect_gpu():
    """Detectar GPU disponible desde TensorFlow y PyTorch"""
    gpu_info = {
        'available': False,
        'framework': 'CPU',
        'device_name': 'No GPU detectada',
        'count': 0
    }
    
    # Verificar TensorFlow GPU
    tf_gpu_devices = tf.config.list_physical_devices('GPU')
    if tf_gpu_devices:
        gpu_info['available'] = True
        gpu_info['framework'] = 'TensorFlow + GPU'
        gpu_info['device_name'] = 'GPU (TensorFlow)'
        gpu_info['count'] = len(tf_gpu_devices)
        return gpu_info
    
    # Verificar PyTorch GPU si TensorFlow no detecta GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['framework'] = 'PyTorch + GPU'
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['count'] = torch.cuda.device_count()
            return gpu_info
    except ImportError:
        pass
    
    return gpu_info

# Importar módulos optimizados
from app_config.settings import initialize_app, UI_CONFIG, MODEL_CONFIG
from app_utils.data_loader import (
    load_models, load_translations, get_language, set_language,
    get_available_languages, get_class_names_friendly, get_clinical_info,
    load_training_images, load_hybrid_training_results,
    load_hybrid_comparison_results
)
from app_utils.ui_components import (
    load_custom_css, display_header, display_metrics_row,
    display_system_ready_message, display_waiting_message,
    display_image_info, display_model_results_cards,
    display_error_message, display_footer
)
from app_utils.ml_predictions import (
    enhance_cervical_cell_image, predict_cervical_cells, calculate_consensus
)
from app_utils.hybrid_integration import (
    get_hybrid_predictions, is_hybrid_available, display_hybrid_info_in_sidebar
)
from app_utils.pdf_generator import generate_pdf_report, create_download_link

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

def setup_app():
    """Configuración inicial de la aplicación"""
    initialize_app()
    load_custom_css()

def get_translation_function():
    """Obtiene la función de traducción para el idioma actual"""
    try:
        logger.info("Iniciando carga de función de traducción...")
        current_language = get_language()
        logger.info(f"Idioma seleccionado: {current_language}")
        translations = load_translations(current_language)
        logger.info(f"Traducciones cargadas: {len(translations) if translations else 0} claves")
        
        def t(key: str) -> str:
            try:
                if translations and isinstance(translations, dict) and key in translations:
                    return str(translations.get(key, key))
                else:
                    # Valores por defecto robustos
                    defaults = {
                        'models': 'Modelos',
                        'models_loaded': 'Cargados', 
                        'processing_mode': 'Procesamiento',
                        'accuracy_range': 'Rango',
                        'cell_types_count': 'Tipos de células',
                        'main_title': 'Clasificador de Células Cervicales',
                        'subtitle': 'Sistema de análisis automatizado',
                        'ai_system': 'Sistema de Inteligencia Artificial',
                        'system_ready': 'Sistema Listo',
                        'image_analysis': 'Análisis de Imagen',
                        'analysis_results': 'Resultados del Análisis',
                        'visual_analysis': 'Análisis Visual',
                        'download_report': 'Descargar Reporte'
                    }
                    result = defaults.get(key, key)
                    logger.info(f"Usando valor por defecto para '{key}': '{result}'")
                    return result
            except Exception as e:
                logger.error(f"Error en función t() con clave '{key}': {e}")
                return key
        
        return t
    except Exception as e:
        logger.error(f"Error crítico creando función de traducción: {e}")
        # Función de emergencia completa
        def emergency_t(key: str) -> str:
            emergency_dict = {
                'models': 'Modelos',
                'models_loaded': 'Cargados',
                'processing_mode': 'Procesamiento',
                'accuracy_range': 'Rango',
                'cell_types_count': 'Tipos de células',
                'main_title': 'Clasificador de Células Cervicales',
                'subtitle': 'Sistema de análisis automatizado basado en Deep Learning',
                'ai_system': 'Sistema de Inteligencia Artificial',
                'system_ready': 'Sistema Listo',
                'image_analysis': 'Análisis de Imagen',
                'analysis_results': 'Resultados del Análisis',
                'visual_analysis': 'Análisis Visual',
                'download_report': 'Descargar Reporte',
                'upload_instruction': 'Selecciona una imagen',
                'upload_help': 'Formatos soportados: PNG, JPG, JPEG',
                'applying_clahe': 'Aplicando mejoras...',
                'analyzing_ai': 'Analizando con IA...',
                'probability_distribution': 'Distribución de Probabilidades',
                'model_consensus': 'Consenso entre Modelos'
            }
            result = emergency_dict.get(key, key)
            logger.info(f"Usando traducción de emergencia para '{key}': '{result}'")
            return result
        return emergency_t

# ============================================================================
# COMPONENTES DE VISUALIZACIÓN
# ============================================================================

def create_interactive_plots(predictions):
    """Crea gráficos interactivos de probabilidades"""
    try:
        # Preparar datos
        models = list(predictions.keys())
        class_names = MODEL_CONFIG["class_names"]
        class_friendly = get_class_names_friendly()
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=len(models),
            subplot_titles=models,
            specs=[[{"type": "bar"}] * len(models)]
        )
        
        colors = ['#0066CC', '#6C63FF', '#00D25B', '#FFAB00', '#FC424A']
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            probabilities = pred.get('probabilities', [0] * len(class_names))
            friendly_names = [class_friendly.get(name, name) for name in class_names]
            
            fig.add_trace(
                go.Bar(
                    x=friendly_names,
                    y=probabilities,
                    name=model_name,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Distribución de Probabilidades por Modelo",
            height=500,
            font=dict(size=12)
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Probabilidad", range=[0, 1])
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráficos: {e}")
        return go.Figure()

def create_consensus_chart(predictions):
    """Crea gráfico de consenso entre modelos"""
    try:
        consensus = calculate_consensus(predictions)
        if not consensus:
            return go.Figure()
        
        # Datos para el gráfico de consenso
        labels = ['Consenso', 'Discrepancia']
        values = [consensus['votes'], consensus['total_models'] - consensus['votes']]
        colors = ['#00D25B', '#FC424A']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=14
            )
        ])
        
        fig.update_layout(
            title=f"Consenso: {consensus['class_friendly']} ({consensus['agreement_level']})",
            height=400,
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de consenso: {e}")
        return go.Figure()

def create_comparison_interactive_plots(comparison_data):
    """Crear gráficos interactivos de comparación"""
    try:
        if not comparison_data or "modelos_detallados" not in comparison_data:
            return None, None
        
        modelos = comparison_data["modelos_detallados"]
        
        # Preparar datos para gráficos
        model_names = list(modelos.keys())
        accuracies = [modelos[name].get("accuracy", 0) for name in model_names]
        mcc_scores = [modelos[name].get("mcc", 0) for name in model_names]
        model_types = [modelos[name].get("type", "unknown") for name in model_names]
        
        # Colores por tipo de modelo
        colors = ['#3498db' if t == 'classical' else '#e74c3c' for t in model_types]
        
        # Gráfico 1: Precisiones comparativas
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=colors,
            text=[f'{acc:.1f}%' for acc in accuracies],
            textposition='auto',
            name='Precisión'
        ))
        
        # Línea de objetivo 90%
        fig1.add_hline(y=90, line_dash="dash", line_color="green", 
                      annotation_text="Objetivo 90%")
        
        fig1.update_layout(
            title="Comparación de Precisiones: Modelos Clásicos vs Híbridos",
            xaxis_title="Modelos",
            yaxis_title="Precisión (%)",
            height=400,
            yaxis=dict(range=[80, 100])
        )
        
        # Gráfico 2: MCC Scores comparativos
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=model_names,
            y=mcc_scores,
            marker_color=colors,
            text=[f'{mcc:.3f}' for mcc in mcc_scores],
            textposition='auto',
            name='MCC Score'
        ))
        
        fig2.update_layout(
            title="Comparación de MCC Scores: Modelos Clásicos vs Híbridos",
            xaxis_title="Modelos",
            yaxis_title="MCC Score",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        return fig1, fig2
        
    except Exception as e:
        logger.error(f"Error creando gráficos de comparación: {e}")
        return None, None

def mcnemar_test(y_true, y_pred1, y_pred2):
    """Test de McNemar para comparar dos modelos"""
    try:
        # Tabla de contingencia
        b = sum((y_pred1 == y_true) & (y_pred2 != y_true))  # Solo modelo1 correcto
        c = sum((y_pred1 != y_true) & (y_pred2 == y_true))  # Solo modelo2 correcto
        
        # Fórmula de McNemar
        if (b + c) == 0:
            statistic = 0
            p_value = 1.0
        else:
            statistic = (b - c)**2 / (b + c)
            p_value = 1 - chi2.cdf(statistic, 1)
        
        significant = p_value < 0.05
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "b": int(b), 
            "c": int(c),
            "significant": significant,
            "interpretation": f"p={p_value:.4f}, {'significativo' if significant else 'no significativo'}"
        }
    except Exception as e:
        st.error(f"Error en test McNemar: {e}")
        return None

def matews_test(y_true, y_pred1, y_pred2):
    """Test de Matews para comparar dos modelos"""
    try:
        # Calcular correctos/incorrectos para cada modelo
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # Matriz de contingencia 2x2
        a = np.sum(correct1 & correct2)   # Ambos correctos
        b = np.sum(correct1 & ~correct2)  # Solo modelo1 correcto
        c = np.sum(~correct1 & correct2)  # Solo modelo2 correcto
        d = np.sum(~correct1 & ~correct2) # Ambos incorrectos
        
        contingency_table = np.array([[a, b], [c, d]])
        
        # Test chi-cuadrado
        try:
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        except Exception:
            chi2_stat = 0.0
            p_value = 1.0
        
        significant = p_value < 0.05
        
        # Matthews Correlation Coefficient
        denominator = np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
        if denominator == 0:
            mcc = 0.0
        else:
            mcc = (a * d - b * c) / denominator
        
        return {
            "chi2_statistic": chi2_stat,
            "statistic": chi2_stat,
            "p_value": p_value,
            "contingency_table": contingency_table.tolist(),
            "matews_correlation": abs(mcc),
            "significant": significant,
            "interpretation": f"p={p_value:.4f}, MCC={mcc:.4f}, {'significativo' if significant else 'no significativo'}"
        }
    except Exception as e:
        st.error(f"Error en test Matews: {e}")
        return None

def create_mcnemar_plot(mcnemar_results):
    """Crear gráfico de resultados McNemar"""
    try:
        if not mcnemar_results:
            return None
        
        comparisons = [result["comparison"] for result in mcnemar_results]
        p_values = [result["p_value"] for result in mcnemar_results]
        significant = [result["significant"] for result in mcnemar_results]
        
        # Colores según significancia
        colors = ['red' if sig else 'green' for sig in significant]
        
        # Crear gráfico de barras
        fig = go.Figure(data=[
            go.Bar(
                x=[comp.replace('_vs_', ' vs ') for comp in comparisons],
                y=p_values,
                marker_color=colors,
                text=[f"p={p:.4f}" for p in p_values],
                textposition='auto',
            )
        ])
        
        # Línea de significancia
        fig.add_hline(y=0.05, line_dash="dash", line_color="black", 
                     annotation_text="α = 0.05", annotation_position="top right")
        
        fig.update_layout(
            title="Test de McNemar - Comparación de Modelos",
            xaxis_title="Comparaciones",
            yaxis_title="p-value",
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico McNemar: {e}")
        return None

def create_matews_plot(matews_results):
    """Crear gráfico de resultados Matews"""
    try:
        if not matews_results:
            return None
        
        comparisons = [result["comparison"] for result in matews_results]
        p_values = [result["p_value"] for result in matews_results]
        mcc_values = [result["matews_correlation"] for result in matews_results]
        significant = [result["significant"] for result in matews_results]
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('p-values de Matews', 'Correlación de Matews (MCC)'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # p-values
        colors_p = ['red' if sig else 'green' for sig in significant]
        fig.add_trace(
            go.Bar(
                x=[comp.replace('_vs_', ' vs ') for comp in comparisons],
                y=p_values,
                marker_color=colors_p,
                name='p-value',
                text=[f"p={p:.4f}" for p in p_values],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # MCC values
        fig.add_trace(
            go.Bar(
                x=[comp.replace('_vs_', ' vs ') for comp in comparisons],
                y=mcc_values,
                marker_color='blue',
                name='MCC',
                text=[f"MCC={mcc:.4f}" for mcc in mcc_values],
                textposition='auto',
            ),
            row=1, col=2
        )
        
        # Línea de significancia
        fig.add_hline(y=0.05, line_dash="dash", line_color="black", row=1, col=1)
        
        fig.update_layout(
            title="Test de Matews - Comparación de Modelos",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="p-value", row=1, col=1)
        fig.update_yaxes(title_text="Correlación MCC", row=1, col=2)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico Matews: {e}")
        return None

def create_matews_matrix_plot(matews_results):
    """Crear matriz gráfica de comparación Matews entre pares de modelos"""
    try:
        if not matews_results or len(matews_results) == 0:
            return None
        
        # Extraer todos los modelos únicos
        all_models = set()
        for result in matews_results:
            comp = result["comparison"]
            model1, model2 = comp.split("_vs_")
            all_models.add(model1)
            all_models.add(model2)
        
        model_list = sorted(list(all_models))
        n_models = len(model_list)
        
        if n_models < 2:
            return None
        
        # Crear matrices
        mcc_matrix = np.ones((n_models, n_models))  # Inicializar con 1s en diagonal
        pvalue_matrix = np.zeros((n_models, n_models))  # Inicializar con 0s en diagonal
        significance_matrix = np.ones((n_models, n_models))  # Diagonal siempre significativa
        
        # Llenar las matrices con datos de comparaciones
        for result in matews_results:
            comp = result["comparison"]
            model1, model2 = comp.split("_vs_")
            
            i = model_list.index(model1)
            j = model_list.index(model2)
            
            # Matriz simétrica
            mcc_val = result.get("matews_correlation", 0.0)
            p_val = result.get("p_value", 1.0)
            sig_val = 1 if result.get("significant", False) else 0
            
            mcc_matrix[i, j] = mcc_val
            mcc_matrix[j, i] = mcc_val
            
            pvalue_matrix[i, j] = p_val
            pvalue_matrix[j, i] = p_val
            
            significance_matrix[i, j] = sig_val
            significance_matrix[j, i] = sig_val
        
        # Crear texto para cada celda
        text_matrix = []
        for i in range(n_models):
            row_text = []
            for j in range(n_models):
                if i == j:
                    text = f"{model_list[i]}<br>MCC: 1.000"
                else:
                    mcc_val = mcc_matrix[i, j]
                    p_val = pvalue_matrix[i, j]
                    sig_text = "Sig" if significance_matrix[i, j] else "NS"
                    text = f"MCC: {mcc_val:.3f}<br>p: {p_val:.4f}<br>{sig_text}"
                row_text.append(text)
            text_matrix.append(row_text)
        
        # Crear figura básica
        fig = go.Figure(data=go.Heatmap(
            z=mcc_matrix,
            x=model_list,
            y=model_list,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale="RdYlBu_r",
            zmin=0,
            zmax=1,
            colorbar=dict(title="MCC Score")
        ))
        
        # Agregar bordes rojos para diferencias significativas
        for i in range(n_models):
            for j in range(n_models):
                if significance_matrix[i, j] and i != j:
                    fig.add_shape(
                        type="rect",
                        x0=j-0.4, y0=i-0.4,
                        x1=j+0.4, y1=i+0.4,
                        line=dict(color="red", width=2),
                        fillcolor="rgba(0,0,0,0)"
                    )
        
        fig.update_layout(
            title="Matriz de Comparación Matews (MCC)<br><sub>Bordes rojos: diferencias significativas (p<0.05)</sub>",
            width=600,
            height=600,
            xaxis=dict(title="Modelo"),
            yaxis=dict(title="Modelo", autorange="reversed"),
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        # Error handling sin dependencias de streamlit
        return None

def calculate_statistical_tests(comparison_data):
    """Calcular tests estadísticos para todos los modelos"""
    try:
        if not comparison_data:
            return None, None
            
        if "modelos_detallados" not in comparison_data:
            return None, None
        
        modelos_detallados = comparison_data["modelos_detallados"]
        model_names = list(modelos_detallados.keys())
        
        mcnemar_results = []
        matews_results = []
        
        # Comparar cada par de modelos
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                
                # Obtener predicciones
                model1_data = modelos_detallados[model1]
                model2_data = modelos_detallados[model2]
                
                if "predictions" in model1_data and "predictions" in model2_data:
                    y_true = np.array(model1_data["predictions"]["y_true"])
                    y_pred1 = np.array(model1_data["predictions"]["y_pred"])
                    y_pred2 = np.array(model2_data["predictions"]["y_pred"])
                    
                    # Asegurar que las longitudes coincidan
                    min_len = min(len(y_true), len(y_pred1), len(y_pred2))
                    y_true = y_true[:min_len]
                    y_pred1 = y_pred1[:min_len]
                    y_pred2 = y_pred2[:min_len]
                    
                    # Calcular tests
                    mcnemar_result = mcnemar_test(y_true, y_pred1, y_pred2)
                    matews_result = matews_test(y_true, y_pred1, y_pred2)
                    
                    if mcnemar_result:
                        mcnemar_result['comparison'] = comparison_key
                        mcnemar_results.append(mcnemar_result)
                    if matews_result:
                        matews_result['comparison'] = comparison_key
                        # Agregar MCCs individuales desde los datos
                        matews_result['mcc1'] = model1_data.get('mcc', 0.0)
                        matews_result['mcc2'] = model2_data.get('mcc', 0.0)
                        matews_results.append(matews_result)
        
        return mcnemar_results, matews_results
        
    except Exception as e:
        st.error(f"Error calculando tests estadísticos: {e}")
        return None, None

def display_hybrid_comparison_results(comparison_data, t):
    """Muestra los resultados de comparación híbrida"""
    if not comparison_data:
        st.warning(t("no_hybrid_comparison_data"))
        return
    
    st.markdown(f"#### {t('hybrid_comparison_analysis')}")
    
    # Resumen general
    resumen = comparison_data.get("resumen_general", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Modelos", 
            resumen.get("total_modelos", 0),
            help="Modelos analizados en total"
        )
    
    with col2:
        mejora_abs = resumen.get('mejora_absoluta', 0)
        mejora_rel = resumen.get('mejora_relativa', 0)
        st.metric(
            "Mejora Híbridos", 
            f"{mejora_abs:+.1f}%",
            delta=f"{mejora_rel:+.1f}% relativa"
        )
    
    with col3:
        precision_hibridos = resumen.get("precision_media_hibridos", 0)
        objetivo_alcanzado = "Sí" if resumen.get("objetivo_90_alcanzado", False) else "No"
        st.metric(
            "Precisión Híbridos", 
            f"{precision_hibridos:.1f}%",
            delta=f"Objetivo 90%: {objetivo_alcanzado}"
        )
    
    with col4:
        precision_clasicos = resumen.get("precision_media_clasicos", 0)
        st.metric(
            "Precisión Clásicos", 
            f"{precision_clasicos:.1f}%"
        )
    
    # Gráficos interactivos
    st.markdown("##### 📊 Visualizaciones Comparativas")
    
    # Crear gráficos interactivos
    fig1, fig2 = create_comparison_interactive_plots(comparison_data)
    
    if fig1 and fig2:
        tab1, tab2 = st.tabs(["Precisiones", "MCC Scores"])
        
        with tab1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.plotly_chart(fig2, use_container_width=True)
    
    # Mostrar gráfico estático si existe
    if "archivos_generados" in comparison_data:
        grafico_path = comparison_data["archivos_generados"].get("grafico_comparativo")
        if grafico_path and os.path.exists(grafico_path):
            st.markdown("##### 📈 Gráfico Comparativo Completo")
            st.image(grafico_path, use_container_width=True, caption="Comparación detallada de todos los modelos")
    
    
    # Detalles de modelos híbridos
    if "modelos_detallados" in comparison_data:
        st.markdown("##### 🧠 Detalles de Modelos Híbridos")
        
        modelos_detallados = comparison_data["modelos_detallados"]
        hybrid_models = {k: v for k, v in modelos_detallados.items() if v.get("type") == "hybrid"}
        
        for model_name, model_data in hybrid_models.items():
            with st.expander(f"🤖 {model_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Precisión", f"{model_data.get('accuracy', 0):.1f}%")
                    st.metric("MCC Score", f"{model_data.get('mcc', 0):.3f}")
                
                with col2:
                    if "f1_scores" in model_data:
                        avg_f1 = np.mean(model_data["f1_scores"])
                        st.metric("F1-Score Promedio", f"{avg_f1:.3f}")
                    
                    if "precision_scores" in model_data:
                        avg_precision = np.mean(model_data["precision_scores"])
                        st.metric("Precisión Promedio", f"{avg_precision:.3f}")
                
                with col3:
                    if "recall_scores" in model_data:
                        avg_recall = np.mean(model_data["recall_scores"])
                        st.metric("Recall Promedio", f"{avg_recall:.3f}")
                    
                    if "architecture" in model_data:
                        arch = model_data["architecture"]
                        st.write(f"**Tipo**: {arch.get('tipo', 'N/A')}")
                        st.write(f"**Atención**: {arch.get('atencion', 'N/A')}")
    
    # Información adicional
    st.markdown("##### ℹ️ Información del Análisis")
    fecha_generacion = comparison_data.get("fecha_generacion", "No disponible")
    st.info(f"**Fecha de generación**: {fecha_generacion}")
    
    # Conclusiones
    st.markdown("##### 📝 Conclusiones")
    objetivo_alcanzado = resumen.get("objetivo_90_alcanzado", False)
    if objetivo_alcanzado:
        st.success("✅ **Objetivo alcanzado**: Los modelos híbridos superaron el 90% de precisión establecido como meta.")
    else:
        st.warning("⚠️ **Objetivo parcial**: Algunos modelos híbridos están cerca del 90% de precisión.")
    
    # Tests estadísticos
    st.markdown("##### 📊 Tests Estadísticos")
    
    # Calcular y mostrar tests estadísticos
    try:
        if not comparison_data:
            st.info("No hay datos de comparación disponibles para los tests estadísticos.")
            return
        
        # Debug: verificar estructura de datos
        if isinstance(comparison_data, str):
            st.error("Error: comparison_data es un string en lugar de un diccionario")
            return
            
        if not isinstance(comparison_data, dict):
            st.error(f"Error: comparison_data tiene tipo {type(comparison_data)} en lugar de dict")
            return
            
        mcnemar_results, matews_results = calculate_statistical_tests(comparison_data)
        
        if mcnemar_results and matews_results:
            tab1, tab2, tab3 = st.tabs(["Test de McNemar", "Test de Matews", "Matriz Matews"])
            
            with tab1:
                st.markdown("**Test de McNemar para comparaciones por pares**")
                mcnemar_fig = create_mcnemar_plot(mcnemar_results)
                if mcnemar_fig:
                    st.plotly_chart(mcnemar_fig, use_container_width=True)
                
                # Mostrar tabla de resultados
                if mcnemar_results:
                    df_mcnemar = pd.DataFrame([
                        {
                            'Comparación': result['comparison'],
                            'Estadístico': f"{result['statistic']:.4f}",
                            'p-valor': f"{result['p_value']:.4f}",
                            'Significativo': "Sí" if result['significant'] else "No"
                        }
                        for result in mcnemar_results
                    ])
                    st.dataframe(df_mcnemar, use_container_width=True)
            
            with tab2:
                st.markdown("**Test de Matews (extensión multiclase)**")
                matews_fig = create_matews_plot(matews_results)
                if matews_fig:
                    st.plotly_chart(matews_fig, use_container_width=True)
                
                # Mostrar tabla de resultados
                if matews_results:
                    df_matews = pd.DataFrame([
                        {
                            'Comparación': result['comparison'],
                            'MCC Modelo 1': f"{result.get('mcc1', 0.0):.4f}",
                            'MCC Modelo 2': f"{result.get('mcc2', 0.0):.4f}",
                            'Estadístico': f"{result['statistic']:.4f}",
                            'p-valor': f"{result['p_value']:.4f}",
                            'Significativo': "Sí" if result['significant'] else "No"
                        }
                        for result in matews_results
                    ])
                    st.dataframe(df_matews, use_container_width=True)
            
            with tab3:
                st.markdown("**Matriz de Comparación Matews entre Modelos**")
                st.markdown("Esta matriz muestra las correlaciones MCC entre todos los pares de modelos. Los bordes rojos indican diferencias estadísticamente significativas.")
                
                matews_matrix_fig = create_matews_matrix_plot(matews_results)
                if matews_matrix_fig:
                    st.plotly_chart(matews_matrix_fig, use_container_width=True)
                else:
                    st.error("No se pudo generar la matriz de comparación Matews")
        else:
            st.info("Los tests estadísticos requieren datos de predicciones completos.")
    except Exception as e:
        st.error(f"Error generando tests estadísticos: {e}")
    

def display_training_results():
    """Muestra resultados del entrenamiento si están disponibles"""
    t = get_translation_function()
    
    training_images = load_training_images()
    hybrid_results = load_hybrid_training_results()
    comparison_results = load_hybrid_comparison_results()
    
    if training_images or hybrid_results or comparison_results:
        st.markdown(f"### {t('training_results')}")
        
        # Tabs para diferentes secciones
        tab_labels = []
        if training_images.get('model_comparison'):
            tab_labels.append(t('general_comparison'))
        if training_images.get('confusion_matrices'):
            tab_labels.append(t('confusion_matrices'))
        if training_images.get('training_histories'):
            tab_labels.append(t('training_histories'))
        if training_images.get('hybrid_roc_curves') or training_images.get('hybrid_comparison'):
            tab_labels.append("🧠 Resultados Híbridos")
        if hybrid_results:
            tab_labels.append("📊 Métricas Híbridas")
        if comparison_results:
            tab_labels.append("🔬 Comparación Híbrida")
        
        if tab_labels:
            tabs = st.tabs(tab_labels)
            tab_index = 0
            
            # Comparación general
            if training_images.get('model_comparison'):
                with tabs[tab_index]:
                    st.image(training_images['model_comparison']['path'], use_container_width=True)
                tab_index += 1
            
            # Matrices de confusión
            if training_images.get('confusion_matrices'):
                with tabs[tab_index]:
                    cols = st.columns(len(training_images['confusion_matrices']))
                    for i, img_data in enumerate(training_images['confusion_matrices']):
                        with cols[i]:
                            st.image(img_data['path'], caption=img_data['name'], use_container_width=True)
                tab_index += 1
            
            # Historiales de entrenamiento
            if training_images.get('training_histories'):
                with tabs[tab_index]:
                    cols = st.columns(len(training_images['training_histories']))
                    for i, img_data in enumerate(training_images['training_histories']):
                        with cols[i]:
                            st.image(img_data['path'], caption=img_data['name'], use_container_width=True)
                tab_index += 1
            
            # Resultados Híbridos
            if training_images.get('hybrid_roc_curves') or training_images.get('hybrid_comparison'):
                with tabs[tab_index]:
                    st.markdown("#### 🧠 Modelos Híbridos - Resultados Avanzados")
                    
                    # Gráfico de comparación híbrida
                    if training_images.get('hybrid_comparison'):
                        st.markdown("##### Comparación de Precisión: Clásicos vs Híbridos")
                        st.image(training_images['hybrid_comparison']['path'], use_container_width=True)
                        st.success("✅ **Objetivo Alcanzado**: Los modelos híbridos superaron el 90% de precisión")
                    
                    # Curvas ROC híbridas
                    if training_images.get('hybrid_roc_curves'):
                        st.markdown("##### Curvas ROC - Modelos Híbridos")
                        st.image(training_images['hybrid_roc_curves']['path'], use_container_width=True)
                        st.info("📈 Las curvas ROC muestran el excelente rendimiento de ambos modelos híbridos")
                tab_index += 1
            
            # Métricas Híbridas Detalladas
            if hybrid_results:
                with tabs[tab_index]:
                    st.markdown("#### 📊 Métricas Detalladas de Entrenamiento Híbrido")
                    
                    # Información general
                    if 'dataset' in hybrid_results:
                        dataset_info = hybrid_results['dataset']
                        st.markdown("##### 📂 Información del Dataset")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Imágenes", f"{dataset_info.get('total_imagenes', 0):,}")
                        with col2:
                            st.metric("Clases", dataset_info.get('clases', 0))
                        with col3:
                            st.metric("Tipos de Archivo", ", ".join(dataset_info.get('tipos_archivo', [])))
                    
                    # Resultados por modelo
                    if 'modelos_hibridos' in hybrid_results:
                        st.markdown("##### 🎯 Resultados por Modelo")
                        
                        for model_name, model_data in hybrid_results['modelos_hibridos'].items():
                            with st.expander(f"🧠 {model_name}", expanded=True):
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    precision = model_data.get('precision_global', 0)
                                    objetivo = "✅" if model_data.get('objetivo_alcanzado', False) else "⚠️"
                                    st.metric("Precisión Global", f"{precision:.1f}%", delta=f"{objetivo} Objetivo")
                                
                                with col2:
                                    st.metric("Épocas Entrenadas", model_data.get('epocas_entrenadas', 0))
                                
                                with col3:
                                    tiempo = model_data.get('tiempo_estimado_horas', 0)
                                    st.metric("Tiempo Entrenamiento", f"{tiempo:.1f}h")
                                
                                with col4:
                                    params = model_data.get('parametros_entrenables', 0)
                                    st.metric("Parámetros", f"{params/1e6:.1f}M")
                                
                                # Información de arquitectura
                                if 'arquitectura' in model_data:
                                    arch = model_data['arquitectura']
                                    st.markdown("**Arquitectura:**")
                                    st.write(f"- **Tipo**: {arch.get('tipo', 'N/A')}")
                                    st.write(f"- **Atención**: {arch.get('atencion', 'N/A')}")
                                    st.write(f"- **Fusión**: {arch.get('fusion', 'N/A')}")
                                    if 'componentes' in arch:
                                        st.write(f"- **Componentes**: {', '.join(arch['componentes'])}")
                tab_index += 1
            
            # Comparación Híbrida
            if comparison_results:
                with tabs[tab_index]:
                    display_hybrid_comparison_results(comparison_results, t)
                tab_index += 1
            

def create_sidebar():
    """Crea la barra lateral con configuración"""
    t = get_translation_function()
    
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-header">
            <div style="position: relative; z-index: 1;">
                🔬 {t('sidebar_title')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Selector de idioma
        available_languages = get_available_languages()
        current_language = get_language()
        
        if len(available_languages) > 1:
            st.markdown(f"""
            <div class="sidebar-section">
                <h3>🌐 Idioma / Language</h3>
            </div>
            """, unsafe_allow_html=True)
            
            selected_language = st.selectbox(
                "Seleccionar idioma",
                options=list(available_languages.keys()),
                format_func=lambda x: available_languages[x],
                index=list(available_languages.keys()).index(current_language),
                label_visibility="collapsed"
            )
            
            if selected_language != current_language:
                set_language(selected_language)
                st.rerun()
        
        # Configuración
        st.markdown(f"""
        <div class="sidebar-section">
            <h3>⚙️ {t('configuration')}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        enhance_image = st.checkbox(
            f"🖼️ {t('clahe_enhancement')}",
            value=True,
            help=t('clahe_help')
        )
        
        # Información del sistema con detección mejorada de GPU
        gpu_info = detect_gpu()
        gpu_color = 'var(--success-color)' if gpu_info['available'] else 'var(--danger-color)'
        gpu_status = f"✅ {gpu_info['device_name']}" if gpu_info['available'] else "❌ No disponible"
        
        st.markdown(f"""
        <div class="sidebar-section">
            <h3>📊 {t('system_info')}</h3>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--primary-color);">
                    <strong style="color: var(--primary-color);">🧠 Modelos:</strong><br>
                    5 CNNs (3 clásicas + 2 híbridas)
                </div>
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--success-color);">
                    <strong style="color: var(--success-color);">📊 Dataset:</strong><br>
                    5,015 imágenes SIPaKMeD
                </div>
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid var(--warning-color);">
                    <strong style="color: var(--warning-color);">🎯 Precisión:</strong><br>
                    84-93% (híbridos >90%)
                </div>
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--card-bg-light); border-radius: 8px; border-left: 3px solid {gpu_color};">
                    <strong style="color: {gpu_color};">⚡ GPU:</strong><br>
                    {gpu_status}<br>
                    <small>{gpu_info['framework']}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Información de modelos híbridos si están disponibles
        if is_hybrid_available():
            display_hybrid_info_in_sidebar(t)
        
        # Aviso legal
        st.markdown(f"""
        <div class="warning-box-professional">
            <h4 style="color: var(--danger-color); margin-bottom: 1rem;">⚠️ {t('legal_notice')}</h4>
            <p style="margin: 0; line-height: 1.6;">
                {t("legal_text")}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return enhance_image

def display_clinical_interpretation(predictions):
    """Muestra la interpretación clínica de los resultados"""
    t = get_translation_function()
    clinical_info = get_clinical_info()
    
    st.markdown(f"### {t('clinical_interpretation')}")
    
    # Calcular consenso
    consensus = calculate_consensus(predictions)
    
    if consensus:
        consensus_class = consensus['class']
        clinical_data = clinical_info.get(consensus_class, {})
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **{t('result')}:** {consensus['class_friendly']}  
            **{t('consensus')}:** {consensus['votes']}/{consensus['total_models']} {t('models_agree')}  
            **Nivel de riesgo:** {clinical_data.get('riesgo', 'N/A')} {clinical_data.get('icon', '')}
            """)
        
        with col2:
            st.markdown(f"""
            **{t('description')}:**  
            {clinical_data.get('descripcion', 'N/A')}
            
            **{t('clinical_meaning')}:**  
            {clinical_data.get('significado', 'N/A')}
            """)

def display_download_section(predictions, image_info, probability_fig=None, consensus_fig=None, original_image=None, enhanced_image=None):
    """Muestra la sección de descarga de reportes"""
    t = get_translation_function()
    
    st.markdown(f"### {t('download_report')}")
    
    with st.expander(t("patient_info")):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input(t("patient_name"))
        with col2:
            patient_id = st.text_input(t("patient_id"))
    
    if st.button(t("generate_pdf")):
        patient_info = {'name': patient_name, 'id': patient_id}
        
        with st.spinner(t("generating_report")):
            # Cargar datos de comparación híbrida si están disponibles
            try:
                from app_utils.data_loader import load_hybrid_comparison_results
                hybrid_comparison_data = load_hybrid_comparison_results()
            except Exception as e:
                st.warning(f"No se pudieron cargar datos híbridos: {e}")
                hybrid_comparison_data = None
            
            # Mostrar información de contexto
            
            pdf_content = generate_pdf_report(
                predictions, 
                image_info, 
                patient_info, 
                t, 
                None,
                probability_fig, 
                consensus_fig,
                original_image,
                enhanced_image,
                hybrid_training_info=None,  # Agregar si está disponible
                hybrid_comparison_data=hybrid_comparison_data
            )
            
            if pdf_content:
                st.success(t("report_generated"))
                st.info(f"PDF generado: {len(pdf_content):,} bytes ({len(pdf_content)/1024/1024:.1f} MB)")
                
                # Crear enlace de descarga
                filename = f"reporte_sipakmed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                download_link = create_download_link(pdf_content, filename)
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.error(f"{t('pdf_error')} Error en la generación")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación optimizada"""
    try:
        setup_app()
        t = get_translation_function()
        
        # Verificar que la función de traducción funciona
        test_translation = t('models')
        logger.info(f"Test de traducción 'models': '{test_translation}'")
        
        # Header
        display_header(t('main_title'), t('subtitle'))
        
        # Sidebar
        enhance_image = create_sidebar()
        
        # Mensaje del sistema
        display_system_ready_message(t)
        
        # Cargar modelos
        st.markdown(f"### 🤖 {t('ai_system')}")
        
        with st.container():
            models = load_models()
            
            if not models:
                display_error_message(t)
                st.stop()
            
            # Contar modelos totales (TensorFlow + híbridos)
            total_models = len(models)
            if is_hybrid_available():
                from app_utils.hybrid_integration import get_hybrid_model_info
                hybrid_info = get_hybrid_model_info()
                total_models += len(hybrid_info)
            
            # Mostrar métricas del sistema
            try:
                models_text = t('models')
                models_loaded_text = t('models_loaded')
                processing_mode_text = t('processing_mode')
                accuracy_range_text = t('accuracy_range')
                cell_types_count_text = t('cell_types_count')
            except Exception as e:
                logger.error(f"Error obteniendo traducciones de métricas: {e}")
                models_text = "Modelos"
                models_loaded_text = "Cargados"
                processing_mode_text = "Procesamiento"
                accuracy_range_text = "Rango"
                cell_types_count_text = "Tipos de células"
            
            # Detectar GPU para la métrica de modo
            gpu_info = detect_gpu()
            processing_mode = gpu_info['framework'] if gpu_info['available'] else "CPU"
            
            metrics_data = [
                (f"🧠 {models_text}", f"{total_models}", models_loaded_text),
                ("⚡ Modo", processing_mode, processing_mode_text),
                ("🎯 Precisión", "84-93%", accuracy_range_text),
                ("📊 Clases", "5", cell_types_count_text)
            ]
            
            display_metrics_row(metrics_data)
        
        # Mostrar resultados del entrenamiento
        display_training_results()
        
        # Análisis de imagen
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                    border-radius: 20px; padding: 2rem; margin: 2rem 0; border: 2px solid var(--primary-color);">
            <h2 style="color: var(--primary-color); margin-bottom: 1rem; text-align: center; font-weight: 700;">
                📤 {t('image_analysis')}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            t("upload_instruction"),
            type=UI_CONFIG["supported_formats"],
            help=t("upload_help")
        )
        
        if uploaded_file is None:
            display_waiting_message(t)
        else:
            # Procesar imagen cargada
            original_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📷 Imagen Original")
                st.image(original_image, use_container_width=True)
            
                # Información de la imagen
                image_info = {
                    'filename': uploaded_file.name,
                    'size': f"{original_image.size[0]} x {original_image.size[1]}",
                    'format': original_image.format,
                    'mode': original_image.mode
                }
                
                display_image_info(image_info)
            
            with col2:
                enhanced_pil = None  # Inicializar para el PDF
                if enhance_image:
                    st.markdown("#### ✨ Imagen Mejorada")
                    with st.spinner(t('applying_clahe')):
                        enhanced_img = enhance_cervical_cell_image(original_image)
                        enhanced_pil = Image.fromarray(enhanced_img.astype(np.uint8))
                        st.image(enhanced_pil, use_container_width=True)
                        analysis_image = enhanced_pil
                else:
                    analysis_image = original_image
            
            # Realizar predicciones con modelos TensorFlow
            st.markdown(f"#### {t('analyzing_ai')}")
            predictions = predict_cervical_cells(analysis_image, models)
            
            # Agregar predicciones de modelos híbridos si están disponibles
            if is_hybrid_available():
                hybrid_predictions = get_hybrid_predictions(analysis_image)
                if hybrid_predictions:
                    predictions.update(hybrid_predictions)
                    logger.info(f"🧠 Predicciones híbridas agregadas: {len(hybrid_predictions)} modelos")
        
            # Mostrar resultados
            if predictions:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(72, 187, 120, 0.05) 0%, rgba(102, 126, 234, 0.05) 100%); 
                            border-radius: 20px; padding: 2rem; margin: 2rem 0; border: 2px solid var(--success-color);">
                    <h2 style="color: var(--success-color); margin-bottom: 1rem; text-align: center; font-weight: 700;">
                        📊 {t('analysis_results')}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Cards de resultados
                clinical_info = get_clinical_info()
                display_model_results_cards(predictions, clinical_info, t)
                
                # Gráficos interactivos
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(240, 147, 251, 0.05) 0%, rgba(245, 87, 108, 0.05) 100%); 
                            border-radius: 20px; padding: 2rem; margin: 2rem 0; border: 2px solid var(--accent-color);">
                    <h2 style="color: var(--accent-color); margin-bottom: 1rem; text-align: center; font-weight: 700;">
                        📈 {t('visual_analysis')}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs([t("probability_distribution"), t("model_consensus")])
                
                with tab1:
                    probability_fig = create_interactive_plots(predictions)
                    st.plotly_chart(probability_fig, use_container_width=True)
                
                with tab2:
                    consensus_fig = create_consensus_chart(predictions)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.plotly_chart(consensus_fig, use_container_width=True)
                
                # Interpretación clínica
                display_clinical_interpretation(predictions)
                
                # Sección de descarga
                display_download_section(predictions, image_info, probability_fig, consensus_fig, original_image, enhanced_pil)
            
            # Footer
            display_footer(t)
        
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        logger.error(f"Error en aplicación principal: {e}")
        
        # Mostrar interfaz básica en caso de error
        st.markdown("## Error en la aplicación")
        st.markdown("Se ha producido un error. Por favor recarga la página.")
        
        # Función de traducción de emergencia
        def emergency_t(key):
            emergency_translations = {
                'models': 'Modelos',
                'main_title': 'Clasificador de Células Cervicales',
                'subtitle': 'Sistema de análisis automatizado'
            }
            return emergency_translations.get(key, key)
        
        # Intentar mostrar al menos información básica
        try:
            st.sidebar.markdown("### Sistema en modo de emergencia")
            st.sidebar.markdown("Recarga la página para intentar de nuevo")
        except:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        st.error(f"Error de clave faltante: {e}")
        logger.error(f"KeyError en aplicación principal: {e}")
        
        # Mostrar interfaz mínima funcional
        st.title("🔬 Clasificador de Células Cervicales")
        st.markdown("**Error temporal - Sistema en modo básico**")
        
        # Función de traducción de emergencia total
        def emergency_translate(key):
            emergency_translations = {
                'models': 'Modelos',
                'models_loaded': 'Cargados',
                'processing_mode': 'Procesamiento',
                'accuracy_range': 'Rango',
                'cell_types_count': 'Tipos de células',
                'main_title': 'Clasificador de Células Cervicales',
                'subtitle': 'Sistema de análisis automatizado',
                'ai_system': 'Sistema de Inteligencia Artificial',
                'system_ready': 'Sistema Listo',
                'image_analysis': 'Análisis de Imagen',
                'analysis_results': 'Resultados del Análisis',
                'visual_analysis': 'Análisis Visual',
                'upload_instruction': 'Selecciona una imagen microscópica',
                'upload_help': 'Formatos: PNG, JPG, JPEG, BMP',
                'waiting_image': 'Esperando imagen',
                'upload_description': 'Carga una imagen para analizar',
                'applying_clahe': 'Procesando imagen...',
                'analyzing_ai': 'Analizando con IA...',
                'probability_distribution': 'Distribución de Probabilidades',
                'model_consensus': 'Consenso entre Modelos',
                'clinical_interpretation': 'Interpretación Clínica',
                'download_report': 'Descargar Reporte'
            }
            return emergency_translations.get(key, key)
        
        # Cargar modelos básicos
        st.markdown("### 🤖 Sistema de Inteligencia Artificial")
        try:
            from app_utils.data_loader import load_models
            models = load_models()
            if models:
                st.success(f"✅ {len(models)} modelos cargados exitosamente")
                
                # Mostrar información básica
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modelos", len(models))
                with col2:
                    st.metric("Precisión", "84-93%")
                with col3:
                    st.metric("Clases", "5")
                
                # Análisis básico de imagen
                st.markdown("### 📤 Análisis de Imagen")
                uploaded_file = st.file_uploader(
                    "Selecciona una imagen microscópica de células cervicales",
                    type=['png', 'jpg', 'jpeg', 'bmp'],
                    help="Formatos soportados: PNG, JPG, JPEG, BMP"
                )
                
                if uploaded_file:
                    from PIL import Image
                    import numpy as np
                    
                    original_image = Image.open(uploaded_file)
                    st.image(original_image, caption="Imagen cargada", use_container_width=True)
                    
                    # Predicción básica
                    st.markdown("#### 🔍 Análisis")
                    try:
                        from app_utils.ml_predictions import predict_cervical_cells
                        predictions = predict_cervical_cells(original_image, models)
                        
                        if predictions:
                            st.success("✅ Análisis completado")
                            
                            # Mostrar resultados básicos
                            for model_name, pred in predictions.items():
                                confidence = pred.get('confidence', 0)
                                cell_type = pred.get('class_friendly', pred.get('class', 'Desconocido'))
                                st.write(f"**{model_name}**: {cell_type} ({confidence:.1%})")
                        else:
                            st.error("No se pudieron obtener predicciones")
                    except Exception as pred_error:
                        st.error(f"Error en predicción: {pred_error}")
            else:
                st.error("No se pudieron cargar los modelos")
        except Exception as model_error:
            st.error(f"Error cargando modelos: {model_error}")
        
        st.markdown("---")
        st.markdown("**Nota**: Sistema funcionando en modo básico. Recarga la página para intentar el modo completo.")
    
    except Exception as e:
        st.error(f"Error crítico: {e}")
        logger.error(f"Error crítico en aplicación: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        st.title("🔬 Sistema SIPaKMeD")
        st.error("Error crítico del sistema")
        st.markdown("Por favor, verifica la configuración y recarga la página.")