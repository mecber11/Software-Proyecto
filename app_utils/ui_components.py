"""
Componentes de UI reutilizables para la aplicación SIPaKMeD
"""
import streamlit as st
import pandas as pd

def load_custom_css():
    """Carga los estilos CSS personalizados"""
    import os
    
    # Obtener la ruta base del proyecto
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Desde utils/ ir a ISO-Final/
    css_path = os.path.join(current_dir, "static", "styles.css")
    
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        # CSS cargado exitosamente (sin mensaje para no saturar la UI)
    except FileNotFoundError:
        st.warning(f"⚠️ No se encontró el archivo de estilos CSS en: {css_path}")
    except Exception as e:
        st.error(f"❌ Error cargando estilos CSS: {e}")

def professional_card(content, title=None, color=None, center=False):
    """Crea una tarjeta profesional con contenido"""
    alignment = "text-align: center;" if center else ""
    title_html = f"<h4 style='color: {color or 'var(--primary-color)'}; margin-bottom: 1rem;'>{title}</h4>" if title else ""
    
    return f"""
    <div class="professional-card" style="{alignment}">
        {title_html}
        {content}
    </div>
    """

def metric_card(icon_label, value, sublabel, color="var(--primary-color)"):
    """Crea una tarjeta de métrica estilizada"""
    return f"""
    <div class="metric-card" style="padding: 1.5rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon_label}</div>
        <div style="font-size: 2rem; font-weight: 700; color: {color};">{value}</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">{sublabel}</div>
    </div>
    """

def status_badge(text, badge_type="normal"):
    """Crea un badge de estado"""
    badge_classes = {
        "normal": "status-normal",
        "warning": "status-warning", 
        "danger": "status-danger"
    }
    
    css_class = badge_classes.get(badge_type, "status-normal")
    return f'<div class="status-badge {css_class}">{text}</div>'

def display_header(title, subtitle):
    """Muestra el header principal de la aplicación"""
    st.markdown(f"""
    <div class="main-header">
        <div style="position: relative; z-index: 1;">
            {title}
            <div style="font-size: 1.2rem; font-weight: 400; margin-top: 1rem; opacity: 0.9;">
                {subtitle}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_metrics_row(metrics_data):
    """Muestra una fila de métricas"""
    cols = st.columns(len(metrics_data))
    
    for col, (icon_label, value, sublabel) in zip(cols, metrics_data):
        with col:
            st.markdown(
                metric_card(icon_label, value, sublabel),
                unsafe_allow_html=True
            )

def display_system_ready_message(t_func):
    """Muestra mensaje de sistema listo"""
    st.markdown(f"""
    <div class="analysis-card">
        <h2 style="color: var(--primary-color); margin-bottom: 1.5rem; text-align: center; font-weight: 700;">
            🎯 {t_func('system_ready')}
        </h2>
        <p style="color: var(--text-secondary); line-height: 1.8; text-align: center; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            {t_func('upload_description')}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_waiting_message(t_func):
    """Muestra mensaje de espera para imagen"""
    st.markdown(f"""
    <div class="analysis-card" style="text-align: center;">
        <h2 style="color: var(--primary-color); margin-bottom: 2rem; font-weight: 700;">
            📸 {t_func('waiting_image')}
        </h2>
        <p style="color: var(--text-secondary); line-height: 1.8; max-width: 600px; margin: 0 auto 2rem auto; font-size: 1.1rem;">
            {t_func('upload_description')}
        </p>
        <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border-radius: 16px; border: 2px solid var(--primary-color);">
            <p style="color: var(--primary-color); font-weight: 600; margin: 0; font-size: 1rem;">
                💡 {t_func('tip_quality')}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_image_info(image_info):
    """Muestra información de la imagen en formato tabla"""
    info_df = pd.DataFrame([
        ["📄 Archivo", image_info['filename']],
        ["📐 Dimensiones", image_info['size']], 
        ["🎨 Formato", image_info['format']],
        ["🔧 Modo", image_info['mode']]
    ], columns=["Propiedad", "Valor"])
    
    st.dataframe(info_df, use_container_width=True, hide_index=True)

def display_model_results_cards(predictions, clinical_info, t_func):
    """Muestra cards de resultados por modelo"""
    cols = st.columns(len(predictions))
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        with cols[i]:
            pred_clinical_info = pred['clinical_info']
            
            # Determinar color según riesgo
            risk_colors = {
                t_func('high_risk'): "#f56565",
                t_func('moderate_risk'): "#ed8936", 
                t_func('low_risk'): "#667eea",
                t_func('normal_risk'): "#48bb78"
            }
            
            color = risk_colors.get(pred_clinical_info['riesgo'], "#667eea")
            
            # Determinar clase de status badge
            if pred_clinical_info['riesgo'] in [t_func('normal_risk'), t_func('low_risk')]:
                status_class = 'status-normal'
            elif pred_clinical_info['riesgo'] == t_func('moderate_risk'):
                status_class = 'status-warning'
            else:
                status_class = 'status-danger'
            
            st.markdown(f"""
            <div class="professional-card" style="text-align: center; position: relative; overflow: hidden;">
                <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                            color: white; padding: 1.5rem; 
                            border-radius: 16px; margin: -2.5rem -2.5rem 1.5rem -2.5rem; position: relative;">
                    <h3 style="margin: 0; font-weight: 700; font-size: 1.2rem;">{model_name}</h3>
                    <div style="position: absolute; top: -5px; right: 10px; font-size: 2rem; opacity: 0.3;">🤖</div>
                </div>
                <div style="margin: 1.5rem 0;">
                    <div style="font-size: 3rem; font-weight: 800; color: {color}; margin: 0.5rem 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        {pred['confidence']:.1%}
                    </div>
                    <p style="color: var(--text-secondary); font-weight: 600; font-size: 1.1rem; margin: 1rem 0;">
                        {pred['class_friendly']}
                    </p>
                </div>
                <div class="status-badge {status_class}" style="margin-top: 1rem;">
                    {pred_clinical_info['icon']} {pred_clinical_info['riesgo']}
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_error_message(t_func):
    """Muestra mensaje de error para modelos"""
    st.error(t_func("model_error"))
    st.markdown(f"""
    <div class="warning-box-professional" style="background: linear-gradient(135deg, #FC424A 0%, #FF6B6B 100%);">
        <h4>🚨 {t_func('model_error')}</h4>
        <p>{t_func('model_error_solution')}</p>
        <ul style="margin: 1rem 0; padding-left: 1.5rem;">
            <li>{t_func('verify_files')}</li>
            <li>{t_func('run_training')}</li>
            <li>{t_func('restart_app')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def display_footer(t_func):
    """Muestra el footer de la aplicación"""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); 
                color: white; padding: 3rem 2rem; margin: 3rem -3rem -3rem -3rem; text-align: center; position: relative; overflow: hidden;'>
        <div style="position: relative; z-index: 1;">
            <h3 style='font-size: 1.5rem; margin-bottom: 1rem; font-weight: 700;'>
                {t_func('footer_title')}
            </h3>
            <p style='font-size: 1rem; margin-bottom: 1.5rem; opacity: 0.9;'>
                {t_func('footer_subtitle')}
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🧠</span>
                    <span>Deep Learning</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🔬</span>
                    <span>Análisis Médico</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">📊</span>
                    <span>Estadísticas</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">🎯</span>
                    <span>Alta Precisión</span>
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1.5rem; margin-top: 2rem;">
                <p style='font-size: 0.9rem; opacity: 0.8; margin: 0;'>
                    {t_func('footer_disclaimer')}
                </p>
            </div>
        </div>
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><pattern id=\"grain\" width=\"100\" height=\"100\" patternUnits=\"userSpaceOnUse\"><circle cx=\"50\" cy=\"50\" r=\"1\" fill=\"white\" opacity=\"0.05\"/></pattern></defs><rect width=\"100\" height=\"100\" fill=\"url(%23grain)\"/></svg>'); pointer-events: none;"></div>
    </div>
    """, unsafe_allow_html=True)