"""
Generador de reportes PDF para SIPaKMeD
"""
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import streamlit as st
import logging
import plotly.io as pio
import os
from PIL import Image

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generador de reportes PDF profesionales"""
    
    def __init__(self, translations_func):
        self.t = translations_func
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Configura estilos personalizados para el PDF"""
        # Intentar registrar fuentes Unicode para soporte de caracteres chinos
        try:
            # Usar fuentes del sistema que soporten Unicode
            # Estas son fuentes comúnmente disponibles en sistemas Windows/Linux/Mac
            font_candidates = [
                # Windows
                ('NotoSansCJK', 'C:/Windows/Fonts/NotoSansCJK-Regular.ttc'),
                ('SimHei', 'C:/Windows/Fonts/simhei.ttf'),
                ('SimSun', 'C:/Windows/Fonts/simsun.ttc'),
                # Linux
                ('NotoSansCJK', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                ('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                # Mac
                ('PingFangSC', '/System/Library/Fonts/PingFang.ttc'),
                ('STHeiti', '/System/Library/Fonts/STHeiti Light.ttc'),
            ]
            
            self.unicode_font = None
            for font_name, font_path in font_candidates:
                try:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        self.unicode_font = font_name
                        logger.info(f"Registrada fuente Unicode: {font_name}")
                        break
                except Exception as font_e:
                    logger.debug(f"No se pudo registrar fuente {font_name}: {font_e}")
                    continue
            
            if not self.unicode_font:
                # Fallback: usar fuentes incorporadas de ReportLab (limitadas)
                self.unicode_font = 'Helvetica'
                self.unicode_font_bold = 'Helvetica-Bold'
                logger.warning("No se encontraron fuentes Unicode, usando Helvetica (caracteres chinos pueden no mostrarse correctamente)")
            else:
                # Para fuentes Unicode personalizadas, usar la misma fuente para bold (la mayoría soporta bold)
                self.unicode_font_bold = self.unicode_font
                
        except Exception as e:
            logger.error(f"Error configurando fuentes Unicode: {e}")
            self.unicode_font = 'Helvetica'
            self.unicode_font_bold = 'Helvetica-Bold'
        
        # Estilo para títulos principales
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontName=self.unicode_font,
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0066CC')
        ))
        
        # Estilo para subtítulos
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontName=self.unicode_font,
            fontSize=14,
            spaceAfter=20,
            textColor=colors.HexColor('#333333')
        ))
        
        # Actualizar estilo Normal para usar fuente Unicode
        self.styles.add(ParagraphStyle(
            name='UnicodeNormal',
            parent=self.styles['Normal'],
            fontName=self.unicode_font,
            fontSize=10,
            spaceAfter=12
        ))
    
    def _create_unicode_table_style(self, base_style_list):
        """Crea un estilo de tabla con soporte Unicode agregando la fuente al inicio"""
        unicode_style = [('FONTNAME', (0, 0), (-1, -1), self.unicode_font)] + base_style_list
        return TableStyle(unicode_style)
    
    def generate_report(self, predictions, image_info, patient_info=None, statistical_results=None, probability_fig=None, consensus_fig=None, original_image=None, enhanced_image=None, hybrid_training_info=None, hybrid_comparison_data=None):
        """
        Genera un reporte PDF completo con gráficos e imágenes
        
        Args:
            predictions: Predicciones de los modelos
            image_info: Información de la imagen
            patient_info: Información del paciente (opcional)
            statistical_results: Resultados estadísticos (opcional)
            probability_fig: Gráfico de probabilidades (opcional)
            consensus_fig: Gráfico de consenso (opcional)
            original_image: Imagen original PIL (opcional)
            enhanced_image: Imagen mejorada PIL (opcional)
        
        Returns:
            bytes: Contenido del PDF
        """
        try:
            # Crear archivo temporal
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Header del reporte
            story.extend(self._create_header())
            
            # Información de la imagen y paciente
            story.extend(self._create_image_info(image_info))
            if patient_info:
                story.extend(self._create_patient_info(patient_info))
            
            # Incluir imágenes analizadas
            if original_image or enhanced_image:
                story.extend(self._create_analyzed_images(original_image, enhanced_image))
            
            # Resultados por modelo
            story.extend(self._create_model_results(predictions))
            
            # Interpretación clínica
            story.extend(self._create_clinical_interpretation(predictions))
            
            # Gráficos de análisis
            story.extend(self._create_analysis_charts(probability_fig, consensus_fig))
            
            # Recomendaciones
            story.extend(self._create_recommendations(predictions))
            
            # Análisis estadístico si está disponible
            if statistical_results:
                story.extend(self._create_statistical_analysis(statistical_results))
            
            # Información de entrenamiento híbrido si está disponible
            if hybrid_training_info:
                story.extend(self._create_hybrid_training_info(hybrid_training_info))
            
            # Tests estadísticos (McNemar y Matews) si están disponibles
            if hybrid_comparison_data:
                story.extend(self._create_statistical_tests_section(hybrid_comparison_data))
            
            # Gráficos de entrenamiento
            story.extend(self._create_training_charts())
            
            # Disclaimer
            story.extend(self._create_disclaimer())
            
            # Construir PDF
            doc.build(story)
            buffer.seek(0)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generando PDF: {e}")
            return None
    
    def _create_header(self):
        """Crea el header del reporte"""
        elements = []
        
        # Título principal
        title = Paragraph(self.t('pdf_title'), self.styles['CustomTitle'])
        elements.append(title)
        
        # Subtítulo
        subtitle = Paragraph(self.t('pdf_subtitle'), self.styles['CustomSubtitle'])
        elements.append(subtitle)
        
        # Fecha y sistema
        date_info = f"{self.t('analysis_date')} {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        system_info = f"{self.t('system')} SIPaKMeD AI v1.0"
        
        info_data = [
            [date_info, system_info]
        ]
        
        info_table = Table(info_data, colWidths=[3*inch, 3*inch])
        info_table.setStyle(self._create_unicode_table_style([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_image_info(self, image_info):
        """Crea la sección de información de imagen"""
        elements = []
        
        elements.append(Paragraph(self.t('analyzed_images_title'), self.styles['CustomSubtitle']))
        
        data = [
            [self.t('analyzed_image'), image_info.get('filename', 'N/A')],
            [self.t('dimensions'), image_info.get('size', 'N/A')],
            [self.t('format'), image_info.get('format', 'N/A')]
        ]
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(self._create_unicode_table_style([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_patient_info(self, patient_info):
        """Crea la sección de información del paciente"""
        elements = []
        
        if patient_info.get('name') or patient_info.get('id'):
            elements.append(Paragraph(self.t('patient_info'), self.styles['CustomSubtitle']))
            
            data = []
            if patient_info.get('name'):
                data.append([self.t('patient'), patient_info['name']])
            if patient_info.get('id'):
                data.append([self.t('id'), patient_info['id']])
            
            if data:
                table = Table(data, colWidths=[2*inch, 4*inch])
                table.setStyle(self._create_unicode_table_style([
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
                ]))
                
                elements.append(table)
                elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_analyzed_images(self, original_image, enhanced_image):
        """Crea la sección de imágenes analizadas"""
        elements = []
        
        elements.append(Paragraph(self.t('analyzed_images_title'), self.styles['CustomSubtitle']))
        
        try:
            # Crear tabla para mostrar las imágenes lado a lado
            images_data = []
            
            if original_image and enhanced_image:
                # Ambas imágenes disponibles
                # Convertir PIL a bytes para ReportLab
                original_buffer = io.BytesIO()
                enhanced_buffer = io.BytesIO()
                
                # Redimensionar imágenes para PDF (tamaño regular)
                original_resized = original_image.copy()
                enhanced_resized = enhanced_image.copy()
                
                # Calcular tamaño manteniendo aspecto (máximo 200x200 píxeles)
                max_size = (200, 200)
                original_resized.thumbnail(max_size, Image.Resampling.LANCZOS)
                enhanced_resized.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Guardar en buffers
                original_resized.save(original_buffer, format='PNG')
                enhanced_resized.save(enhanced_buffer, format='PNG')
                
                original_buffer.seek(0)
                enhanced_buffer.seek(0)
                
                # Crear imágenes para ReportLab (tamaño fijo en pulgadas)
                img_original = RLImage(original_buffer, width=2*inch, height=2*inch)
                img_enhanced = RLImage(enhanced_buffer, width=2*inch, height=2*inch)
                
                # Crear tabla con las imágenes
                images_data = [
                    [self.t('original_image'), self.t('enhanced_image')],
                    [img_original, img_enhanced]
                ]
                
            elif original_image:
                # Solo imagen original
                original_buffer = io.BytesIO()
                original_resized = original_image.copy()
                original_resized.thumbnail((200, 200), Image.Resampling.LANCZOS)
                original_resized.save(original_buffer, format='PNG')
                original_buffer.seek(0)
                
                img_original = RLImage(original_buffer, width=2*inch, height=2*inch)
                images_data = [
                    [self.t('original_image')],
                    [img_original]
                ]
            
            elif enhanced_image:
                # Solo imagen mejorada
                enhanced_buffer = io.BytesIO()
                enhanced_resized = enhanced_image.copy()
                enhanced_resized.thumbnail((200, 200), Image.Resampling.LANCZOS)
                enhanced_resized.save(enhanced_buffer, format='PNG')
                enhanced_buffer.seek(0)
                
                img_enhanced = RLImage(enhanced_buffer, width=2*inch, height=2*inch)
                images_data = [
                    [self.t('enhanced_image')],
                    [img_enhanced]
                ]
            
            if images_data:
                # Determinar ancho de columnas
                if len(images_data[0]) == 2:
                    col_widths = [3*inch, 3*inch]
                else:
                    col_widths = [3*inch]
                
                images_table = Table(images_data, colWidths=col_widths)
                images_table.setStyle(self._create_unicode_table_style([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('FONTWEIGHT', (0, 0), (-1, 0), 'BOLD'),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('TOPPADDING', (0, 1), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 10)
                ]))
                
                elements.append(images_table)
                elements.append(Spacer(1, 20))
                
        except Exception as e:
            logger.error(f"Error agregando imágenes al PDF: {e}")
            elements.append(Paragraph("Error al incluir imágenes analizadas", self.styles['UnicodeNormal']))
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_model_results(self, predictions):
        """Crea la sección de resultados por modelo"""
        elements = []
        
        elements.append(Paragraph(self.t('results_by_model'), self.styles['CustomSubtitle']))
        
        # Separar modelos clásicos y híbridos
        classic_models = {}
        hybrid_models = {}
        
        for model_name, pred in predictions.items():
            if pred.get('framework') == 'PyTorch' or 'Hybrid' in model_name:
                hybrid_models[model_name] = pred
            else:
                classic_models[model_name] = pred
        
        # Tabla de modelos clásicos
        if classic_models:
            elements.append(Paragraph(self.t('classical_models'), self.styles['Heading3']))
            
            data = [
                [self.t('model'), self.t('cell_type'), self.t('confidence'), self.t('risk_level')]
            ]
            
            for model_name, pred in classic_models.items():
                clinical_info = pred.get('clinical_info', {})
                data.append([
                    model_name,
                    pred.get('class_friendly', 'N/A'),
                    f"{pred.get('confidence', 0):.1%}",
                    clinical_info.get('riesgo', 'N/A')
                ])
            
            table = Table(data, colWidths=[1.5*inch, 2.5*inch, 1*inch, 1*inch])
            table.setStyle(self._create_unicode_table_style([
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 15))
        
        # Tabla de modelos híbridos
        if hybrid_models:
            elements.append(Paragraph(self.t('hybrid_models'), self.styles['Heading3']))
            
            data = [
                [self.t('model'), self.t('cell_type'), self.t('confidence'), self.t('risk_level'), self.t('accuracy')]
            ]
            
            for model_name, pred in hybrid_models.items():
                clinical_info = pred.get('clinical_info', {})
                accuracy = pred.get('accuracy', 0)
                data.append([
                    model_name,
                    pred.get('class_friendly', 'N/A'),
                    f"{pred.get('confidence', 0):.1%}",
                    clinical_info.get('riesgo', 'N/A'),
                    f"{accuracy:.1f}%" if accuracy > 0 else 'N/A'
                ])
            
            table = Table(data, colWidths=[1.2*inch, 2*inch, 0.8*inch, 0.8*inch, 1.2*inch])
            table.setStyle(self._create_unicode_table_style([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 10))
            
            # Agregar información técnica de modelos híbridos
            elements.append(Paragraph(self.t('hybrid_models_details'), self.styles['Heading4']))
            
            hybrid_info_text = """
            • <b>HybridEnsemble:</b> Fusión inteligente de ResNet50, MobileNetV2 y EfficientNet-B0 con mecanismos de atención<br/>
            • <b>HybridMultiScale:</b> Arquitectura personalizada con bloques multi-escala y atención espacial<br/>
            • <b>Preprocesamiento:</b> Normalización CLAHE, segmentación por rotación (±15°), zoom y desplazamientos<br/>
            • <b>Framework:</b> PyTorch con optimización CUDA para GPU RTX 4070<br/>
            • <b>Objetivo de precisión:</b> >90% (superando modelos clásicos)<br/>
            • <b>Augmentación avanzada:</b> 15+ técnicas de transformación de datos
            """
            
            elements.append(Paragraph(hybrid_info_text, self.styles['UnicodeNormal']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_hybrid_training_info(self, hybrid_training_info):
        """Crea la sección de información de entrenamiento híbrido"""
        if not hybrid_training_info:
            return []
        
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph(self.t('training_info_title'), self.styles['CustomSubtitle']))
        
        # Información general del entrenamiento
        if 'general_info' in hybrid_training_info:
            info = hybrid_training_info['general_info']
            
            general_text = f"""
            <b>Dataset utilizado:</b> SIPaKMeD (5,015 imágenes de células cervicales)<br/>
            <b>GPU utilizada:</b> {info.get('gpu_name', 'NVIDIA RTX 4070')}<br/>
            <b>Framework:</b> PyTorch con CUDA {info.get('cuda_version', '12.8')}<br/>
            <b>Preprocesamiento:</b> Normalización CLAHE, segmentación por rotación (±15°), zoom y desplazamientos<br/>
            <b>Augmentación:</b> 15+ técnicas avanzadas de transformación de datos<br/>
            <b>Objetivo de precisión:</b> >90% (superando modelos clásicos)
            """
            
            elements.append(Paragraph(general_text, self.styles['UnicodeNormal']))
            elements.append(Spacer(1, 15))
        
        # Tabla de tiempos de entrenamiento
        if 'training_times' in hybrid_training_info:
            elements.append(Paragraph("Tiempos de Entrenamiento por Modelo", self.styles['Heading3']))
            
            data = [
                ['Modelo', 'Épocas', 'Tiempo Total', 'Precisión Final', 'Objetivo Alcanzado']
            ]
            
            for model_name, times in hybrid_training_info['training_times'].items():
                objetivo_alcanzado = "✅ Sí" if times.get('final_accuracy', 0) >= 90 else "⚠️ No"
                data.append([
                    model_name,
                    f"{times.get('epochs', 0)}",
                    f"{times.get('total_hours', 0):.1f}h",
                    f"{times.get('final_accuracy', 0):.1f}%",
                    objetivo_alcanzado
                ])
            
            table = Table(data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1.2*inch])
            table.setStyle(self._create_unicode_table_style([
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 15))
        
        # Detalles técnicos
        if 'technical_details' in hybrid_training_info:
            elements.append(Paragraph("Detalles Técnicos del Entrenamiento", self.styles['Heading3']))
            
            details = hybrid_training_info['technical_details']
            details_text = f"""
            <b>Tamaño de batch:</b> {details.get('batch_size', 16)}<br/>
            <b>Tasa de aprendizaje inicial:</b> {details.get('learning_rate', 1e-4)}<br/>
            <b>Optimizador:</b> {details.get('optimizer', 'AdamW con weight decay')}<br/>
            <b>Función de pérdida:</b> {details.get('loss_function', 'CrossEntropyLoss con Label Smoothing')}<br/>
            <b>Scheduler:</b> {details.get('scheduler', 'ReduceLROnPlateau')}<br/>
            <b>Early Stopping:</b> {details.get('early_stopping', 'Activado (paciencia: 10 épocas)')}<br/>
            <b>Validation Split:</b> {details.get('validation_split', '20% del dataset')}<br/>
            <b>Memoria GPU utilizada:</b> {details.get('gpu_memory', '~8GB de 12GB disponibles')}
            """
            
            elements.append(Paragraph(details_text, self.styles['UnicodeNormal']))
            elements.append(Spacer(1, 15))
        
        # Comparación de rendimiento
        if 'performance_comparison' in hybrid_training_info:
            elements.append(Paragraph("Comparación de Rendimiento vs Modelos Clásicos", self.styles['Heading3']))
            
            comparison = hybrid_training_info['performance_comparison']
            
            data = [
                ['Tipo de Modelo', 'Modelo', 'Precisión', 'Mejora vs Clásicos']
            ]
            
            # Modelos clásicos
            for model, acc in comparison.get('classic_models', {}).items():
                data.append(['Clásico (TensorFlow)', model, f"{acc:.1f}%", 'Referencia'])
            
            # Modelos híbridos
            for model, info in comparison.get('hybrid_models', {}).items():
                mejora = info.get('accuracy', 0) - comparison.get('classic_average', 0)
                data.append([
                    'Híbrido (PyTorch)', 
                    model, 
                    f"{info.get('accuracy', 0):.1f}%", 
                    f"+{mejora:.1f}%"
                ])
            
            table = Table(data, colWidths=[1.3*inch, 1.5*inch, 1*inch, 1.2*inch])
            table.setStyle(self._create_unicode_table_style([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightyellow),
                ('BACKGROUND', (1, 1), (-1, -1), colors.lightblue),  # Clásicos
                ('BACKGROUND', (1, 4), (-1, -1), colors.lightgreen),  # Híbridos (asumiendo 3 clásicos)
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            
            elements.append(table)
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_clinical_interpretation(self, predictions):
        """Crea la sección de interpretación clínica"""
        elements = []
        
        elements.append(Paragraph(self.t('clinical_interpretation_title'), self.styles['CustomSubtitle']))
        
        # Obtener el consenso (clase más frecuente)
        class_counts = {}
        for pred in predictions.values():
            cell_class = pred.get('class', '')
            class_counts[cell_class] = class_counts.get(cell_class, 0) + 1
        
        if class_counts:
            consensus_class = max(class_counts, key=class_counts.get)
            consensus_pred = None
            for pred in predictions.values():
                if pred.get('class') == consensus_class:
                    consensus_pred = pred
                    break
            
            if consensus_pred:
                clinical_info = consensus_pred.get('clinical_info', {})
                
                result_text = f"""
                <b>{self.t('predominant_cell_type')}</b> {consensus_pred.get('class_friendly', 'N/A')}<br/>
                <b>{self.t('consensus_text')}</b> {class_counts[consensus_class]}/{len(predictions)} modelos<br/>
                <b>{self.t('risk_level')}:</b> {clinical_info.get('riesgo', 'N/A')}
                """
                
                elements.append(Paragraph(result_text, self.styles['UnicodeNormal']))
                elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_recommendations(self, predictions):
        """Crea la sección de recomendaciones"""
        elements = []
        
        elements.append(Paragraph(self.t('recommendations_title'), self.styles['CustomSubtitle']))
        
        # Recomendaciones generales
        recommendations = [
            self.t('consult_doctor'),
            self.t('pathologist_interpretation'),
            self.t('regular_followup')
        ]
        
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['UnicodeNormal']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_analysis_charts(self, probability_fig, consensus_fig):
        """Crea la sección de gráficos de análisis"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph(self.t('analysis_charts'), self.styles['CustomSubtitle']))
        
        # Gráfico de probabilidades
        if probability_fig:
            try:
                elements.append(Paragraph(self.t('probability_distribution_title'), self.styles['UnicodeNormal']))
                img_bytes = pio.to_image(probability_fig, format="png", width=600, height=400)
                img_temp = io.BytesIO(img_bytes)
                img = RLImage(img_temp, width=5*inch, height=3.3*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
            except Exception as e:
                logger.error(f"Error añadiendo gráfico de probabilidades: {e}")
                elements.append(Paragraph(self.t('chart_error'), self.styles['UnicodeNormal']))
        
        # Gráfico de consenso
        if consensus_fig:
            try:
                elements.append(Paragraph(self.t('model_consensus_title'), self.styles['UnicodeNormal']))
                img_bytes = pio.to_image(consensus_fig, format="png", width=400, height=400)
                img_temp = io.BytesIO(img_bytes)
                img = RLImage(img_temp, width=3*inch, height=3*inch)
                elements.append(img)
                elements.append(Spacer(1, 20))
            except Exception as e:
                logger.error(f"Error añadiendo gráfico de consenso: {e}")
                elements.append(Paragraph(self.t('chart_error'), self.styles['UnicodeNormal']))
        
        return elements
    
    def _create_statistical_analysis(self, statistical_results):
        """Crea la sección de análisis estadístico"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph(self.t('statistical_analysis_pdf'), self.styles['CustomSubtitle']))
        
        # Matthews Correlation Coefficient
        elements.append(Paragraph(self.t('mcc_full'), self.styles['CustomSubtitle']))
        elements.append(Paragraph(self.t('mcc_explanation'), self.styles['UnicodeNormal']))
        elements.append(Spacer(1, 12))
        
        mcc_scores = statistical_results.get('mcc_scores', {})
        if mcc_scores:
            # Crear tabla de MCC
            mcc_data = [[self.t("model"), 'MCC', self.t("interpretation")]]
            for model, mcc in sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True):
                interpretation = self.t("excellent") if mcc > 0.5 else self.t("good") if mcc > 0.3 else self.t("regular")
                mcc_data.append([model, f'{mcc:.4f}', interpretation])
            
            mcc_table = Table(mcc_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            mcc_table.setStyle(self._create_unicode_table_style([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(mcc_table)
        
        # Prueba de McNemar
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(self.t('mcnemar_full'), self.styles['CustomSubtitle']))
        elements.append(Paragraph(self.t('mcnemar_explanation'), self.styles['UnicodeNormal']))
        elements.append(Spacer(1, 12))
        
        mcnemar_results = statistical_results.get('mcnemar_tests', {})
        if mcnemar_results:
            # Crear tabla de McNemar para todos los modelos (incluyendo híbridos)
            all_models = list(set())
            for key in mcnemar_results.keys():
                models_in_key = key.split('_vs_')
                all_models.extend(models_in_key)
            all_models = sorted(list(set(all_models)))
            
            mcnemar_matrix_data = [['Comparación McNemar'] + all_models]
            
            for i, model1 in enumerate(all_models):
                row = [model1]
                for j, model2 in enumerate(all_models):
                    if i == j:
                        row.append('-')
                    else:
                        key1 = f"{model1}_vs_{model2}"
                        key2 = f"{model2}_vs_{model1}"
                        
                        result = None
                        if key1 in mcnemar_results:
                            result = mcnemar_results[key1]
                        elif key2 in mcnemar_results:
                            result = mcnemar_results[key2]
                        
                        if result:
                            p_val = result['p_value']
                            if p_val < 0.001:
                                cell_text = "p<0.001***"
                            elif p_val < 0.01:
                                cell_text = "p<0.01**"
                            elif p_val < 0.05:
                                cell_text = "p<0.05*"
                            else:
                                cell_text = f"p={p_val:.3f}"
                            row.append(cell_text)
                        else:
                            row.append('-')
                
                mcnemar_matrix_data.append(row)
            
            # Ajustar ancho de columnas según número de modelos
            col_width = 6.0 / len(all_models) * inch
            col_widths = [1.2*inch] + [col_width] * len(all_models)
            
            mcnemar_matrix = Table(mcnemar_matrix_data, colWidths=col_widths)
            mcnemar_matrix.setStyle(self._create_unicode_table_style([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('BACKGROUND', (0, 0), (0, -1), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 1), (-1, -1), colors.beige)
            ]))
            
            elements.append(mcnemar_matrix)
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(f"<i>NS: No significativo, *: p<0.05, **: p<0.01, ***: p<0.001</i>", self.styles['UnicodeNormal']))
        
        # Prueba de Mateos
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Prueba de Mateos (Multiclase)", self.styles['CustomSubtitle']))
        elements.append(Paragraph("La prueba de Mateos es una extensión multiclase de McNemar que evalúa diferencias significativas entre modelos considerando todas las clases simultáneamente.", self.styles['UnicodeNormal']))
        elements.append(Spacer(1, 12))
        
        mateos_results = statistical_results.get('mateos_tests', {})
        if mateos_results:
            # Crear tabla de Mateos
            mateos_data = [['Comparación', 'Chi-cuadrado', 'p-value', 'Significancia']]
            
            for comp_name, result in sorted(mateos_results.items()):
                model1, model2 = comp_name.split('_vs_')
                chi2_stat = result.get('chi2_statistic', 0)
                p_val = result.get('p_value', 1.0)
                significant = result.get('significant', False)
                
                significance_text = "Significativo" if significant else "No significativo"
                
                mateos_data.append([
                    f"{model1} vs {model2}",
                    f"{chi2_stat:.4f}",
                    f"{p_val:.4f}",
                    significance_text
                ])
            
            mateos_table = Table(mateos_data, colWidths=[2*inch, 1.2*inch, 1*inch, 1.3*inch])
            mateos_table.setStyle(self._create_unicode_table_style([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(mateos_table)
            elements.append(Spacer(1, 15))
        
        # Matriz de Correlación de Mateos
        correlation_matrix = statistical_results.get('mateos_correlation_matrix', {})
        if correlation_matrix:
            elements.append(Paragraph("Matriz de Correlación de Mateos", self.styles['CustomSubtitle']))
            elements.append(Paragraph("Coeficientes de correlación entre modelos basados en la prueba de Mateos. Valores cercanos a 1.0 indican comportamientos similares.", self.styles['UnicodeNormal']))
            elements.append(Spacer(1, 12))
            
            # Obtener modelos de la matriz
            models = list(correlation_matrix.keys())
            
            # Crear tabla de matriz de correlación
            matrix_data = [['Modelo'] + models]
            
            for model1 in models:
                row = [model1]
                for model2 in models:
                    coeff = correlation_matrix.get(model1, {}).get(model2, 0.0)
                    row.append(f"{coeff:.3f}")
                matrix_data.append(row)
            
            # Ajustar ancho de columnas
            col_width = 5.5 / len(models) * inch
            col_widths = [1*inch] + [col_width] * len(models)
            
            correlation_table = Table(matrix_data, colWidths=col_widths)
            correlation_table.setStyle(self._create_unicode_table_style([
                ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
                ('BACKGROUND', (0, 0), (0, -1), colors.orange),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 1), (-1, -1), colors.lightyellow)
            ]))
            
            elements.append(correlation_table)
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<i>Interpretación: 1.0 = idénticos, 0.8-0.9 = muy similares, 0.5-0.7 = moderadamente similares, <0.5 = diferentes</i>", self.styles['UnicodeNormal']))
        
        return elements
    
    def _create_training_charts(self):
        """Crea la sección de gráficos de entrenamiento"""
        elements = []
        
        # Obtener directorio base del proyecto
        current_dir = os.getcwd()
        
        # Lógica simplificada para detectar el directorio ISO-Final
        if current_dir.endswith('ISO-Final'):
            # Estamos dentro del directorio ISO-Final
            base_dir = current_dir
        else:
            # Buscar ISO-Final en subdirectorios
            potential_paths = [
                os.path.join(current_dir, 'ISO-Final'),
                os.path.join(current_dir, 'ISO-Final(optimizado)', 'ISO-Final')
            ]
            base_dir = None
            for path in potential_paths:
                if os.path.exists(path):
                    base_dir = path
                    break
            
            if base_dir is None:
                base_dir = os.path.join(current_dir, 'ISO-Final')  # Fallback
        
        # Buscar gráficos de entrenamiento clásicos con rutas absolutas
        training_figures = [
            (os.path.join(base_dir, "reports/figures/models_comparison.png"), "Comparación General de Modelos Clásicos"),
            (os.path.join(base_dir, "reports/figures/confusion_matrix_MobileNetV2.png"), "Matriz Confusión MobileNetV2"),
            (os.path.join(base_dir, "reports/figures/confusion_matrix_ResNet50.png"), "Matriz Confusión ResNet50"),
            (os.path.join(base_dir, "reports/figures/confusion_matrix_EfficientNetB0.png"), "Matriz Confusión EfficientNetB0")
        ]
        
        # Buscar gráficos de entrenamiento híbridos con rutas absolutas
        hybrid_figures = [
            (os.path.join(base_dir, "hybrid_training_results/figures/models_comparison_hybrid.png"), "Comparación de Modelos Híbridos vs Clásicos"),
            (os.path.join(base_dir, "hybrid_training_results/figures/confusion_matrix_HybridEnsemble.png"), "Matriz Confusión HybridEnsemble"),
            (os.path.join(base_dir, "hybrid_training_results/figures/confusion_matrix_HybridMultiScale.png"), "Matriz Confusión HybridMultiScale"),
            (os.path.join(base_dir, "hybrid_training_results/figures/roc_curves_hybrid_models.png"), "Curvas ROC - Modelos Híbridos"),
            (os.path.join(base_dir, "hybrid_training_results/figures/training_history_HybridEnsemble.png"), "Historial Entrenamiento HybridEnsemble"),
            (os.path.join(base_dir, "hybrid_training_results/figures/training_history_HybridMultiScale.png"), "Historial Entrenamiento HybridMultiScale")
        ]
        
        added_charts = False
        
        # Debug: Log de archivos encontrados
        logger.info(f"Buscando gráficos en directorio base: {base_dir}")
        
        # Agregar gráficos clásicos
        classic_found = 0
        for figure_path, title in training_figures:
            figure_path = figure_path.replace('/', os.sep)  # Normalizar separadores
            exists = os.path.exists(figure_path)
            logger.info(f"Archivo clásico {figure_path}: {'ENCONTRADO' if exists else 'NO ENCONTRADO'}")
            
            if exists:
                classic_found += 1
                if not added_charts:
                    elements.append(PageBreak())
                    elements.append(Paragraph("Graficos de Entrenamiento y Evaluacion", self.styles['CustomSubtitle']))
                    elements.append(Paragraph("Modelos Clasicos (TensorFlow)", self.styles['Heading3']))
                    added_charts = True
                
                try:
                    elements.append(Paragraph(title, self.styles['UnicodeNormal']))
                    img = RLImage(figure_path, width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 15))
                    logger.info(f"Gráfico clásico agregado exitosamente: {title}")
                except Exception as e:
                    logger.error(f"Error añadiendo gráfico {figure_path}: {e}")
        
        logger.info(f"Total gráficos clásicos agregados: {classic_found}")
        
        # Agregar gráficos híbridos
        hybrid_charts_added = False
        hybrid_found = 0
        
        for figure_path, title in hybrid_figures:
            figure_path = figure_path.replace('/', os.sep)  # Normalizar separadores
            exists = os.path.exists(figure_path)
            logger.info(f"Archivo híbrido {figure_path}: {'ENCONTRADO' if exists else 'NO ENCONTRADO'}")
            
            if exists:
                hybrid_found += 1
                if not hybrid_charts_added:
                    if not added_charts:
                        elements.append(PageBreak())
                        elements.append(Paragraph("Graficos de Entrenamiento y Evaluacion", self.styles['CustomSubtitle']))
                        added_charts = True
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph("Modelos Hibridos Avanzados (PyTorch)", self.styles['Heading3']))
                    hybrid_charts_added = True
                
                try:
                    elements.append(Paragraph(title, self.styles['UnicodeNormal']))
                    
                    # Ajustar tamaño según tipo de gráfico
                    if "comparison" in figure_path:
                        # Gráfico de comparación más grande
                        img = RLImage(figure_path, width=6*inch, height=4*inch)
                    elif "roc_curves" in figure_path:
                        # Curvas ROC tamaño estándar
                        img = RLImage(figure_path, width=5.5*inch, height=4*inch)
                    else:
                        # Matrices de confusión e historiales
                        img = RLImage(figure_path, width=5*inch, height=3.5*inch)
                    
                    elements.append(img)
                    elements.append(Spacer(1, 15))
                    logger.info(f"Gráfico híbrido agregado exitosamente: {title}")
                except Exception as e:
                    logger.error(f"Error añadiendo gráfico híbrido {figure_path}: {e}")
        
        logger.info(f"Total gráficos híbridos agregados: {hybrid_found}")
        
        # Agregar resumen si se encontraron gráficos híbridos
        if hybrid_charts_added:
            elements.append(Spacer(1, 10))
            hybrid_summary = """
            <b>Interpretacion de Graficos Hibridos:</b><br/>
            • <b>Comparacion General:</b> Muestra mejora de precision de modelos hibridos vs clasicos<br/>
            • <b>Matrices de Confusion:</b> Rendimiento detallado por clase de celulas cervicales<br/>
            • <b>Curvas ROC:</b> Capacidad discriminativa para cada clase (AUC cercano a 1.0 = excelente)<br/>
            • <b>Historiales:</b> Evolucion de precision y perdida durante entrenamiento<br/>
            • <b>Objetivo >90%:</b> Alcanzado por ambos modelos hibridos (93.2% y 90.7%)
            """
            elements.append(Paragraph(hybrid_summary, self.styles['UnicodeNormal']))
        
        total_found = classic_found + hybrid_found
        logger.info(f"Resumen final: {total_found} gráficos encontrados ({classic_found} clásicos, {hybrid_found} híbridos)")
        
        if not added_charts:
            logger.warning("No se encontraron gráficos de entrenamiento para incluir en el PDF")
            logger.info(f"Directorio base usado: {base_dir}")
            logger.info(f"Directorio de trabajo actual: {os.getcwd()}")
        
        return elements
    
    def _create_statistical_tests_section(self, hybrid_comparison_data):
        """Crea la sección de tests estadísticos McNemar y Matews"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Tests Estadísticos", self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        try:
            # Importar las funciones necesarias para los tests
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.append(parent_dir)
            
            from app_optimized import calculate_statistical_tests, create_mcnemar_matrix_plot, create_mcc_scores_plot
            
            # Calcular los tests estadísticos
            mcnemar_results, mcc_scores = calculate_statistical_tests(hybrid_comparison_data)
            
            if mcnemar_results:
                # Sección McNemar
                elements.append(Paragraph("Test de McNemar", self.styles['CustomSubtitle']))
                elements.append(Paragraph(
                    "El test de McNemar evalúa si existe una diferencia significativa entre las predicciones de dos modelos en el mismo conjunto de datos.",
                    self.styles['UnicodeNormal']
                ))
                elements.append(Spacer(1, 10))
                
                # Crear matriz McNemar
                mcnemar_fig = create_mcnemar_matrix_plot(mcnemar_results)
                if mcnemar_fig:
                    img_buffer = io.BytesIO()
                    mcnemar_fig.write_image(img_buffer, format='png', width=800, height=600)
                    img_buffer.seek(0)
                    
                    try:
                        # Usar buffer de memoria directamente en lugar de archivo temporal
                        img_buffer.seek(0)
                        img = RLImage(img_buffer, width=6*inch, height=4.5*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 20))
                    except Exception as e:
                        logger.error(f"Error agregando imagen McNemar al PDF: {e}")
                        elements.append(Paragraph("Error al cargar gráfico McNemar", self.styles['UnicodeNormal']))
                        elements.append(Spacer(1, 20))
                
                # Tabla de resultados McNemar
                mcnemar_data = [['Comparación', 'Ganador', 'p-valor', 'Significativo', 'Interpretación']]
                for result in mcnemar_results:
                    winner = result.get('winner', 'empate')
                    if winner == 'modelo1':
                        winner_text = result['model1']
                    elif winner == 'modelo2':
                        winner_text = result['model2']
                    else:
                        winner_text = 'Empate'
                    
                    interpretation = 'Diferencia significativa' if result['significant'] else 'Sin diferencia significativa'
                    
                    mcnemar_data.append([
                        result['comparison'].replace('_vs_', ' vs '),
                        winner_text,
                        f"{result['p_value']:.4f}",
                        "Sí" if result['significant'] else "No",
                        interpretation
                    ])
                
                mcnemar_table = Table(mcnemar_data)
                mcnemar_table.setStyle(self._create_unicode_table_style([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), self.unicode_font_bold),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(mcnemar_table)
                elements.append(Spacer(1, 30))
            
            if mcc_scores:
                # Sección MCC Scores
                elements.append(Paragraph("Matthews Correlation Coefficient (MCC) por Modelo", self.styles['CustomSubtitle']))
                elements.append(Paragraph(
                    "Los MCC scores muestran la calidad de cada modelo individualmente. El MCC es una métrica balanceada que considera verdaderos/falsos positivos y negativos, proporcionando una evaluación robusta del rendimiento del modelo.",
                    self.styles['UnicodeNormal']
                ))
                elements.append(Spacer(1, 10))
                
                # Crear gráfico de MCC Scores
                mcc_fig = create_mcc_scores_plot(mcc_scores)
                if mcc_fig:
                    img_buffer = io.BytesIO()
                    mcc_fig.write_image(img_buffer, format='png', width=800, height=600)
                    img_buffer.seek(0)
                    
                    try:
                        # Usar buffer de memoria directamente en lugar de archivo temporal
                        img_buffer.seek(0)
                        img = RLImage(img_buffer, width=6*inch, height=4.5*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 20))
                    except Exception as e:
                        logger.error(f"Error agregando imagen MCC al PDF: {e}")
                        elements.append(Paragraph("Error al cargar gráfico MCC", self.styles['UnicodeNormal']))
                        elements.append(Spacer(1, 20))
                
                # Tabla de MCC Scores individuales
                elements.append(Paragraph("Tabla de MCC Scores por Modelo", self.styles['CustomSubtitle']))
                elements.append(Paragraph(
                    "Resumen de los scores MCC individuales de cada modelo evaluado. Los valores MCC van de -1 a 1, donde 1 indica una predicción perfecta.",
                    self.styles['UnicodeNormal']
                ))
                elements.append(Spacer(1, 10))
                
                # Crear tabla de MCC scores
                mcc_data = [['Modelo', 'MCC Score', 'Calidad']]
                for model_name, mcc_score in mcc_scores.items():
                    if mcc_score > 0.8:
                        quality = "Excelente"
                    elif mcc_score > 0.6:
                        quality = "Bueno"
                    elif mcc_score > 0.4:
                        quality = "Regular"
                    else:
                        quality = "Malo"
                    
                    mcc_data.append([
                        model_name,
                        f"{mcc_score:.4f}",
                        quality
                    ])
                
                mcc_table = Table(mcc_data)
                mcc_table.setStyle(self._create_unicode_table_style([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), self.unicode_font_bold),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(mcc_table)
                elements.append(Spacer(1, 20))
                
            
            # Interpretación de resultados
            elements.append(Paragraph("Interpretación de Resultados Estadísticos", self.styles['CustomSubtitle']))
            elements.append(Paragraph(
                "El Test de McNemar evalúa diferencias significativas entre modelos por pares (p < 0.05 = diferencia significativa). "
                "Los MCC Scores individuales muestran la calidad de cada modelo: >0.8 = Excelente, >0.6 = Bueno, >0.4 = Regular, <0.4 = Malo. "
                "Estos análisis proporcionan una evaluación objetiva y estadísticamente respaldada del rendimiento de cada modelo.",
                self.styles['UnicodeNormal']
            ))
            elements.append(Spacer(1, 20))
            
        except Exception as e:
            logger.error(f"Error generando sección de tests estadísticos: {e}")
            elements.append(Paragraph(
                "Error: No se pudieron generar los tests estadísticos para este reporte.",
                self.styles['UnicodeNormal']
            ))
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_disclaimer(self):
        """Crea el disclaimer del reporte"""
        elements = []
        
        elements.append(Paragraph(self.t('important_notice'), self.styles['CustomSubtitle']))
        elements.append(Paragraph(self.t('disclaimer_text'), self.styles['UnicodeNormal']))
        
        return elements

def generate_pdf_report(predictions, image_info, patient_info, t_func, statistical_results=None, probability_fig=None, consensus_fig=None, original_image=None, enhanced_image=None, hybrid_training_info=None, hybrid_comparison_data=None):
    """
    Función helper para generar el reporte PDF con gráficos e imágenes
    
    Args:
        predictions: Predicciones de los modelos
        image_info: Información de la imagen
        patient_info: Información del paciente
        t_func: Función de traducción
        statistical_results: Resultados estadísticos (opcional)
        probability_fig: Gráfico de probabilidades (opcional)
        consensus_fig: Gráfico de consenso (opcional)
        original_image: Imagen original PIL (opcional)
        enhanced_image: Imagen mejorada PIL (opcional)
        hybrid_training_info: Información de entrenamiento híbrido (opcional)
        hybrid_comparison_data: Datos de comparación híbrida con McNemar y Mateos (opcional)
    
    Returns:
        bytes: Contenido del PDF o None si hay error
    """
    try:
        # Log para debugging desde Streamlit
        logger.info("=== INICIANDO GENERACION PDF DESDE STREAMLIT ===")
        logger.info(f"Directorio de trabajo: {os.getcwd()}")
        logger.info(f"Datos híbridos disponibles: {bool(hybrid_comparison_data)}")
        
        generator = PDFReportGenerator(t_func)
        
        pdf_bytes = generator.generate_report(predictions, image_info, patient_info, None, probability_fig, consensus_fig, original_image, enhanced_image, hybrid_training_info, hybrid_comparison_data)
        
        if pdf_bytes:
            logger.info(f"PDF generado exitosamente: {len(pdf_bytes):,} bytes")
        else:
            logger.error("PDF no se pudo generar (bytes vacíos)")
            
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Error en generación de PDF: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        if 'st' in globals():
            st.error(f"Error generando PDF: {str(e)}")
        return None

def create_download_link(pdf_content, filename="reporte_sipakmed.pdf"):
    """
    Crea un enlace de descarga para el PDF
    
    Args:
        pdf_content: Contenido del PDF en bytes
        filename: Nombre del archivo
    
    Returns:
        str: HTML del enlace de descarga
    """
    if pdf_content:
        b64 = base64.b64encode(pdf_content).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Descargar PDF</a>'
    return ""
