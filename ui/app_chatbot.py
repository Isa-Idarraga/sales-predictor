"""
Sales-Predictor - Interfaz Chatbot
Sistema Interactivo de Valoración de Inmuebles en Colombia
Basado en Random Forest con MAPE = 0.80%
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QScrollArea, QLabel, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor, QIcon

import joblib
import pandas as pd
import numpy as np


class PredictorBot:
    """Lógica de conversación y predicción del chatbot"""
    
    def __init__(self):
        self.step = 0
        self.data = {}
        self.modelo = None
        self.df = None
        self.ciudades_validas = []
        self.departamentos_validos = []
        self.tipos_propiedad_validos = []
        self.mapeo_ciudad_depto = {}
        self.esperando_coordenadas = False
        self.coordenadas_preguntadas = False
        
        # Cargar modelo y dataset
        self.cargar_modelo()
        
    def cargar_modelo(self):
        """Carga el modelo y dataset"""
        try:
            # Cambiar al directorio raíz del proyecto
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            os.chdir(project_root)
            
            self.modelo = joblib.load('models/random_forest_model.pkl')
            
            try:
                self.df = pd.read_csv('data/dataset_limpio.csv')
                self.ciudades_validas = sorted(self.df['ciudad'].unique())
                self.departamentos_validos = sorted(self.df['departamento'].unique())
                self.tipos_propiedad_validos = sorted(self.df['tipo_propiedad'].unique())
                self.mapeo_ciudad_depto = self.df.groupby('ciudad')['departamento'].first().to_dict()
            except FileNotFoundError:
                # Valores por defecto
                self.ciudades_validas = ['Bogotá D.C', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena']
                self.departamentos_validos = ['Cundinamarca', 'Antioquia', 'Valle del Cauca', 'Atlántico', 'Bolívar']
                self.tipos_propiedad_validos = ['Apartamento', 'Casa', 'Lote', 'Finca']
                self.mapeo_ciudad_depto = {
                    'Bogotá D.C': 'Cundinamarca',
                    'Medellín': 'Antioquia',
                    'Cali': 'Valle del Cauca',
                    'Barranquilla': 'Atlántico',
                    'Cartagena': 'Bolívar'
                }
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {str(e)}")
    
    def get_mensaje_bienvenida(self):
        """Mensaje de bienvenida inicial"""
        return """🏠 ¡Bienvenido al Sistema de Valoración Inmobiliaria!

Soy tu asistente virtual y te ayudaré a estimar el valor de tu propiedad usando inteligencia artificial.

Voy a hacerte algunas preguntas sobre la propiedad. ¡Empecemos!

📐 **¿Cuál es el área total de la propiedad en metros cuadrados (m²)?**"""
    
    def procesar_respuesta(self, respuesta):
        """Procesa la respuesta del usuario según el paso actual"""
        respuesta = respuesta.strip()
        
        if self.step == 0:  # Área
            return self._procesar_area(respuesta)
        elif self.step == 1:  # Habitaciones
            return self._procesar_habitaciones(respuesta)
        elif self.step == 2:  # Baños
            return self._procesar_banos(respuesta)
        elif self.step == 3:  # Ciudad
            return self._procesar_ciudad(respuesta)
        elif self.step == 4:  # ¿Conoce coordenadas?
            return self._procesar_coordenadas_pregunta(respuesta)
        elif self.step == 5 and self.esperando_coordenadas:  # Latitud
            return self._procesar_latitud(respuesta)
        elif self.step == 6 and self.esperando_coordenadas:  # Longitud
            return self._procesar_longitud(respuesta)
        elif (self.step == 5 and not self.esperando_coordenadas) or self.step == 7:  # Tipo propiedad
            return self._procesar_tipo_propiedad(respuesta)
        elif self.step == 8:  # ¿Valorar otra?
            return self._procesar_otra_valoracion(respuesta)
        
        return "error", "Lo siento, algo salió mal. Por favor intenta de nuevo."
    
    def _procesar_area(self, respuesta):
        """Procesa el área ingresada"""
        try:
            area = float(respuesta.replace(',', '.'))
            if area < 10 or area > 2000:
                return "error", "🚫 El área debe estar entre 10 y 2000 m². Por favor ingresa un valor válido."
            
            self.data['area'] = area
            self.step = 1
            return "success", f"✅ Perfecto, {area} m² registrados.\n\n🛏️ **¿Cuántas habitaciones tiene la propiedad?**"
        except ValueError:
            return "error", "❌ Parece que tu entrada no es un número válido. Por favor ingresa el área en m² (ejemplo: 85 o 120.5)"
    
    def _procesar_habitaciones(self, respuesta):
        """Procesa las habitaciones ingresadas"""
        try:
            habitaciones = int(float(respuesta))
            if habitaciones < 0 or habitaciones > 20:
                return "error", "🚫 El número de habitaciones debe estar entre 0 y 20. Por favor ingresa un valor válido."
            
            self.data['habitaciones'] = habitaciones
            self.step = 2
            return "success", f"✅ {habitaciones} habitación(es) registradas.\n\n🚿 **¿Cuántos baños tiene la propiedad?**"
        except ValueError:
            return "error", "❌ Por favor ingresa un número entero válido (ejemplo: 3 o 2)"
    
    def _procesar_banos(self, respuesta):
        """Procesa los baños ingresados"""
        try:
            banos = int(float(respuesta))
            if banos < 0 or banos > 10:
                return "error", "🚫 El número de baños debe estar entre 0 y 10. Por favor ingresa un valor válido."
            
            self.data['banos'] = banos
            self.step = 3
            
            # Mostrar opciones de ciudad
            ciudades_muestra = self.ciudades_validas[:15]
            ciudades_texto = "\n".join([f"   {i+1}. {ciudad}" for i, ciudad in enumerate(ciudades_muestra)])
            total_ciudades = len(self.ciudades_validas)
            
            mensaje = f"✅ {banos} baño(s) registrado(s).\n\n📍 **¿En qué ciudad se encuentra la propiedad?**\n\n"
            mensaje += f"Algunas opciones ({total_ciudades} disponibles):\n{ciudades_texto}"
            if total_ciudades > 15:
                mensaje += f"\n   ... y {total_ciudades - 15} ciudades más"
            mensaje += "\n\n💡 Escribe el número o el nombre de la ciudad (ejemplo: 2 o Medellín)"
            
            return "success", mensaje
        except ValueError:
            return "error", "❌ Por favor ingresa un número entero válido (ejemplo: 2 o 1)"
    
    def _procesar_ciudad(self, respuesta):
        """Procesa la ciudad ingresada"""
        ciudad_encontrada = None
        
        # Intentar como número primero
        try:
            idx = int(respuesta) - 1
            if 0 <= idx < len(self.ciudades_validas):
                ciudad_encontrada = self.ciudades_validas[idx]
        except ValueError:
            # Si no es número, búsqueda flexible por nombre (case insensitive y sin acentos)
            respuesta_lower = respuesta.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            
            for ciudad in self.ciudades_validas:
                ciudad_lower = ciudad.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
                if respuesta_lower == ciudad_lower or respuesta_lower in ciudad_lower:
                    ciudad_encontrada = ciudad
                    break
        
        if not ciudad_encontrada:
            # Buscar sugerencias
            sugerencias = [c for c in self.ciudades_validas if respuesta.lower() in c.lower()][:5]
            mensaje_error = f"❌ No encontré la ciudad '{respuesta}'."
            if sugerencias:
                mensaje_error += f"\n\n¿Quisiste decir alguna de estas?\n" + "\n".join([f"   {i+1}. {s}" for i, s in enumerate(sugerencias)])
            else:
                mensaje_error += "\n\nPor favor escribe el número o el nombre completo de la ciudad."
            return "error", mensaje_error
        
        self.data['ciudad'] = ciudad_encontrada
        self.data['departamento'] = self.mapeo_ciudad_depto.get(ciudad_encontrada, 'Desconocido')
        self.step = 4
        
        return "success", f"✅ Ciudad: {ciudad_encontrada}, {self.data['departamento']}\n\n🗺️ **¿Conoces las coordenadas geográficas exactas de la propiedad?**\n_(Responde 'sí' o 'no')_"
    
    def _procesar_coordenadas_pregunta(self, respuesta):
        """Procesa si el usuario conoce las coordenadas"""
        respuesta_lower = respuesta.lower()
        
        if respuesta_lower in ['si', 'sí', 's', 'yes', 'y']:
            self.esperando_coordenadas = True
            self.step = 5
            return "success", "📍 **¿Cuál es la latitud?**\n_(Debe estar entre -4.3 y 13.5 para Colombia)_"
        elif respuesta_lower in ['no', 'n', 'nop', 'nope']:
            self.esperando_coordenadas = False
            self._usar_coordenadas_promedio()
            self.step = 5
            
            # Mostrar tipos de propiedad
            tipos_texto = "\n".join([f"   {i+1}. {tipo}" for i, tipo in enumerate(self.tipos_propiedad_validos)])
            mensaje = f"✅ Usaré coordenadas aproximadas de {self.data['ciudad']}: ({self.data['latitud']:.2f}, {self.data['longitud']:.2f})\n\n"
            mensaje += f"🏘️ **¿Qué tipo de propiedad es?**\n\n{tipos_texto}\n\n💡 Escribe el número o el nombre del tipo de propiedad"
            
            return "success", mensaje
        else:
            return "error", "❌ Por favor responde 'sí' o 'no'"
    
    def _procesar_latitud(self, respuesta):
        """Procesa la latitud ingresada"""
        try:
            latitud = float(respuesta.replace(',', '.'))
            if latitud < -4.3 or latitud > 13.5:
                return "error", "🚫 La latitud debe estar entre -4.3 y 13.5 para Colombia. Por favor verifica el valor."
            
            self.data['latitud'] = latitud
            self.step = 6
            return "success", f"✅ Latitud: {latitud}\n\n📍 **¿Cuál es la longitud?**"
        except ValueError:
            return "error", "❌ Por favor ingresa un número válido (ejemplo: 4.60 o -74.08)"
    
    def _procesar_longitud(self, respuesta):
        """Procesa la longitud ingresada"""
        try:
            longitud = float(respuesta.replace(',', '.'))
            if longitud < -79.0 or longitud > -66.8:
                return "error", "🚫 La longitud debe estar entre -79.0 y -66.8 para Colombia. Por favor verifica el valor."
            
            self.data['longitud'] = longitud
            self.step = 7
            
            # Mostrar tipos de propiedad
            tipos_texto = "\n".join([f"   {i+1}. {tipo}" for i, tipo in enumerate(self.tipos_propiedad_validos)])
            mensaje = f"✅ Longitud: {longitud}\n\n🏘️ **¿Qué tipo de propiedad es?**\n\n{tipos_texto}\n\n💡 Escribe el número o el nombre del tipo de propiedad"
            
            return "success", mensaje
        except ValueError:
            return "error", "❌ Por favor ingresa un número válido (ejemplo: -74.08 o -75.5)"
    
    def _procesar_tipo_propiedad(self, respuesta):
        """Procesa el tipo de propiedad ingresado"""
        tipo_encontrado = None
        
        # Intentar como número primero
        try:
            idx = int(respuesta) - 1
            if 0 <= idx < len(self.tipos_propiedad_validos):
                tipo_encontrado = self.tipos_propiedad_validos[idx]
        except ValueError:
            # Si no es número, buscar por nombre
            respuesta_lower = respuesta.lower()
            for tipo in self.tipos_propiedad_validos:
                if respuesta_lower == tipo.lower() or respuesta_lower in tipo.lower():
                    tipo_encontrado = tipo
                    break
        
        if not tipo_encontrado:
            tipos_texto = "\n".join([f"   {i+1}. {tipo}" for i, tipo in enumerate(self.tipos_propiedad_validos)])
            return "error", f"❌ Tipo de propiedad no reconocido.\n\nOpciones válidas:\n{tipos_texto}\n\n💡 Escribe el número o el nombre"
        
        self.data['tipo_propiedad'] = tipo_encontrado
        self.step = 8
        
        # Realizar predicción
        return self._realizar_prediccion()
    
    def _usar_coordenadas_promedio(self):
        """Usa coordenadas promedio de la ciudad"""
        if self.df is not None:
            coords_ciudad = self.df[self.df['ciudad'] == self.data['ciudad']][['latitud', 'longitud']].mean()
            self.data['latitud'] = coords_ciudad['latitud'] if not pd.isna(coords_ciudad['latitud']) else 4.6
            self.data['longitud'] = coords_ciudad['longitud'] if not pd.isna(coords_ciudad['longitud']) else -74.0
        else:
            self.data['latitud'] = 4.6
            self.data['longitud'] = -74.0
    
    def _calcular_categorias(self):
        """Calcula las categorías de tamaño y precio"""
        area = self.data['area']
        
        # Categoría de tamaño
        if area < 60:
            self.data['categoria_tamano'] = 'Pequeña'
        elif area < 120:
            self.data['categoria_tamano'] = 'Mediana'
        elif area < 200:
            self.data['categoria_tamano'] = 'Grande'
        else:
            self.data['categoria_tamano'] = 'Muy Grande'
        
        # Calcular precio_m2
        if self.df is not None:
            precio_m2_promedio = self.df[self.df['ciudad'] == self.data['ciudad']]['precio_m2'].median()
            if pd.isna(precio_m2_promedio):
                precio_m2_promedio = self.df['precio_m2'].median()
        else:
            precio_m2_promedio = 3000000
        
        self.data['precio_m2'] = precio_m2_promedio
        
        # Categoría de precio
        if self.df is not None:
            cuartiles = self.df['precio'].quantile([0.25, 0.5, 0.75]).values
            precio_estimado_inicial = area * precio_m2_promedio
            if precio_estimado_inicial < cuartiles[0]:
                self.data['categoria_precio'] = 'Económica'
            elif precio_estimado_inicial < cuartiles[1]:
                self.data['categoria_precio'] = 'Media'
            elif precio_estimado_inicial < cuartiles[2]:
                self.data['categoria_precio'] = 'Alta'
            else:
                self.data['categoria_precio'] = 'Premium'
        else:
            precio_estimado_inicial = area * precio_m2_promedio
            if precio_estimado_inicial < 200000000:
                self.data['categoria_precio'] = 'Económica'
            elif precio_estimado_inicial < 350000000:
                self.data['categoria_precio'] = 'Media'
            elif precio_estimado_inicial < 600000000:
                self.data['categoria_precio'] = 'Alta'
            else:
                self.data['categoria_precio'] = 'Premium'
    
    def _realizar_prediccion(self):
        """Realiza la predicción del precio"""
        try:
            # Calcular categorías
            self._calcular_categorias()
            
            # Crear DataFrame
            datos_input = pd.DataFrame([{
                'area': self.data['area'],
                'habitaciones': self.data['habitaciones'],
                'banos': self.data['banos'],
                'latitud': self.data['latitud'],
                'longitud': self.data['longitud'],
                'precio_m2': self.data['precio_m2'],
                'ciudad': self.data['ciudad'],
                'departamento': self.data['departamento'],
                'tipo_propiedad': self.data['tipo_propiedad'],
                'categoria_tamano': self.data['categoria_tamano'],
                'categoria_precio': self.data['categoria_precio']
            }])
            
            # Codificar variables
            if self.df is not None and not self.df.empty:
                df_template = self.df.drop('precio', axis=1, errors='ignore').copy()
                df_combined = pd.concat([df_template, datos_input], ignore_index=True)
                datos_encoded = pd.get_dummies(df_combined,
                                               columns=['ciudad', 'departamento', 'tipo_propiedad',
                                                       'categoria_tamano', 'categoria_precio'],
                                               drop_first=False)
                datos_final = datos_encoded.iloc[[-1]].copy()
                
                expected_features = self.modelo.feature_names_in_
                for col in expected_features:
                    if col not in datos_final.columns:
                        datos_final[col] = 0
                datos_final = datos_final[expected_features]
            else:
                datos_final = pd.get_dummies(datos_input,
                                             columns=['ciudad', 'departamento', 'tipo_propiedad',
                                                     'categoria_tamano', 'categoria_precio'],
                                             drop_first=False)
                expected_features = self.modelo.feature_names_in_
                for col in expected_features:
                    if col not in datos_final.columns:
                        datos_final[col] = 0
                datos_final = datos_final[expected_features]
            
            # Predicción
            prediccion = self.modelo.predict(datos_final)[0]
            self.data['prediccion'] = prediccion
            
            # Generar mensaje de resultado
            mensaje = self._generar_mensaje_resultado(prediccion)
            
            return "success", mensaje
            
        except Exception as e:
            return "error", f"❌ Error al realizar la predicción: {str(e)}\n\nPor favor intenta de nuevo."
    
    def _generar_mensaje_resultado(self, prediccion):
        """Genera el mensaje con los resultados de la predicción"""
        mensaje = "🎉 **VALORACIÓN COMPLETADA**\n\n"
        mensaje += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        mensaje += f"📋 **Resumen de la Propiedad:**\n"
        mensaje += f"   • Tipo: {self.data['tipo_propiedad']}\n"
        mensaje += f"   • Área: {self.data['area']:.0f} m²\n"
        mensaje += f"   • Habitaciones: {self.data['habitaciones']}\n"
        mensaje += f"   • Baños: {self.data['banos']}\n"
        mensaje += f"   • Ubicación: {self.data['ciudad']}, {self.data['departamento']}\n"
        mensaje += f"   • Categoría: {self.data['categoria_tamano']} - {self.data['categoria_precio']}\n\n"
        mensaje += f"💰 **PRECIO ESTIMADO:** ${prediccion:,.0f} COP\n"
        mensaje += f"💵 **Precio por m²:** ${prediccion/self.data['area']:,.0f} COP/m²\n\n"
        
        # Comparación con propiedades similares
        if self.df is not None:
            df_similares = self.df[
                (self.df['ciudad'] == self.data['ciudad']) &
                (self.df['tipo_propiedad'] == self.data['tipo_propiedad']) &
                (self.df['area'] >= self.data['area'] * 0.8) &
                (self.df['area'] <= self.data['area'] * 1.2)
            ]
            
            if len(df_similares) > 0:
                mensaje += f"📊 **Comparación con el Mercado:**\n"
                mensaje += f"   • Propiedades similares: {len(df_similares)}\n"
                mensaje += f"   • Precio promedio: ${df_similares['precio'].mean():,.0f} COP\n"
                mensaje += f"   • Rango: ${df_similares['precio'].min():,.0f} - ${df_similares['precio'].max():,.0f} COP\n\n"
                
                diferencia_prom = ((prediccion - df_similares['precio'].mean()) / df_similares['precio'].mean()) * 100
                if abs(diferencia_prom) < 10:
                    mensaje += f"✅ Tu propiedad está dentro del rango normal del mercado\n\n"
                elif diferencia_prom > 0:
                    mensaje += f"📈 Tu propiedad está {diferencia_prom:.1f}% por encima del promedio\n\n"
                else:
                    mensaje += f"📉 Tu propiedad está {abs(diferencia_prom):.1f}% por debajo del promedio\n\n"
        
        mensaje += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        mensaje += "¿Deseas valorar otra propiedad? (responde 'sí' o 'no')"
        
        return mensaje
    
    def _procesar_otra_valoracion(self, respuesta):
        """Procesa si el usuario quiere valorar otra propiedad"""
        respuesta_lower = respuesta.lower()
        
        if respuesta_lower in ['si', 'sí', 's', 'yes', 'y']:
            self.reiniciar()
            return "success", self.get_mensaje_bienvenida()
        else:
            return "final", "¡Gracias por usar Sales-Predictor! 🏠\n\nEspero haberte ayudado. ¡Hasta pronto! 👋"
    
    def reiniciar(self):
        """Reinicia el bot para una nueva conversación"""
        self.step = 0
        self.data = {}
        self.esperando_coordenadas = False
        self.coordenadas_preguntadas = False


class ChatbotWindow(QMainWindow):
    """Ventana principal de la aplicación chatbot"""
    
    def __init__(self):
        super().__init__()
        self.bot = None
        self.init_ui()
        self.iniciar_bot()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario"""
        self.setWindowTitle("Sales-Predictor - Sistema de Valoración Inmobiliaria")
        self.setGeometry(100, 100, 800, 700)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.crear_header()
        main_layout.addWidget(header)
        
        # Área de chat
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: none;
                padding: 20px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        main_layout.addWidget(self.chat_area)
        
        # Área de input
        input_widget = self.crear_input_area()
        main_layout.addWidget(input_widget)
        
        central_widget.setLayout(main_layout)
        
        # Aplicar estilos globales
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
        """)
    
    def crear_header(self):
        """Crea el header de la aplicación"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border: none;
                padding: 20px;
            }
            QLabel {
                color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Título
        titulo = QLabel("🏠 Sales-Predictor")
        titulo.setFont(QFont('Segoe UI', 24, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        
        # Subtítulo
        subtitulo = QLabel("Sistema Inteligente de Valoración Inmobiliaria")
        subtitulo.setFont(QFont('Segoe UI', 12))
        subtitulo.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(titulo)
        layout.addWidget(subtitulo)
        header.setLayout(layout)
        
        return header
    
    def crear_input_area(self):
        """Crea el área de input para el usuario"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-top: 2px solid #e0e0e0;
                padding: 15px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Campo de texto
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Escribe tu respuesta aquí...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f9f9f9;
            }
            QLineEdit:focus {
                border: 2px solid #667eea;
                background-color: white;
            }
        """)
        self.input_field.returnPressed.connect(self.enviar_mensaje)
        
        # Botón enviar
        self.send_button = QPushButton("Enviar 📤")
        self.send_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5568d3, stop:1 #6a3f8f);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a5bc4, stop:1 #5c3680);
            }
        """)
        self.send_button.clicked.connect(self.enviar_mensaje)
        
        layout.addWidget(self.input_field, stretch=4)
        layout.addWidget(self.send_button, stretch=1)
        
        widget.setLayout(layout)
        return widget
    
    def iniciar_bot(self):
        """Inicia el bot y muestra mensaje de bienvenida"""
        try:
            self.bot = PredictorBot()
            mensaje_bienvenida = self.bot.get_mensaje_bienvenida()
            self.agregar_mensaje_bot(mensaje_bienvenida)
        except Exception as e:
            self.agregar_mensaje_error(f"Error al iniciar el sistema: {str(e)}\n\nVerifica que el modelo esté en la carpeta 'models/'.")
    
    def enviar_mensaje(self):
        """Envía el mensaje del usuario y procesa la respuesta"""
        mensaje = self.input_field.text().strip()
        
        if not mensaje:
            return
        
        # Mostrar mensaje del usuario
        self.agregar_mensaje_usuario(mensaje)
        self.input_field.clear()
        
        # Procesar respuesta
        if self.bot:
            tipo, respuesta = self.bot.procesar_respuesta(mensaje)
            
            if tipo == "success":
                self.agregar_mensaje_bot(respuesta)
            elif tipo == "error":
                self.agregar_mensaje_error(respuesta)
            elif tipo == "final":
                self.agregar_mensaje_bot(respuesta)
                self.input_field.setEnabled(False)
                self.send_button.setEnabled(False)
    
    def agregar_mensaje_usuario(self, mensaje):
        """Agrega un mensaje del usuario al chat"""
        html = f"""
        <div style='text-align: right; margin: 10px 0;'>
            <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px; 
                        max-width: 70%; text-align: left; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                <strong>Tú:</strong><br>{mensaje}
            </div>
        </div>
        """
        self.chat_area.append(html)
        self.scroll_to_bottom()
    
    def agregar_mensaje_bot(self, mensaje):
        """Agrega un mensaje del bot al chat"""
        # Convertir markdown básico a HTML
        mensaje_html = mensaje.replace('\n', '<br>')
        mensaje_html = mensaje_html.replace('**', '<strong>').replace('**', '</strong>')
        
        html = f"""
        <div style='text-align: left; margin: 10px 0;'>
            <div style='display: inline-block; background-color: white; 
                        color: #333; padding: 12px 18px; border-radius: 18px 18px 18px 4px; 
                        max-width: 75%; text-align: left; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        border-left: 4px solid #667eea;'>
                <strong style='color: #667eea;'>🤖 Sales-Predictor:</strong><br>{mensaje_html}
            </div>
        </div>
        """
        self.chat_area.append(html)
        self.scroll_to_bottom()
    
    def agregar_mensaje_error(self, mensaje):
        """Agrega un mensaje de error al chat"""
        mensaje_html = mensaje.replace('\n', '<br>')
        
        html = f"""
        <div style='text-align: left; margin: 10px 0;'>
            <div style='display: inline-block; background-color: #fff3cd; 
                        color: #856404; padding: 12px 18px; border-radius: 18px 18px 18px 4px; 
                        max-width: 75%; text-align: left; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        border-left: 4px solid #ffc107;'>
                <strong style='color: #d39e00;'>⚠️ Sales-Predictor:</strong><br>{mensaje_html}
            </div>
        </div>
        """
        self.chat_area.append(html)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Hace scroll automático al final del chat"""
        QTimer.singleShot(100, lambda: self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        ))


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Configurar la aplicación
    app.setApplicationName("Sales-Predictor")
    app.setOrganizationName("Sales-Predictor")
    
    # Crear y mostrar ventana
    window = ChatbotWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
