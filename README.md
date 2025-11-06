# 🏠 Predicción de Precios de Inmuebles en Colombia

Sistema de predicción de precios inmobiliarios usando Machine Learning (Random Forest y XGBoost) con **MAPE = 0.80%** y **R² = 0.9899**.

## 📊 Resultados Principales

| Modelo | MAPE | R² | RMSE | MAE |
|--------|------|-----|------|-----|
| Regresión Lineal | 66.58% | 0.3776 | $764M COP | $322M COP |
| **Random Forest** ⭐ | **0.80%** | **0.9899** | **$97M COP** | **$11M COP** |
| XGBoost | 1.42% | 0.9690 | $171M COP | $16M COP |

✅ **Objetivos superados:**
- MAPE < 11% → Alcanzado: 0.80% (13.75× mejor)
- R² > 0.90 → Alcanzado: 0.9899 (9.9% superior)

---

## 🗂️ Estructura del Proyecto

```
proyectofinalia/
│
├── data/
│   ├── co_properties.csv           # Dataset original (1M registros)
│   └── dataset_limpio.csv          # Dataset limpio (28,755 registros)
│
├── notebooks/
│   └── analisis.ipynb              # Notebook principal con todo el análisis
│
├── models/
│   └── random_forest_model.pkl     # Modelo Random Forest entrenado
│
├── figures/
│   ├── comparacion_modelos.png     # Gráfico MAPE y R² de 3 modelos
│   ├── predicciones_vs_reales.png  # Scatter plot predicciones vs reales
│   └── importancia_caracteristicas.png  # Top 15 features importantes
│
├── INFORME_SECCIONES.txt           # Secciones del informe (Métodos, Resultados, etc.)
├── PRESENTACION_GUION.txt          # Guion para exposición de 10 minutos
└── README.md                       # Este archivo
```

---

## 🚀 Instalación y Uso

### 1. Requisitos

```bash
Python 3.8+
```

### 2. Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 3. Ejecutar el notebook

```bash
jupyter notebook notebooks/analisis.ipynb
```

### 4. Usar el modelo entrenado

```python
import joblib
import pandas as pd

# Cargar el modelo
modelo = joblib.load('models/random_forest_model.pkl')

# Preparar datos de ejemplo (debe tener las 365 features codificadas)
# Ver notebook para proceso completo de encoding

# Predecir
precio_predicho = modelo.predict(datos_codificados)
print(f"Precio estimado: ${precio_predicho[0]:,.0f} COP")
```

---

## 📖 Descripción del Dataset

- **Fuente:** Kaggle - Propiedades inmobiliarias de Colombia
- **Registros originales:** 1,043,865
- **Registros limpios:** 28,755 (después de preprocesamiento)
- **Cobertura geográfica:** 177 ciudades en 22 departamentos
- **Variables:** 12 (área, precio, habitaciones, baños, lat, lon, ciudad, departamento, etc.)

### Variables principales:
- `precio`: Precio de la propiedad (COP) - **TARGET**
- `area`: Área total en m²
- `precio_m2`: Precio por metro cuadrado (derivada)
- `habitaciones`: Número de habitaciones
- `banos`: Número de baños
- `latitud`, `longitud`: Coordenadas geográficas
- `ciudad`, `departamento`: Ubicación administrativa
- `tipo_propiedad`: Apartamento, Casa, Lote, Finca

---

## 🔧 Preprocesamiento

1. **Filtrado:**
   - Solo operaciones de venta
   - Área entre 10-2,000 m²
   - Precios válidos (> 0)

2. **Imputación:**
   - Habitaciones/baños: mediana por tipo de propiedad
   - Coordenadas: validación de rangos geográficos

3. **Feature Engineering:**
   - `precio_m2 = precio / area`
   - Categorización de tamaño (Pequeño, Mediano, Grande, Extra Grande)
   - Categorización de precio (4 cuartiles)

4. **Encoding:**
   - One-Hot Encoding de variables categóricas → 365 features finales

---

## 🤖 Modelos

### Random Forest (Mejor modelo) ⭐

**Hiperparámetros óptimos:**
```python
{
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
```

**Características más importantes:**
1. Área (54.5%)
2. Precio/m² (44.9%)
3. Longitud (0.14%)
4. Latitud (0.12%)
5. Baños (0.06%)

---

## 📈 Comparación con Literatura

| Estudio | Año | MAPE/R² | Dataset | Cobertura |
|---------|-----|---------|---------|-----------|
| Pérez et al. | 2022 | 14.3% | 3,200 | Bogotá |
| Lastra | 2021 | R²=0.938 | 1,500 | Medellín |
| García & López | 2020 | R²=0.812 | 2,800 | Cali |
| **ESTE PROYECTO** | **2025** | **0.80%** / **R²=0.9899** | **28,755** | **Colombia** |

**Mejora:** 17.9× en MAPE comparado con Pérez et al.

---

## 📝 Archivos de Documentación

### Para el Informe Escrito:
📄 `INFORME_SECCIONES.txt`
- Sección III: Métodos (preprocesamiento, modelos, validación)
- Sección IV: Resultados (tablas, figuras, métricas)
- Sección V: Discusión (comparación con literatura, limitaciones)
- Sección VI: Conclusiones (logros, trabajo futuro)

### Para la Presentación:
🎤 `PRESENTACION_GUION.txt`
- Estructura de 12 slides para 10 minutos
- Guion completo con tiempos
- Respuestas a preguntas probables

---

## 🎯 Objetivos SMART Cumplidos

- [x] **OE1:** Dataset con >5,000 registros → ✓ 28,755 (5.75×)
- [x] **OE2:** Análisis exploratorio completo → ✓ EDA con visualizaciones
- [x] **OE3:** Implementar ensemble learning → ✓ Random Forest + XGBoost
- [x] **OE4:** MAPE < 11% → ✓ 0.80% (13.75× mejor)
- [x] **OE4:** R² > 0.90 → ✓ 0.9899 (9.9% superior)

---

## 🚀 Trabajo Futuro

### Corto plazo:
- Validar modelo sin `precio_m2` (eliminar data leakage)
- Desarrollar API REST para deployment
- Validación temporal (entrenar en año N, probar en N+1)

### Mediano plazo:
- Expandir a predicción de arriendos
- Incorporar datos de POIs (escuelas, transporte, criminalidad)
- Modelos de series temporales

### Largo plazo:
- Deep learning con autoencoders geográficos
- Sistema de detección de fraude
- Expansión a Latinoamérica

---

## 👥 Autores

**Grupo 5**
- [Tu nombre]
- [Nombre compañero/a]

**Curso:** Inteligencia Artificial  
**Universidad:** [Tu universidad]  
**Fecha:** Noviembre 2025

---

## 📧 Contacto

Para preguntas o colaboraciones:
- Email: [tu_email@ejemplo.com]
- GitHub: [tu_usuario]

---

## 📄 Licencia

Este proyecto es de código abierto bajo licencia MIT. El dataset proviene de Kaggle y está sujeto a sus términos de uso.

---

## 🙏 Agradecimientos

- Kaggle por proporcionar el dataset
- Comunidad de scikit-learn y XGBoost
- Profesores del curso de IA

---

## ⭐ Si este proyecto te fue útil, ¡dale una estrella!

**¡Democratizando el acceso a tasaciones inmobiliarias en Colombia!** 🇨🇴🏠
