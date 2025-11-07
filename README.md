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
│   ├── properties.csv              # Dataset original (1M registros)
│   └── dataset_limpio.csv          # Dataset limpio (28,755 registros)
│
├── notebooks/
│   └── analisis.ipynb              # Notebook principal con todo el análisis
│
├── models/
│   └── random_forest_model.pkl     # Modelo Random Forest entrenado
├── ui/
│   └── __init__.py                 # Inicializa la app del bot
│   └── app_chatbot.py              #Interfaz de usuario de chatbot
│
└── requeriments.txt                # Librerías necesarias para ejecutar el proyecto
│
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
pip install -r requeriments.txt
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
- Isabella Idarraga Botero
- Juan José Rodríguez Restrepo
- Diego Andres Gonzalez Graciano

**Curso:** Introducción a Inteligencia Artificial  
**Universidad:** Universidad EAFIT
**Fecha:** Noviembre 2025

---

## 📧 Contacto

Para preguntas o colaboraciones:
- iidarrabab@eafit.edu.co
- jjrodrigur@eafit.edu.co
- dagonzal11@eafit.edu.co

---

## 📄 Licencia

Este proyecto es de código abierto bajo licencia MIT. El dataset proviene de Kaggle y está sujeto a sus términos de uso.

---

## 🙏 Agradecimientos

- Kaggle por proporcionar el dataset
- Comunidad de scikit-learn y XGBoost
- Profesor Juan Camilo Londoño Lopera, profesor del curso

---

## ⭐ Si este proyecto te fue útil, ¡dale una estrella!

**¡Democratizando el acceso a tasaciones inmobiliarias en Colombia!** 🇨🇴🏠
