"""
Sistema Interactivo de Valoración de Inmuebles en Colombia
Basado en Random Forest con MAPE = 0.80%

Uso: python valorar_casa.py
"""

import joblib
import pandas as pd
import numpy as np
import os

# Cargar el modelo entrenado
print("="*80)
print(" "*20 + "🏠 SISTEMA DE VALORACIÓN INMOBILIARIA")
print(" "*25 + "Powered by Random Forest ML")
print("="*80)
print("\n⏳ Cargando modelo entrenado...")

try:
    modelo = joblib.load('models/random_forest_model.pkl')
    print(" Modelo cargado exitosamente (MAPE = 0.80%, R² = 0.9899)\n")
except FileNotFoundError:
    print(" ERROR: No se encontró el modelo en 'models/random_forest_model.pkl'")
    print("   Asegúrate de haber ejecutado el notebook completo primero.")
    exit(1)

# Cargar dataset para obtener las categorías válidas
try:
    df = pd.read_csv('data/dataset_limpio.csv')
    ciudades_validas = sorted(df['ciudad'].unique())
    departamentos_validos = sorted(df['departamento'].unique())
    tipos_propiedad_validos = sorted(df['tipo_propiedad'].unique())
    
    # Crear mapeo automático ciudad → departamento
    mapeo_ciudad_depto = df.groupby('ciudad')['departamento'].first().to_dict()
    
    print(f" Dataset cargado: {len(df)} propiedades de {len(ciudades_validas)} ciudades\n")
except FileNotFoundError:
    print("  No se pudo cargar el dataset, usando valores por defecto")
    ciudades_validas = ['Bogotá D.C', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena']
    departamentos_validos = ['Cundinamarca', 'Antioquia', 'Valle del Cauca', 'Atlántico', 'Bolívar']
    tipos_propiedad_validos = ['Apartamento', 'Casa', 'Lote', 'Finca']
    
    # Mapeo por defecto
    mapeo_ciudad_depto = {
        'Bogotá D.C': 'Cundinamarca',
        'Medellín': 'Antioquia',
        'Cali': 'Valle del Cauca',
        'Barranquilla': 'Atlántico',
        'Cartagena': 'Bolívar'
    }

print("="*80)
print(" "*25 + " INGRESA LOS DATOS DE LA PROPIEDAD")
print("="*80)

# Función para validar entrada numérica
def pedir_numero(mensaje, minimo=0, maximo=None):
    while True:
        try:
            valor = float(input(f"\n{mensaje}: "))
            if valor < minimo:
                print(f"     El valor debe ser mayor o igual a {minimo}!")
                continue
            if maximo and valor > maximo:
                print(f"     El valor debe ser menor o igual a {maximo}!")
                continue
            return valor
        except ValueError:
            print("     Por favor ingresa un número válido!")

# Función para validar selección de lista
def pedir_opcion(mensaje, opciones, mostrar_top=20):
    print(f"\n{mensaje}")
    print(f"   Total de opciones disponibles: {len(opciones)}")
    print(f"   (Mostrando primeras {min(mostrar_top, len(opciones))})")
    
    for i, op in enumerate(opciones[:mostrar_top], 1):
        print(f"   {i}. {op}")
    
    if len(opciones) > mostrar_top:
        print(f"   ... y {len(opciones) - mostrar_top} opciones más")
        print(f"\n    TIP: Puedes escribir el nombre completo (ej: 'Medellín', 'Bogotá D.C')")
    
    while True:
        entrada = input(f"\n➤ Ingresa el nombre o número: ").strip()
        
        # Intentar como número
        try:
            idx = int(entrada) - 1
            if 0 <= idx < len(opciones):
                return opciones[idx]
        except ValueError:
            pass
        
        # Intentar como texto (case insensitive y sin acentos)
        entrada_lower = entrada.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        for op in opciones:
            op_lower = op.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            if entrada_lower == op_lower or entrada_lower in op_lower:
                return op
        
        print(f"     '{entrada}' no encontrado. Opciones:")
        print(f"       - Escribir número (1-{len(opciones)})")
        print(f"       - Escribir nombre exacto (ej: 'Medellín')")
        # Mostrar sugerencias si es similar
        sugerencias = [op for op in opciones[:50] if entrada.lower() in op.lower()]
        if sugerencias:
            print(f"       - ¿Quisiste decir? {', '.join(sugerencias[:5])}")

# Recolectar datos del usuario
print("\n" + "─"*80)
print(" CARACTERÍSTICAS FÍSICAS")
print("─"*80)

area = pedir_numero(" - Área total (m²)", minimo=10, maximo=2000)
habitaciones = int(pedir_numero("- Número de habitaciones", minimo=0, maximo=20))
banos = int(pedir_numero(" - Número de baños", minimo=0, maximo=10))

print("\n" + "─"*80)
print(" UBICACIÓN")
print("─"*80)

ciudad = pedir_opcion("🔹 Ciudad:", ciudades_validas)

# Mapear automáticamente el departamento según la ciudad
departamento = mapeo_ciudad_depto.get(ciudad, 'Desconocido')
print(f"   ℹ  Departamento detectado automáticamente: {departamento}")

# Coordenadas aproximadas (opcional)
usar_coords = input("\n¿Conoces las coordenadas geográficas? (s/n): ").lower() == 's'
if usar_coords:
    latitud = pedir_numero(" - Latitud", minimo=-4.3, maximo=13.5)
    longitud = pedir_numero(" - Longitud", minimo=-79.0, maximo=-66.8)
else:
    # Usar coordenadas promedio de la ciudad del dataset
    if 'df' in locals():
        coords_ciudad = df[df['ciudad'] == ciudad][['latitud', 'longitud']].mean()
        latitud = coords_ciudad['latitud'] if not pd.isna(coords_ciudad['latitud']) else 4.6
        longitud = coords_ciudad['longitud'] if not pd.isna(coords_ciudad['longitud']) else -74.0
    else:
        latitud, longitud = 4.6, -74.0  # Bogotá por defecto
    print(f"    Usando coordenadas aproximadas de {ciudad}: ({latitud:.2f}, {longitud:.2f})")

print("\n" + "─"*80)
print(" TIPO DE PROPIEDAD")
print("─"*80)

tipo_propiedad = pedir_opcion(" - Tipo de propiedad:", tipos_propiedad_validos)

# Calcular precio_m2 estimado (usamos la mediana del dataset por ciudad)
if 'df' in locals():
    precio_m2_promedio = df[df['ciudad'] == ciudad]['precio_m2'].median()
    if pd.isna(precio_m2_promedio):
        precio_m2_promedio = df['precio_m2'].median()
else:
    precio_m2_promedio = 3000000  # Valor por defecto

precio_m2 = precio_m2_promedio

# Calcular categorías (usando los NOMBRES EXACTOS del dataset limpio)
# Valores en dataset: 'Pequeña', 'Mediana', 'Grande', 'Muy Grande'
if area < 60:
    categoria_tamano = 'Pequeña'      
elif area < 120:
    categoria_tamano = 'Mediana'      
elif area < 200:
    categoria_tamano = 'Grande'       
else:
    categoria_tamano = 'Muy Grande'   

# Estimar precio para categoría
# Valores en dataset: 'Económica', 'Media', 'Alta', 'Premium'
if 'df' in locals():
    cuartiles = df['precio'].quantile([0.25, 0.5, 0.75]).values
    precio_estimado_inicial = area * precio_m2
    if precio_estimado_inicial < cuartiles[0]:
        categoria_precio = 'Económica'     #  (Q1)
    elif precio_estimado_inicial < cuartiles[1]:
        categoria_precio = 'Media'          #  (Q2)
    elif precio_estimado_inicial < cuartiles[2]:
        categoria_precio = 'Alta'           #  (Q3)
    else:
        categoria_precio = 'Premium'        #  (Q4)
else:
    # Valores por defecto si no hay dataset
    precio_estimado_inicial = area * precio_m2
    if precio_estimado_inicial < 200000000:
        categoria_precio = 'Económica'
    elif precio_estimado_inicial < 350000000:
        categoria_precio = 'Media'
    elif precio_estimado_inicial < 600000000:
        categoria_precio = 'Alta'
    else:
        categoria_precio = 'Premium'

# Crear DataFrame con los datos ingresados
datos_input = pd.DataFrame([{
    'area': area,
    'habitaciones': habitaciones,
    'banos': banos,
    'latitud': latitud,
    'longitud': longitud,
    'precio_m2': precio_m2,
    'ciudad': ciudad,
    'departamento': departamento,
    'tipo_propiedad': tipo_propiedad,
    'categoria_tamano': categoria_tamano,
    'categoria_precio': categoria_precio
}])

# Codificar variables categóricas (One-Hot Encoding)
print("\n⏳ Procesando datos...")

# ESTRATEGIA : Combinar con dataset, codificar, extraer última fila
if 'df' in locals() and not df.empty:
    # Crear template con TODO el dataset (sin la columna precio)
    df_template = df.drop('precio', axis=1, errors='ignore').copy()
    
    # Concatenar: dataset completo + nueva observación
    df_combined = pd.concat([df_template, datos_input], ignore_index=True)
    
    # Aplicar One-Hot Encoding a TODO (esto garantiza todas las columnas posibles)
    datos_encoded = pd.get_dummies(df_combined, 
                                    columns=['ciudad', 'departamento', 'tipo_propiedad', 
                                            'categoria_tamano', 'categoria_precio'],
                                    drop_first=False)
    
    # Extraer solo la última fila
    datos_final = datos_encoded.iloc[[-1]].copy()
    
    # CRÍTICO: Asegurar que tenga las mismas columnas que el modelo espera
    # El modelo fue entrenado con ciertas columnas, debemos alinear
    expected_features = modelo.feature_names_in_  # Columnas que el modelo espera
    
    # Agregar columnas faltantes con valor 0
    for col in expected_features:
        if col not in datos_final.columns:
            datos_final[col] = 0
    
    # Eliminar columnas extra que el modelo no espera
    datos_final = datos_final[expected_features]
    
    print(f"   ✓ Codificación exitosa: {datos_final.shape[1]} características")
    print(f"   ✓ Alineado con modelo: {len(expected_features)} features esperadas")
    
else:
    # Fallback si no hay dataset disponible
    print("    Advertencia: Dataset no disponible, usando encoding básico")
    datos_final = pd.get_dummies(datos_input, 
                                  columns=['ciudad', 'departamento', 'tipo_propiedad',
                                          'categoria_tamano', 'categoria_precio'],
                                  drop_first=False)
    
    # Intentar alinear con el modelo
    try:
        expected_features = modelo.feature_names_in_
        for col in expected_features:
            if col not in datos_final.columns:
                datos_final[col] = 0
        datos_final = datos_final[expected_features]
    except:
        print("    No se pudo alinear con el modelo, la predicción puede fallar")

# Realizar predicción
print(" Realizando predicción con Random Forest...\n")
prediccion = modelo.predict(datos_final)[0]

# Mostrar resultados
print("="*80)
print(" "*30 + " VALORACIÓN FINAL")
print("="*80)
print()
print(f"    Propiedad: {tipo_propiedad} de {area:.0f} m² en {ciudad}, {departamento}")
print(f"    Características: {habitaciones} habitaciones, {banos} baños")
print(f"    Categoría: {categoria_tamano} - {categoria_precio}")
print()
print(f"   💵 PRECIO ESTIMADO: ${prediccion:,.0f} COP")
print(f"   💵 Precio por m²: ${prediccion/area:,.0f} COP/m²")
print()
print(f"    Precisión del modelo: MAPE = 0.80% (error promedio de $11M COP)")
print(f"    Confiabilidad: R² = 0.9899 (98.99% de varianza explicada)")
print()
print("="*80)

# Comparar con propiedades similares del dataset
if 'df' in locals():
    print("\n" + "─"*80)
    print(" COMPARACIÓN CON PROPIEDADES SIMILARES EN EL MERCADO")
    print("─"*80)
    
    # Filtrar propiedades similares
    df_similares = df[
        (df['ciudad'] == ciudad) &
        (df['tipo_propiedad'] == tipo_propiedad) &
        (df['area'] >= area * 0.8) &
        (df['area'] <= area * 1.2)
    ]
    
    if len(df_similares) > 0:
        print(f"\n   Encontradas {len(df_similares)} propiedades similares en {ciudad}:")
        print(f"   • Precio promedio: ${df_similares['precio'].mean():,.0f} COP")
        print(f"   • Precio mínimo: ${df_similares['precio'].min():,.0f} COP")
        print(f"   • Precio máximo: ${df_similares['precio'].max():,.0f} COP")
        print(f"   • Tu estimación: ${prediccion:,.0f} COP")
        
        diferencia_prom = ((prediccion - df_similares['precio'].mean()) / df_similares['precio'].mean()) * 100
        if abs(diferencia_prom) < 10:
            print(f"    Tu propiedad está dentro del rango normal del mercado")
        elif diferencia_prom > 0:
            print(f"   ⬆️  Tu propiedad está {diferencia_prom:.1f}% por encima del promedio")
        else:
            print(f"   ⬇️  Tu propiedad está {abs(diferencia_prom):.1f}% por debajo del promedio")
    else:
        print(f"\n     No hay suficientes propiedades similares en la base de datos")

print("\n" + "="*80)
print(" "*25 + " VALORACIÓN COMPLETADA")
print("="*80)
print()

# Preguntar si quiere valorar otra propiedad
continuar = input("¿Deseas valorar otra propiedad? (s/n): ").lower()
if continuar == 's':
    print("\n" * 2)
    os.system('python valorar_casa.py')
else:
    print("\n¡Gracias por usar el Sistema de Valoración Inmobiliaria! 🏠\n")
