# app_web.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Optimización Agrícola",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clase del sistema agrícola para la web
class SistemaAgricolaWeb:
    def __init__(self):
        self.datos = None
        self.modelo = None
        self.resultados = {}
        
    def generar_datos_ejemplo(self, n_muestras=1000):
        """Genera datos de ejemplo para la demostración web"""
        np.random.seed(42)
        
        datos = {
            'fecha': pd.date_range(start='2023-01-01', periods=n_muestras, freq='D'),
            'temperatura_promedio': np.random.normal(25, 5, n_muestras),
            'humedad_suelo': np.random.normal(60, 15, n_muestras),
            'ph_suelo': np.random.normal(6.5, 0.8, n_muestras),
            'nitrogeno': np.random.normal(150, 40, n_muestras),
            'fosforo': np.random.normal(80, 25, n_muestras),
            'potasio': np.random.normal(120, 35, n_muestras),
            'precipitacion': np.random.gamma(2, 100, n_muestras),
            'tipo_cultivo': np.random.choice(['Maíz', 'Trigo', 'Soja', 'Arroz', 'Girasol'], n_muestras),
            'area_cultivada': np.random.uniform(5, 200, n_muestras),
            'uso_agua': np.random.normal(500, 150, n_muestras)
        }
        
        # Calcular rendimiento
        datos['rendimiento'] = (
            np.clip(datos['temperatura_promedio'], 15, 35) * 2.5 +
            np.clip(datos['humedad_suelo'], 40, 80) * 1.8 +
            np.where((datos['ph_suelo'] >= 6) & (datos['ph_suelo'] <= 7.5), 50, 20) +
            np.clip(datos['nitrogeno'], 100, 200) * 0.3 +
            np.clip(datos['fosforo'], 50, 120) * 0.4 +
            np.clip(datos['potasio'], 80, 160) * 0.35 +
            np.random.normal(0, 30, n_muestras)
        )
        
        # Ajustar por tipo de cultivo
        ajustes_cultivo = {'Maíz': 1.2, 'Trigo': 1.0, 'Soja': 1.1, 'Arroz': 0.9, 'Girasol': 1.05}
        for cultivo, ajuste in ajustes_cultivo.items():
            mask = datos['tipo_cultivo'] == cultivo
            datos['rendimiento'][mask] = datos['rendimiento'][mask] * ajuste
        
        self.datos = pd.DataFrame(datos)
        return self.datos

# Inicializar sistema en session state
if 'sistema_web' not in st.session_state:
    st.session_state.sistema_web = SistemaAgricolaWeb()
    st.session_state.datos_generados_web = False

# Sidebar
with st.sidebar:
    st.title("🌱 Sistema Agrícola")
    st.markdown("---")
    
    st.header("⚙️ Configuración")
    n_muestras = st.slider("Número de muestras", 100, 2000, 1000)
    
    if st.button("🔄 Generar Datos", use_container_width=True):
        with st.spinner("Generando datos agrícolas..."):
            st.session_state.sistema_web.generar_datos_ejemplo(n_muestras)
            st.session_state.datos_generados_web = True
        st.success("¡Datos generados!")
    
    st.markdown("---")
    st.header("📊 Módulos")
    opcion = st.radio(
        "Selecciona un módulo:",
        ["🏠 Inicio", "📈 Análisis Exploratorio", "🤖 Modelo Predictivo", 
         "💧 Gestión Recursos", "🔍 Monitoreo", "📋 Reporte Completo"]
    )

# Página principal
st.title("🌱 Sistema de Optimización Agrícola Inteligente")
st.markdown("---")

# Navegación entre páginas
if opcion == "🏠 Inicio":
    st.header("🏠 Dashboard Principal")
    
    if st.session_state.datos_generados_web:
        datos = st.session_state.sistema_web.datos
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Muestras", len(datos))
        with col2:
            st.metric("Rendimiento Promedio", f"{datos['rendimiento'].mean():.1f} kg/ha")
        with col3:
            st.metric("Tipos de Cultivo", datos['tipo_cultivo'].nunique())
        with col4:
            st.metric("Eficiencia Hídrica", f"{(datos['rendimiento'] / datos['uso_agua']).mean():.3f} kg/m³")
        
        # Vista previa de datos
        with st.expander("📄 Vista Previa de Datos"):
            st.dataframe(datos.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Genera datos primero usando el botón en la sidebar")
        st.info("""
        **Sistema de Optimización Agrícola incluye:**
        - 📊 Análisis exploratorio de datos
        - 🤖 Modelos predictivos de ML
        - 💧 Optimización de recursos hídricos
        - 🧪 Gestión de fertilizantes
        - 🔍 Monitoreo en tiempo real
        - 📋 Reportes ejecutivos
        """)

elif opcion == "📈 Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio de Datos")
    
    if not st.session_state.datos_generados_web:
        st.error("⚠️ Genera datos primero en la página de inicio")
        st.stop()
    
    sistema = st.session_state.sistema_web
    datos = sistema.datos
    
    # Estadísticas rápidas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución del Rendimiento")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(datos['rendimiento'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Rendimiento (kg/ha)')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Rendimiento por Cultivo")
        fig, ax = plt.subplots(figsize=(10, 6))
        rendimiento_por_cultivo = datos.groupby('tipo_cultivo')['rendimiento'].mean()
        rendimiento_por_cultivo.plot(kind='bar', ax=ax, color='lightblue')
        ax.set_ylabel('Rendimiento (kg/ha)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    fig, ax = plt.subplots(figsize=(12, 8))
    variables_numericas = datos.select_dtypes(include=[np.number])
    correlacion = variables_numericas.corr()
    sns.heatmap(correlacion, annot=True, cmap='RdYlGn', center=0, ax=ax, fmt='.2f')
    st.pyplot(fig)

elif opcion == "🤖 Modelo Predictivo":
    st.header("🤖 Modelo Predictivo de Rendimiento")
    
    if not st.session_state.datos_generados_web:
        st.error("⚠️ Genera datos primero en la página de inicio")
        st.stop()
    
    sistema = st.session_state.sistema_web
    
    if st.button("🚀 Entrenar Modelo", use_container_width=True):
        with st.spinner("Entrenando modelo de Machine Learning..."):
            # Preparar datos
            datos = sistema.datos.copy()
            le = LabelEncoder()
            datos['tipo_cultivo_encoded'] = le.fit_transform(datos['tipo_cultivo'])
            
            caracteristicas = [
                'temperatura_promedio', 'humedad_suelo', 'ph_suelo',
                'nitrogeno', 'fosforo', 'potasio', 'precipitacion',
                'area_cultivada', 'uso_agua', 'tipo_cultivo_encoded'
            ]
            
            X = datos[caracteristicas]
            y = datos['rendimiento']
            
            # Entrenar modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = RandomForestRegressor(n_estimators=100, random_state=42)
            modelo.fit(X_train, y_train)
            
            # Evaluar
            y_pred = modelo.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Importancias
            importancias = pd.DataFrame({
                'caracteristica': caracteristicas,
                'importancia': modelo.feature_importances_
            }).sort_values('importancia', ascending=False)
            
            # Guardar resultados
            sistema.modelo = modelo
            sistema.resultados['modelo'] = {
                'mae': mae,
                'r2': r2,
                'importancias': importancias,
                'y_test': y_test,
                'y_pred': y_pred
            }
        
        st.success("✅ Modelo entrenado exitosamente!")
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("Error Absoluto Medio", f"{mae:.2f}")
        
        with col2:
            st.metric("Calificación", "Excelente" if r2 > 0.8 else "Bueno" if r2 > 0.6 else "Regular")
        
        # Gráfica de predicciones
        st.subheader("Predicciones vs Valores Reales")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Predicciones')
        ax.set_title(f'R² = {r2:.3f}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Importancias
        st.subheader("Importancia de Características")
        fig, ax = plt.subplots(figsize=(10, 6))
        importancias_top = importancias.head(10)
        ax.barh(importancias_top['caracteristica'], importancias_top['importancia'])
        ax.set_xlabel('Importancia')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    else:
        st.info("💡 Haz clic en el botón para entrenar el modelo predictivo")

elif opcion == "💧 Gestión Recursos":
    st.header("💧 Gestión y Optimización de Recursos")
    
    if not st.session_state.datos_generados_web:
        st.error("⚠️ Genera datos primero en la página de inicio")
        st.stop()
    
    sistema = st.session_state.sistema_web
    datos = sistema.datos
    
    # Eficiencia hídrica
    st.subheader("Eficiencia Hídrica por Cultivo")
    datos['eficiencia_hidrica'] = datos['rendimiento'] / datos['uso_agua']
    eficiencia_por_cultivo = datos.groupby('tipo_cultivo')['eficiencia_hidrica'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    eficiencia_por_cultivo.sort_values().plot(kind='barh', ax=ax, color='lightblue')
    ax.set_xlabel('Rendimiento por m³ de agua')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Optimización de fertilizantes
    st.subheader("Optimización de Fertilizantes")
    
    rangos_optimos = {
        'nitrogeno': (120, 180),
        'fosforo': (60, 100),
        'potasio': (100, 140)
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (nutriente, (min_opt, max_opt)) in enumerate(rangos_optimos.items()):
        with [col1, col2, col3][i]:
            dentro_rango = datos[
                (datos[nutriente] >= min_opt) & (datos[nutriente] <= max_opt)
            ]
            porcentaje = len(dentro_rango) / len(datos) * 100
            rendimiento_dentro = dentro_rango['rendimiento'].mean()
            
            st.metric(
                f"{nutriente.capitalize()}",
                f"{porcentaje:.1f}% en rango",
                delta=f"{rendimiento_dentro:.0f} kg/ha" if porcentaje > 50 else None
            )

elif opcion == "🔍 Monitoreo":
    st.header("🔍 Sistema de Monitoreo en Tiempo Real")
    
    # Simular datos de monitoreo
    np.random.seed(42)
    horas = list(range(24))
    temperatura = np.random.normal(25, 3, 24) + 8 * np.sin(2 * np.pi * np.array(horas) / 24)
    humedad_suelo = np.random.normal(65, 10, 24)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Temperatura
    ax1.plot(horas, temperatura, 'r-', linewidth=2, marker='o')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Límite superior')
    ax1.axhline(y=15, color='blue', linestyle='--', alpha=0.7, label='Límite inferior')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Monitoreo de Temperatura - 24 Horas')
    
    # Humedad del suelo
    ax2.plot(horas, humedad_suelo, 'b-', linewidth=2, marker='s')
    ax2.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Límite mínimo')
    ax2.set_ylabel('Humedad (%)')
    ax2.set_xlabel('Hora del día')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Monitoreo de Humedad del Suelo - 24 Horas')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Alertas
    st.subheader("⚠️ Sistema de Alertas")
    alertas_temperatura = sum(temperatura > 30)
    alertas_humedad = sum(humedad_suelo < 40)
    
    col1, col2 = st.columns(2)
    with col1:
        if alertas_temperatura > 0:
            st.error(f"Temperatura alta: {alertas_temperatura} alertas")
        else:
            st.success("Temperatura dentro de rangos normales")
    
    with col2:
        if alertas_humedad > 0:
            st.error(f"Humedad baja: {alertas_humedad} alertas")
        else:
            st.success("Humedad dentro de rangos normales")

elif opcion == "📋 Reporte Completo":
    st.header("📋 Reporte Ejecutivo Completo")
    
    if not st.session_state.datos_generados_web:
        st.error("⚠️ Genera datos primero en la página de inicio")
        st.stop()
    
    sistema = st.session_state.sistema_web
    datos = sistema.datos
    
    if st.button("📊 Generar Reporte Completo", use_container_width=True):
        with st.spinner("Generando reporte ejecutivo..."):
            # Métricas clave
            rendimiento_promedio = datos['rendimiento'].mean()
            eficiencia_hidrica = (datos['rendimiento'] / datos['uso_agua']).mean()
            cultivo_mas_rentable = datos.groupby('tipo_cultivo')['rendimiento'].mean().idxmax()
            
            st.success("✅ Reporte generado exitosamente!")
            
            # Resumen ejecutivo
            st.subheader("🎯 Resumen Ejecutivo")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rendimiento Promedio", f"{rendimiento_promedio:.1f} kg/ha")
            with col2:
                st.metric("Eficiencia Hídrica", f"{eficiencia_hidrica:.3f} kg/m³")
            with col3:
                st.metric("Cultivo Recomendado", cultivo_mas_rentable)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones Estratégicas")
            
            st.info("""
            **1. PRIORIZACIÓN DE CULTIVOS**
            - Enfocar en el cultivo con mayor rendimiento y eficiencia
            - Considerar rotación de cultivos para mejorar el suelo
            
            **2. OPTIMIZACIÓN DE RECURSOS**
            - Implementar sistema de riego inteligente
            - Monitorear niveles de nutrientes regularmente
            - Ajustar fertilización según análisis de suelo
            
            **3. MONITOREO CONTINUO**
            - Establecer sistema de alertas tempranas
            - Monitorear condiciones climáticas críticas
            - Seguimiento continuo del rendimiento
            """)
            
            # Descargar datos
            st.subheader("💾 Exportar Datos")
            csv = datos.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Datos Completos (CSV)",
                data=csv,
                file_name="datos_agricolas_completos.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.caption("🌾 Sistema de Optimización Agrícola - Versión Web 1.0")