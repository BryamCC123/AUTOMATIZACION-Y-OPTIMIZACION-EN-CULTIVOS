# analisis_exploratorio.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnalizadorAgricola:
    def __init__(self):
        self.datos = None
        self.estadisticas = {}
        
    def cargar_datos_ejemplo(self):
        """Genera datos de ejemplo para demostración"""
        print("🌱 Generando dataset agrícola de ejemplo...")
        np.random.seed(42)
        n_muestras = 1500
        
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
        
        # Calcular rendimiento basado en condiciones óptimas con relaciones no lineales
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
        print(f"✅ Dataset generado: {self.datos.shape[0]} registros, {self.datos.shape[1]} variables")
        return self.datos
    
    def analisis_exploratorio(self):
        """Realiza análisis exploratorio completo"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS EXPLORATORIO AGRÍCOLA")
        print("="*60)
        
        print(f"\n📈 Dimensiones del dataset: {self.datos.shape}")
        print(f"📅 Rango de fechas: {self.datos['fecha'].min()} a {self.datos['fecha'].max()}")
        
        print("\n🔍 Primeras 5 filas:")
        print(self.datos.head())
        
        print("\n📋 Tipos de datos:")
        print(self.datos.dtypes)
        
        print("\n🧮 Estadísticas descriptivas:")
        print(self.datos.describe())
        
        print("\n🌾 Distribución de cultivos:")
        print(self.datos['tipo_cultivo'].value_counts())
        
        print("\n🔎 Valores nulos:")
        print(self.datos.isnull().sum())
    
    def visualizar_datos(self):
        """Crea visualizaciones completas para análisis exploratorio"""
        print("\n📈 Generando visualizaciones...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribución de rendimiento
        plt.subplot(3, 3, 1)
        plt.hist(self.datos['rendimiento'], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title('Distribución del Rendimiento', fontsize=12, fontweight='bold')
        plt.xlabel('Rendimiento (kg/ha)')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        
        # 2. Rendimiento por tipo de cultivo
        plt.subplot(3, 3, 2)
        datos_boxplot = [self.datos[self.datos['tipo_cultivo'] == cultivo]['rendimiento'] 
                        for cultivo in self.datos['tipo_cultivo'].unique()]
        plt.boxplot(datos_boxplot, labels=self.datos['tipo_cultivo'].unique())
        plt.title('Rendimiento por Tipo de Cultivo', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Rendimiento (kg/ha)')
        plt.grid(True, alpha=0.3)
        
        # 3. Matriz de correlación
        plt.subplot(3, 3, 3)
        variables_numericas = self.datos.select_dtypes(include=[np.number])
        correlacion = variables_numericas.corr()
        mask = np.triu(np.ones_like(correlacion, dtype=bool))
        sns.heatmap(correlacion, annot=True, cmap='RdYlGn', center=0, 
                   square=True, fmt='.2f', mask=mask, cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlación', fontsize=12, fontweight='bold')
        
        # 4. Temperatura vs Rendimiento
        plt.subplot(3, 3, 4)
        plt.scatter(self.datos['temperatura_promedio'], self.datos['rendimiento'], 
                   alpha=0.6, c='red', s=20)
        plt.xlabel('Temperatura Promedio (°C)')
        plt.ylabel('Rendimiento (kg/ha)')
        plt.title('Temperatura vs Rendimiento', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. Humedad vs Rendimiento
        plt.subplot(3, 3, 5)
        plt.scatter(self.datos['humedad_suelo'], self.datos['rendimiento'], 
                   alpha=0.6, c='blue', s=20)
        plt.xlabel('Humedad del Suelo (%)')
        plt.ylabel('Rendimiento (kg/ha)')
        plt.title('Humedad del Suelo vs Rendimiento', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. Distribución de pH
        plt.subplot(3, 3, 6)
        plt.hist(self.datos['ph_suelo'], bins=20, alpha=0.7, color='brown', edgecolor='black')
        plt.axvline(6.5, color='red', linestyle='--', label='pH Óptimo')
        plt.axvline(7.0, color='red', linestyle='--')
        plt.title('Distribución del pH del Suelo', fontsize=12, fontweight='bold')
        plt.xlabel('pH')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Evolución temporal del rendimiento
        plt.subplot(3, 3, 7)
        rendimiento_mensual = self.datos.set_index('fecha')['rendimiento'].resample('M').mean()
        plt.plot(rendimiento_mensual.index, rendimiento_mensual.values, marker='o', linewidth=2)
        plt.title('Evolución Mensual del Rendimiento', fontsize=12, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Rendimiento Promedio (kg/ha)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Uso de nutrientes por cultivo
        plt.subplot(3, 3, 8)
        nutrientes_por_cultivo = self.datos.groupby('tipo_cultivo')[['nitrogeno', 'fosforo', 'potasio']].mean()
        nutrientes_por_cultivo.plot(kind='bar', ax=plt.gca())
        plt.title('Uso Promedio de Nutrientes por Cultivo', fontsize=12, fontweight='bold')
        plt.xlabel('Tipo de Cultivo')
        plt.ylabel('Cantidad (kg/ha)')
        plt.xticks(rotation=45)
        plt.legend(title='Nutrientes')
        plt.grid(True, alpha=0.3)
        
        # 9. Eficiencia del agua
        plt.subplot(3, 3, 9)
        self.datos['eficiencia_agua'] = self.datos['rendimiento'] / self.datos['uso_agua']
        eficiencia_por_cultivo = self.datos.groupby('tipo_cultivo')['eficiencia_agua'].mean()
        eficiencia_por_cultivo.sort_values().plot(kind='barh', color='lightblue')
        plt.title('Eficiencia del Agua por Cultivo', fontsize=12, fontweight='bold')
        plt.xlabel('Rendimiento por m³ de Agua')
        plt.tight_layout()
        
        plt.tight_layout()
        plt.show()
        
        return self.datos

def ejecutar_analisis():
    """Función para ejecutar el análisis exploratorio"""
    analizador = AnalizadorAgricola()
    datos = analizador.cargar_datos_ejemplo()
    analizador.analisis_exploratorio()
    analizador.visualizar_datos()
    return datos

if __name__ == "__main__":
    datos = ejecutar_analisis()