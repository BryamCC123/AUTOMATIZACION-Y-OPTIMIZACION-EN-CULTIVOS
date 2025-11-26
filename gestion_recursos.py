# gestion_recursos.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GestorRecursosAgricolas:
    def __init__(self, datos):
        self.datos = datos
        self.analisis = {}
        
    def analizar_eficiencia_hidrica(self):
        """Analiza la eficiencia en el uso del agua"""
        print("\n💧 ANALIZANDO EFICIENCIA HÍDRICA")
        print("=" * 50)
        
        # Calcular métricas de eficiencia hídrica
        self.datos['eficiencia_hidrica'] = self.datos['rendimiento'] / self.datos['uso_agua']
        self.datos['agua_por_tonelada'] = self.datos['uso_agua'] / (self.datos['rendimiento'] / 1000)
        
        # Análisis por cultivo
        eficiencia_por_cultivo = self.datos.groupby('tipo_cultivo').agg({
            'eficiencia_hidrica': 'mean',
            'agua_por_tonelada': 'mean',
            'uso_agua': 'mean',
            'rendimiento': 'mean'
        }).round(3)
        
        print("\n📊 Eficiencia Hídrica por Cultivo:")
        print(eficiencia_por_cultivo.sort_values('eficiencia_hidrica', ascending=False))
        
        # Visualizar
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Eficiencia hídrica
        eficiencia_por_cultivo['eficiencia_hidrica'].sort_values().plot(
            kind='barh', ax=axes[0,0], color='lightblue'
        )
        axes[0,0].set_title('Eficiencia Hídrica por Cultivo\n(Rendimiento por m³ de agua)')
        axes[0,0].set_xlabel('Rendimiento por m³ de agua')
        
        # Agua por tonelada
        eficiencia_por_cultivo['agua_por_tonelada'].sort_values(ascending=False).plot(
            kind='barh', ax=axes[0,1], color='salmon'
        )
        axes[0,1].set_title('Agua Requerida por Tonelada')
        axes[0,1].set_xlabel('m³ por tonelada')
        
        # Relación uso agua vs rendimiento
        for cultivo in self.datos['tipo_cultivo'].unique():
            datos_cultivo = self.datos[self.datos['tipo_cultivo'] == cultivo]
            axes[1,0].scatter(datos_cultivo['uso_agua'], datos_cultivo['rendimiento'], 
                            label=cultivo, alpha=0.6, s=50)
        axes[1,0].set_xlabel('Uso de Agua (m³)')
        axes[1,0].set_ylabel('Rendimiento (kg/ha)')
        axes[1,0].set_title('Relación Uso de Agua vs Rendimiento')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Distribución de eficiencia hídrica
        axes[1,1].hist(self.datos['eficiencia_hidrica'], bins=30, alpha=0.7, color='green')
        axes[1,1].axvline(self.datos['eficiencia_hidrica'].mean(), color='red', 
                         linestyle='--', label=f'Promedio: {self.datos["eficiencia_hidrica"].mean():.2f}')
        axes[1,1].set_xlabel('Eficiencia Hídrica')
        axes[1,1].set_ylabel('Frecuencia')
        axes[1,1].set_title('Distribución de Eficiencia Hídrica')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.analisis['eficiencia_hidrica'] = eficiencia_por_cultivo
        return eficiencia_por_cultivo
    
    def optimizar_fertilizantes(self):
        """Optimiza el uso de fertilizantes"""
        print("\n🧪 OPTIMIZANDO USO DE FERTILIZANTES")
        print("=" * 50)
        
        # Definir rangos óptimos basados en literatura agrícola
        rangos_optimos = {
            'nitrogeno': {'min': 120, 'max': 180, 'unidad': 'kg/ha'},
            'fosforo': {'min': 60, 'max': 100, 'unidad': 'kg/ha'},
            'potasio': {'min': 100, 'max': 140, 'unidad': 'kg/ha'}
        }
        
        analisis_fertilizantes = {}
        
        print("📋 RANGOS ÓPTIMOS DE FERTILIZANTES:")
        for nutriente, rango in rangos_optimos.items():
            print(f"   {nutriente.capitalize()}: {rango['min']}-{rango['max']} {rango['unidad']}")
        
        for nutriente, rango in rangos_optimos.items():
            min_opt, max_opt = rango['min'], rango['max']
            
            # Analizar datos dentro y fuera del rango óptimo
            dentro_rango = self.datos[
                (self.datos[nutriente] >= min_opt) & 
                (self.datos[nutriente] <= max_opt)
            ]
            fuera_rango = self.datos[
                (self.datos[nutriente] < min_opt) | 
                (self.datos[nutriente] > max_opt)
            ]
            
            rendimiento_dentro = dentro_rango['rendimiento'].mean()
            rendimiento_fuera = fuera_rango['rendimiento'].mean()
            diferencia = rendimiento_dentro - rendimiento_fuera
            
            analisis_fertilizantes[nutriente] = {
                'muestras_dentro_rango': len(dentro_rango),
                'muestras_fuera_rango': len(fuera_rango),
                'porcentaje_dentro_rango': len(dentro_rango) / len(self.datos) * 100,
                'rendimiento_dentro_rango': rendimiento_dentro,
                'rendimiento_fuera_rango': rendimiento_fuera,
                'diferencia_rendimiento': diferencia,
                'recomendacion': 'OPTIMO' if diferencia > 0 else 'REVISAR'
            }
            
            print(f"\n📊 {nutriente.upper()}:")
            print(f"   Muestras en rango óptimo: {len(dentro_rango)} ({len(dentro_rango)/len(self.datos)*100:.1f}%)")
            print(f"   Rendimiento en rango óptimo: {rendimiento_dentro:.1f} kg/ha")
            print(f"   Rendimiento fuera de rango: {rendimiento_fuera:.1f} kg/ha")
            print(f"   Diferencia: {diferencia:+.1f} kg/ha")
            print(f"   Recomendación: {analisis_fertilizantes[nutriente]['recomendacion']}")
        
        # Visualizar análisis de fertilizantes
        self._visualizar_analisis_fertilizantes(analisis_fertilizantes)
        
        self.analisis['fertilizantes'] = analisis_fertilizantes
        return analisis_fertilizantes
    
    def _visualizar_analisis_fertilizantes(self, analisis_fertilizantes):
        """Visualiza el análisis de fertilizantes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Rendimiento por nivel de nitrógeno
        self.datos['nivel_nitrogeno'] = pd.cut(self.datos['nitrogeno'], 
                                             bins=[0, 100, 120, 180, 200, 300],
                                             labels=['Muy Bajo', 'Bajo', 'Óptimo', 'Alto', 'Muy Alto'])
        rendimiento_n = self.datos.groupby('nivel_nitrogeno')['rendimiento'].mean()
        rendimiento_n.plot(kind='bar', ax=axes[0,0], color=['red', 'orange', 'green', 'orange', 'red'])
        axes[0,0].set_title('Rendimiento por Nivel de Nitrógeno')
        axes[0,0].set_ylabel('Rendimiento (kg/ha)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Rendimiento por nivel de fósforo
        self.datos['nivel_fosforo'] = pd.cut(self.datos['fosforo'],
                                           bins=[0, 40, 60, 100, 120, 200],
                                           labels=['Muy Bajo', 'Bajo', 'Óptimo', 'Alto', 'Muy Alto'])
        rendimiento_p = self.datos.groupby('nivel_fosforo')['rendimiento'].mean()
        rendimiento_p.plot(kind='bar', ax=axes[0,1], color=['red', 'orange', 'green', 'orange', 'red'])
        axes[0,1].set_title('Rendimiento por Nivel de Fósforo')
        axes[0,1].set_ylabel('Rendimiento (kg/ha)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Rendimiento por nivel de potasio
        self.datos['nivel_potasio'] = pd.cut(self.datos['potasio'],
                                           bins=[0, 80, 100, 140, 160, 300],
                                           labels=['Muy Bajo', 'Bajo', 'Óptimo', 'Alto', 'Muy Alto'])
        rendimiento_k = self.datos.groupby('nivel_potasio')['rendimiento'].mean()
        rendimiento_k.plot(kind='bar', ax=axes[1,0], color=['red', 'orange', 'green', 'orange', 'red'])
        axes[1,0].set_title('Rendimiento por Nivel de Potasio')
        axes[1,0].set_ylabel('Rendimiento (kg/ha)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Comparación de diferencias de rendimiento
        diferencias = [analisis_fertilizantes[n]['diferencia_rendimiento'] for n in ['nitrogeno', 'fosforo', 'potasio']]
        nutrientes = ['Nitrógeno', 'Fósforo', 'Potasio']
        colors = ['green' if diff > 0 else 'red' for diff in diferencias]
        
        axes[1,1].bar(nutrientes, diferencias, color=colors, alpha=0.7)
        axes[1,1].set_title('Diferencia de Rendimiento: Dentro vs Fuera del Rango Óptimo')
        axes[1,1].set_ylabel('Diferencia de Rendimiento (kg/ha)')
        axes[1,1].axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir valores en las barras
        for i, v in enumerate(diferencias):
            axes[1,1].text(i, v + (5 if v >= 0 else -5), f'{v:+.1f}', 
                          ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def planificar_rotacion_cultivos(self):
        """Sugiere plan de rotación de cultivos basado en datos"""
        print("\n🔄 PLANIFICANDO ROTACIÓN DE CULTIVOS")
        print("=" * 50)
        
        # Simular beneficios de rotación basados en principios agronómicos
        beneficios_base = {
            ('Maíz', 'Soja'): 12.5,
            ('Soja', 'Trigo'): 8.3,
            ('Trigo', 'Maíz'): 10.1,
            ('Maíz', 'Girasol'): 7.8,
            ('Soja', 'Girasol'): 6.2,
            ('Arroz', 'Soja'): 9.1,
            ('Girasol', 'Trigo'): 5.7
        }
        
        # Calcular rendimientos promedio
        rendimientos = self.datos.groupby('tipo_cultivo')['rendimiento'].mean()
        
        print("📊 RENDIMIENTOS PROMEDIO POR CULTIVO:")
        for cultivo, rendimiento in rendimientos.sort_values(ascending=False).items():
            print(f"   {cultivo}: {rendimiento:.1f} kg/ha")
        
        print("\n🔄 BENEFICIOS DE ROTACIÓN (incremento % esperado):")
        rotaciones_recomendadas = []
        
        for (cultivo1, cultivo2), beneficio in beneficios_base.items():
            if cultivo1 in rendimientos.index and cultivo2 in rendimientos.index:
                rendimiento_base = rendimientos[cultivo2]
                rendimiento_esperado = rendimiento_base * (1 + beneficio/100)
                rotaciones_recomendadas.append({
                    'rotacion': f"{cultivo1} → {cultivo2}",
                    'beneficio_porcentaje': beneficio,
                    'rendimiento_esperado': rendimiento_esperado,
                    'incremento_absoluto': rendimiento_esperado - rendimiento_base
                })
        
        # Ordenar por beneficio
        rotaciones_recomendadas.sort(key=lambda x: x['beneficio_porcentaje'], reverse=True)
        
        for rotacion in rotaciones_recomendadas:
            print(f"   {rotacion['rotacion']}: +{rotacion['beneficio_porcentaje']:.1f}% "
                  f"(+{rotacion['incremento_absoluto']:.1f} kg/ha)")
        
        # Visualizar rotaciones recomendadas
        self._visualizar_rotaciones(rotaciones_recomendadas)
        
        return rotaciones_recomendadas
    
    def _visualizar_rotaciones(self, rotaciones_recomendadas):
        """Visualiza las rotaciones de cultivos recomendadas"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Beneficios porcentuales
        rotaciones_df = pd.DataFrame(rotaciones_recomendadas)
        rotaciones_df = rotaciones_df.head(8)  # Mostrar top 8
        
        ax1.barh(rotaciones_df['rotacion'], rotaciones_df['beneficio_porcentaje'], 
                color='lightgreen')
        ax1.set_xlabel('Beneficio Esperado (%)')
        ax1.set_title('Top 8 Rotaciones por Beneficio Porcentual')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Incrementos absolutos
        ax2.barh(rotaciones_df['rotacion'], rotaciones_df['incremento_absoluto'],
                color='lightcoral')
        ax2.set_xlabel('Incremento de Rendimiento (kg/ha)')
        ax2.set_title('Incremento Absoluto Esperado')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    def generar_reporte_optimizacion(self):
        """Genera un reporte completo de optimización"""
        print("\n" + "="*60)
        print("📋 REPORTE COMPLETO DE OPTIMIZACIÓN DE RECURSOS")
        print("="*60)
        
        # Ejecutar todos los análisis
        self.analizar_eficiencia_hidrica()
        self.optimizar_fertilizantes()
        rotaciones = self.planificar_rotacion_cultivos()
        
        print("\n🎯 RECOMENDACIONES ESTRATÉGICAS:")
        print("1. 💧 EFICIENCIA HÍDRICA:")
        cultivo_eficiente = self.analisis['eficiencia_hidrica']['eficiencia_hidrica'].idxmax()
        print(f"   - Priorizar {cultivo_eficiente} por mayor eficiencia hídrica")
        
        print("\n2. 🧪 FERTILIZANTES:")
        for nutriente, analisis in self.analisis['fertilizantes'].items():
            if analisis['recomendacion'] == 'REVISAR':
                print(f"   - Optimizar aplicación de {nutriente}")
        
        print("\n3. 🔄 ROTACIÓN DE CULTIVOS:")
        mejor_rotacion = rotaciones[0]['rotacion'] if rotaciones else "No disponible"
        print(f"   - Implementar rotación: {mejor_rotacion}")

def ejecutar_gestion_recursos(datos):
    """Función principal para ejecutar la gestión de recursos"""
    gestor = GestorRecursosAgricolas(datos)
    gestor.generar_reporte_optimizacion()
    return gestor

if __name__ == "__main__":
    from analisis_exploratorio import AnalizadorAgricola
    analizador = AnalizadorAgricola()
    datos = analizador.cargar_datos_ejemplo()
    gestor = ejecutar_gestion_recursos(datos)