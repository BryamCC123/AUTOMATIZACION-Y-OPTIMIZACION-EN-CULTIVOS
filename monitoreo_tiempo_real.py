# monitoreo_tiempo_real.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class SistemaMonitoreo:
    def __init__(self, datos_historicos=None):
        self.datos_historicos = datos_historicos
        self.datos_tiempo_real = pd.DataFrame()
        self.alertas = []
        self.umbrales = {
            'temperatura': {'min': 15, 'max': 35},
            'humedad_suelo': {'min': 40, 'max': 80},
            'ph_suelo': {'min': 6.0, 'max': 7.5},
            'humedad_ambiental': {'min': 30, 'max': 85},
            'luminosidad': {'min': 30000, 'max': 120000}
        }
    
    def simular_sensores(self, duracion_horas=24, intervalo_minutos=10):
        """Simula datos de sensores en tiempo real"""
        print("🔧 Iniciando simulación de sensores...")
        
        np.random.seed(42)
        n_muestras = int(duracion_horas * 60 / intervalo_minutos)
        timestamp_actual = datetime.now()
        
        datos_sensor = {
            'timestamp': [timestamp_actual + timedelta(minutes=intervalo_minutos*i) 
                         for i in range(n_muestras)],
            'temperatura': np.random.normal(25, 4, n_muestras),
            'humedad_suelo': np.random.normal(65, 12, n_muestras),
            'ph_suelo': np.random.normal(6.5, 0.3, n_muestras),
            'humedad_ambiental': np.random.normal(65, 15, n_muestras),
            'luminosidad': np.random.normal(80000, 25000, n_muestras),
            'conductividad_electrica': np.random.normal(2.5, 0.5, n_muestras)
        }
        
        # Añadir algunas variaciones y patrones
        for i in range(n_muestras):
            # Patrón diurno de temperatura
            hora = datos_sensor['timestamp'][i].hour
            datos_sensor['temperatura'][i] += 8 * np.sin(2 * np.pi * (hora - 6) / 24)
            
            # Patrón diurno de luminosidad
            if 6 <= hora <= 18:
                datos_sensor['luminosidad'][i] += 30000 * np.sin(np.pi * (hora - 6) / 12)
            else:
                datos_sensor['luminosidad'][i] *= 0.1
            
            # Disminución gradual de humedad del suelo (simula riego periódico)
            if i % 36 == 0:  # Cada 6 horas "regar"
                datos_sensor['humedad_suelo'][i:] += 15
        
        self.datos_tiempo_real = pd.DataFrame(datos_sensor)
        print(f"✅ Simulación completada: {n_muestras} muestras generadas")
        return self.datos_tiempo_real
    
    def monitorear_condiciones(self, mostrar_alertas=True):
        """Monitorea condiciones en tiempo real y genera alertas"""
        print("\n🔍 Monitoreando condiciones en tiempo real...")
        
        self.alertas = []
        
        for _, fila in self.datos_tiempo_real.iterrows():
            alertas_momento = []
            
            # Verificar temperatura
            if fila['temperatura'] > self.umbrales['temperatura']['max']:
                alertas_momento.append(
                    f"🌡️ TEMPERATURA ALTA: {fila['temperatura']:.1f}°C "
                    f"(Umbral: {self.umbrales['temperatura']['max']}°C)"
                )
            elif fila['temperatura'] < self.umbrales['temperatura']['min']:
                alertas_momento.append(
                    f"🌡️ TEMPERATURA BAJA: {fila['temperatura']:.1f}°C "
                    f"(Umbral: {self.umbrales['temperatura']['min']}°C)"
                )
            
            # Verificar humedad del suelo
            if fila['humedad_suelo'] < self.umbrales['humedad_suelo']['min']:
                alertas_momento.append(
                    f"💧 HUMEDAD SUELO BAJA: {fila['humedad_suelo']:.1f}% "
                    f"(Umbral: {self.umbrales['humedad_suelo']['min']}%)"
                )
            elif fila['humedad_suelo'] > self.umbrales['humedad_suelo']['max']:
                alertas_momento.append(
                    f"💧 HUMEDAD SUELO ALTA: {fila['humedad_suelo']:.1f}% "
                    f"(Umbral: {self.umbrales['humedad_suelo']['max']}%)"
                )
            
            # Verificar pH
            if fila['ph_suelo'] < self.umbrales['ph_suelo']['min']:
                alertas_momento.append(
                    f"🧪 pH BAJO: {fila['ph_suelo']:.2f} "
                    f"(Umbral: {self.umbrales['ph_suelo']['min']})"
                )
            elif fila['ph_suelo'] > self.umbrales['ph_suelo']['max']:
                alertas_momento.append(
                    f"🧪 pH ALTO: {fila['ph_suelo']:.2f} "
                    f"(Umbral: {self.umbrales['ph_suelo']['max']})"
                )
            
            # Verificar luminosidad
            if fila['luminosidad'] < self.umbrales['luminosidad']['min']:
                alertas_momento.append(
                    f"☀️ LUMINOSIDAD BAJA: {fila['luminosidad']:.0f} lux "
                    f"(Umbral: {self.umbrales['luminosidad']['min']})"
                )
            
            if alertas_momento and mostrar_alertas:
                print(f"\n🕐 {fila['timestamp'].strftime('%Y-%m-%d %H:%M')}:")
                for alerta in alertas_momento:
                    print(f"   ⚠ {alerta}")
                    self.alertas.append({
                        'timestamp': fila['timestamp'],
                        'alerta': alerta,
                        'tipo': alerta.split(':')[0]
                    })
        
        print(f"\n📊 Resumen de alertas: {len(self.alertas)} alertas generadas")
        
        if self.alertas:
            alertas_por_tipo = pd.DataFrame(self.alertas)['tipo'].value_counts()
            print("\n📋 Alertas por tipo:")
            for tipo, cantidad in alertas_por_tipo.items():
                print(f"   {tipo}: {cantidad} alertas")
    
    def visualizar_tendencias_tiempo_real(self):
        """Visualiza las tendencias de los sensores en tiempo real"""
        print("\n📈 Generando dashboard de monitoreo...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('DASHBOARD DE MONITOREO EN TIEMPO REAL', fontsize=16, fontweight='bold')
        
        # 1. Temperatura
        axes[0,0].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['temperatura'], 
                      color='red', linewidth=2, label='Temperatura')
        axes[0,0].axhline(y=self.umbrales['temperatura']['max'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite superior')
        axes[0,0].axhline(y=self.umbrales['temperatura']['min'], color='blue', 
                         linestyle='--', alpha=0.7, label='Límite inferior')
        axes[0,0].set_title('Temperatura (°C)')
        axes[0,0].set_ylabel('Temperatura (°C)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Humedad del Suelo
        axes[0,1].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['humedad_suelo'], 
                      color='blue', linewidth=2, label='Humedad Suelo')
        axes[0,1].axhline(y=self.umbrales['humedad_suelo']['max'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite superior')
        axes[0,1].axhline(y=self.umbrales['humedad_suelo']['min'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite inferior')
        axes[0,1].set_title('Humedad del Suelo (%)')
        axes[0,1].set_ylabel('Humedad (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. pH del Suelo
        axes[1,0].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['ph_suelo'], 
                      color='brown', linewidth=2, label='pH Suelo')
        axes[1,0].axhline(y=self.umbrales['ph_suelo']['max'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite superior')
        axes[1,0].axhline(y=self.umbrales['ph_suelo']['min'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite inferior')
        axes[1,0].set_title('pH del Suelo')
        axes[1,0].set_ylabel('pH')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Luminosidad
        axes[1,1].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['luminosidad'], 
                      color='orange', linewidth=2, label='Luminosidad')
        axes[1,1].axhline(y=self.umbrales['luminosidad']['min'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite mínimo')
        axes[1,1].set_title('Luminosidad (lux)')
        axes[1,1].set_ylabel('Lux')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 5. Humedad Ambiental
        axes[2,0].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['humedad_ambiental'], 
                      color='green', linewidth=2, label='Humedad Ambiental')
        axes[2,0].axhline(y=self.umbrales['humedad_ambiental']['max'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite superior')
        axes[2,0].axhline(y=self.umbrales['humedad_ambiental']['min'], color='red', 
                         linestyle='--', alpha=0.7, label='Límite inferior')
        axes[2,0].set_title('Humedad Ambiental (%)')
        axes[2,0].set_ylabel('Humedad (%)')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # 6. Conductividad Eléctrica
        axes[2,1].plot(self.datos_tiempo_real['timestamp'], 
                      self.datos_tiempo_real['conductividad_electrica'], 
                      color='purple', linewidth=2, label='Conductividad')
        axes[2,1].set_title('Conductividad Eléctrica (dS/m)')
        axes[2,1].set_ylabel('dS/m')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analizar_tendencias(self):
        """Analiza tendencias y patrones en los datos de monitoreo"""
        print("\n📊 Analizando tendencias y patrones...")
        
        if self.datos_tiempo_real.empty:
            print("❌ No hay datos para analizar")
            return
        
        # Calcular estadísticas por hora del día
        self.datos_tiempo_real['hora'] = self.datos_tiempo_real['timestamp'].dt.hour
        
        estadisticas_por_hora = self.datos_tiempo_real.groupby('hora').agg({
            'temperatura': ['mean', 'min', 'max'],
            'humedad_suelo': ['mean', 'min', 'max'],
            'luminosidad': ['mean', 'min', 'max'],
            'humedad_ambiental': ['mean', 'min', 'max']
        }).round(2)
        
        print("\n📈 ESTADÍSTICAS POR HORA DEL DÍA:")
        print(estadisticas_por_hora)
        
        # Identificar patrones críticos
        temp_max = self.datos_tiempo_real['temperatura'].max()
        temp_min = self.datos_tiempo_real['temperatura'].min()
        humedad_min = self.datos_tiempo_real['humedad_suelo'].min()
        
        print(f"\n🔍 PATRONES CRÍTICOS IDENTIFICADOS:")
        print(f"   • Temperatura máxima: {temp_max:.1f}°C")
        print(f"   • Temperatura mínima: {temp_min:.1f}°C")
        print(f"   • Humedad mínima del suelo: {humedad_min:.1f}%")
        
        # Recomendaciones basadas en patrones
        print(f"\n💡 RECOMENDACIONES:")
        if temp_max > 32:
            print("   • Considerar sombreado para temperaturas altas")
        if humedad_min < 45:
            print("   • Ajustar programa de riego para mantener humedad óptima")
        
        return estadisticas_por_hora

def ejecutar_monitoreo():
    """Función principal para ejecutar el sistema de monitoreo"""
    print("🚀 INICIANDO SISTEMA DE MONITOREO EN TIEMPO REAL")
    print("=" * 50)
    
    monitoreo = SistemaMonitoreo()
    
    # Simular datos de sensores (24 horas, cada 10 minutos)
    datos_sensores = monitoreo.simular_sensores(duracion_horas=24, intervalo_minutos=10)
    
    # Monitorear condiciones y generar alertas
    monitoreo.monitorear_condiciones(mostrar_alertas=True)
    
    # Visualizar tendencias
    monitoreo.visualizar_tendencias_tiempo_real()
    
    # Analizar patrones
    tendencias = monitoreo.analizar_tendencias()
    
    print("\n✅ Sistema de monitoreo ejecutado exitosamente")
    return monitoreo

if __name__ == "__main__":
    monitoreo = ejecutar_monitoreo()