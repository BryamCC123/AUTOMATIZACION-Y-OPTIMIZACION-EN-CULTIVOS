# sistema_completo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del sistema
from analisis_exploratorio import AnalizadorAgricola
from modelo_prediccion import ModeloPrediccionRendimiento
from gestion_recursos import GestorRecursosAgricolas
from monitoreo_tiempo_real import SistemaMonitoreo

class SistemaAgricolaCompleto:
    def __init__(self):
        self.datos = None
        self.analizador = None
        self.predictor = None
        self.gestor = None
        self.monitoreo = None
        self.reporte_final = {}
    
    def ejecutar_sistema_completo(self):
        """Ejecuta todo el sistema agrícola de principio a fin"""
        print("🌱" * 20)
        print("SISTEMA COMPLETO DE OPTIMIZACIÓN AGRÍCOLA")
        print("🌱" * 20)
        print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Paso 1: Análisis Exploratorio
        print("\n" + "="*60)
        print("PASO 1: ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*60)
        self.analizador = AnalizadorAgricola()
        self.datos = self.analizador.cargar_datos_ejemplo()
        self.analizador.analisis_exploratorio()
        self.analizador.visualizar_datos()
        
        # Paso 2: Modelado Predictivo
        print("\n" + "="*60)
        print("PASO 2: MODELADO PREDICTIVO DE RENDIMIENTO")
        print("="*60)
        self.predictor = ModeloPrediccionRendimiento()
        X, y, caracteristicas = self.predictor.preparar_datos(self.datos)
        self.predictor.entrenar_modelos(X, y)
        self.predictor.visualizar_predicciones()
        if hasattr(self.predictor, 'importancias'):
            self.predictor.analizar_importancias(X, caracteristicas)
        
        # Paso 3: Gestión de Recursos
        print("\n" + "="*60)
        print("PASO 3: GESTIÓN Y OPTIMIZACIÓN DE RECURSOS")
        print("="*60)
        self.gestor = GestorRecursosAgricolas(self.datos)
        self.gestor.analizar_eficiencia_hidrica()
        self.gestor.optimizar_fertilizantes()
        self.gestor.planificar_rotacion_cultivos()
        
        # Paso 4: Monitoreo en Tiempo Real
        print("\n" + "="*60)
        print("PASO 4: SISTEMA DE MONITOREO EN TIEMPO REAL")
        print("="*60)
        self.monitoreo = SistemaMonitoreo(self.datos)
        self.monitoreo.simular_sensores()
        self.monitoreo.monitorear_condiciones(mostrar_alertas=False)
        self.monitoreo.visualizar_tendencias_tiempo_real()
        self.monitoreo.analizar_tendencias()
        
        # Generar reporte final
        self._generar_reporte_final()
        
        print("\n🎉 SISTEMA EJECUTADO EXITOSAMENTE!")
        print("📊 Todos los módulos completados y reporte generado")
    
    def _generar_reporte_final(self):
        """Genera un reporte ejecutivo final"""
        print("\n" + "="*70)
        print("📋 REPORTE EJECUTIVO FINAL - SISTEMA AGRÍCOLA")
        print("="*70)
        
        # Métricas clave
        rendimiento_promedio = self.datos['rendimiento'].mean()
        eficiencia_hidrica_promedio = (self.datos['rendimiento'] / self.datos['uso_agua']).mean()
        
        # Mejor modelo predictivo
        mejor_modelo = "Random Forest"  # Valor por defecto
        r2_mejor_modelo = 0.0
        
        if hasattr(self.predictor, 'mejor_modelo') and self.predictor.mejor_modelo:
            mejor_modelo = self.predictor.mejor_modelo[0]
            if mejor_modelo in self.predictor.resultados:
                r2_mejor_modelo = self.predictor.resultados[mejor_modelo]['r2']
        
        # Alertas del sistema
        total_alertas = len(self.monitoreo.alertas) if hasattr(self.monitoreo, 'alertas') else 0
        
        print(f"\n📈 MÉTRICAS CLAVE:")
        print(f"   • Rendimiento promedio: {rendimiento_promedio:.1f} kg/ha")
        print(f"   • Eficiencia hídrica: {eficiencia_hidrica_promedio:.3f} kg/m³")
        print(f"   • Mejor modelo predictivo: {mejor_modelo} (R²: {r2_mejor_modelo:.4f})")
        print(f"   • Alertas generadas: {total_alertas}")
        
        print(f"\n🎯 RECOMENDACIONES ESTRATÉGICAS PRIORITARIAS:")
        
        # Recomendación 1: Cultivo más eficiente
        eficiencia_por_cultivo = self.datos.groupby('tipo_cultivo')['rendimiento'].mean()
        cultivo_mas_eficiente = eficiencia_por_cultivo.idxmax()
        print(f"   1. 💡 PRIORIZAR {cultivo_mas_eficiente.upper()}")
        print(f"      - Mayor rendimiento promedio: {eficiencia_por_cultivo.max():.1f} kg/ha")
        
        # Recomendación 2: Optimización de agua
        eficiencia_hidrica_por_cultivo = (self.datos.groupby('tipo_cultivo')['rendimiento'].mean() / 
                                        self.datos.groupby('tipo_cultivo')['uso_agua'].mean())
        cultivo_mas_eficiente_agua = eficiencia_hidrica_por_cultivo.idxmax()
        print(f"   2. 💧 OPTIMIZAR RIEGO PARA {cultivo_mas_eficiente_agua.upper()}")
        print(f"      - Mayor eficiencia hídrica: {eficiencia_hidrica_por_cultivo.max():.3f} kg/m³")
        
        # Recomendación 3: Fertilizantes
        if hasattr(self.gestor, 'analisis') and 'fertilizantes' in self.gestor.analisis:
            fertilizantes_optimos = self.gestor.analisis.get('fertilizantes', {})
            for nutriente, analisis in fertilizantes_optimos.items():
                if analisis.get('recomendacion') == 'REVISAR':
                    print(f"   3. 🧪 REVISAR APLICACIÓN DE {nutriente.upper()}")
                    print(f"      - {analisis['porcentaje_dentro_rango']:.1f}% en rango óptimo")
        else:
            print(f"   3. 🧪 REVISAR NIVELES DE NUTRIENTES (N-P-K)")
        
        # Recomendación 4: Monitoreo
        if total_alertas > 0:
            print(f"   4. ⚠ ATENDER CONDICIONES CRÍTICAS")
            print(f"      - {total_alertas} alertas requieren atención")
        else:
            print(f"   4. ✅ CONDICIONES ACTUALES ÓPTIMAS")
            print(f"      - No se generaron alertas críticas")
        
        print(f"\n🔄 PRÓXIMOS PASOS SUGERIDOS:")
        print("   • Implementar sistema de riego inteligente")
        print("   • Establecer programa de rotación de cultivos")
        print("   • Monitorear condiciones en tiempo real continuamente")
        print("   • Recolectar más datos para mejorar modelos predictivos")
        
        # Guardar reporte
        self.reporte_final = {
            'fecha_ejecucion': datetime.now(),
            'rendimiento_promedio': rendimiento_promedio,
            'eficiencia_hidrica': eficiencia_hidrica_promedio,
            'mejor_modelo': mejor_modelo,
            'r2_modelo': r2_mejor_modelo,
            'total_alertas': total_alertas,
            'cultivo_recomendado': cultivo_mas_eficiente,
            'recomendaciones': [
                f"Priorizar {cultivo_mas_eficiente}",
                f"Optimizar riego para {cultivo_mas_eficiente_agua}",
                "Revisar aplicación de fertilizantes",
                "Atender condiciones críticas del monitoreo"
            ]
        }
        
        return self.reporte_final
    
    def exportar_resultados(self, formato='excel'):
        """Exporta los resultados a diferentes formatos"""
        print(f"\n💾 Exportando resultados en formato {formato.upper()}...")
        
        try:
            if formato == 'excel':
                # Verificar si openpyxl está disponible
                try:
                    import openpyxl
                except ImportError:
                    print("❌ openpyxl no está instalado. Instálalo con: pip install openpyxl")
                    print("🔄 Exportando en formato CSV en su lugar...")
                    self.exportar_resultados(formato='csv')
                    return
                
                with pd.ExcelWriter('resultados_agricolas.xlsx', engine='openpyxl') as writer:
                    # Datos principales
                    self.datos.to_excel(writer, sheet_name='Datos_Agricolas', index=False)
                    
                    # Importancias del modelo
                    if hasattr(self.predictor, 'importancias') and self.predictor.importancias is not None:
                        self.predictor.importancias.to_excel(writer, sheet_name='Importancias_Modelo', index=False)
                    
                    # Análisis de eficiencia
                    if hasattr(self.gestor, 'analisis') and 'eficiencia_hidrica' in self.gestor.analisis:
                        self.gestor.analisis['eficiencia_hidrica'].to_excel(
                            writer, sheet_name='Eficiencia_Hidrica', index=True
                        )
                    
                    # Reporte final
                    reporte_df = pd.DataFrame([self.reporte_final])
                    reporte_df.to_excel(writer, sheet_name='Reporte_Final', index=False)
                
                print("✅ Resultados exportados a 'resultados_agricolas.xlsx'")
            
            elif formato == 'csv':
                # Exportar múltiples archivos CSV
                self.datos.to_csv('datos_agricolas.csv', index=False)
                print("✅ datos_agricolas.csv")
                
                if hasattr(self.predictor, 'importancias') and self.predictor.importancias is not None:
                    self.predictor.importancias.to_csv('importancias_modelo.csv', index=False)
                    print("✅ importancias_modelo.csv")
                
                if hasattr(self.gestor, 'analisis') and 'eficiencia_hidrica' in self.gestor.analisis:
                    self.gestor.analisis['eficiencia_hidrica'].to_csv('eficiencia_hidrica.csv', index=True)
                    print("✅ eficiencia_hidrica.csv")
                
                reporte_df = pd.DataFrame([self.reporte_final])
                reporte_df.to_csv('reporte_final.csv', index=False)
                print("✅ reporte_final.csv")
                
                print("📁 Todos los archivos CSV exportados exitosamente")
                
        except Exception as e:
            print(f"❌ Error al exportar resultados: {e}")
            print("💡 Los resultados se pueden visualizar en las gráficas generadas")
    
    def mostrar_resumen_ejecucion(self):
        """Muestra un resumen de la ejecución del sistema"""
        print("\n" + "="*50)
        print("📊 RESUMEN DE EJECUCIÓN")
        print("="*50)
        
        print(f"✅ Módulos ejecutados:")
        print(f"   • Análisis Exploratorio: {len(self.datos) if self.datos is not None else 0} registros")
        print(f"   • Modelado Predictivo: {len(self.predictor.resultados) if hasattr(self.predictor, 'resultados') else 0} modelos entrenados")
        print(f"   • Gestión de Recursos: {len(self.gestor.analisis) if hasattr(self.gestor, 'analisis') else 0} análisis realizados")
        print(f"   • Monitoreo: {len(self.monitoreo.alertas) if hasattr(self.monitoreo, 'alertas') else 0} alertas generadas")
        
        if self.reporte_final:
            print(f"\n🎯 Recomendación principal: {self.reporte_final['recomendaciones'][0]}")
        
        print(f"\n📅 Ejecución completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Función principal"""
    sistema = SistemaAgricolaCompleto()
    
    try:
        # Ejecutar sistema completo
        sistema.ejecutar_sistema_completo()
        
        # Mostrar resumen
        sistema.mostrar_resumen_ejecucion()
        
        # Preguntar si exportar resultados
        while True:
            exportar = input("\n¿Desea exportar los resultados? (excel/csv/no): ").lower().strip()
            
            if exportar in ['excel', 'e']:
                sistema.exportar_resultados(formato='excel')
                break
            elif exportar in ['csv', 'c']:
                sistema.exportar_resultados(formato='csv')
                break
            elif exportar in ['no', 'n', '']:
                print("📊 Los resultados están disponibles en las visualizaciones generadas")
                break
            else:
                print("❌ Opción no válida. Use 'excel', 'csv' o 'no'")
        
        print(f"\n✨ Proceso completado exitosamente!")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Ejecución interrumpida por el usuario")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        print("💡 Verifique que todas las dependencias estén instaladas:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl")

if __name__ == "__main__":
    main()