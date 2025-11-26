# modelo_prediccion.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModeloPrediccionRendimiento:
    def __init__(self):
        self.modelos = {}
        self.resultados = {}
        self.mejor_modelo = None
        self.importancias = None
        
    def preparar_datos(self, datos):
        """Prepara los datos para el modelado"""
        print("🛠️ Preparando datos para modelado...")
        
        # Crear copia para no modificar original
        datos_modelo = datos.copy()
        
        # Codificar variables categóricas
        le = LabelEncoder()
        datos_modelo['tipo_cultivo_encoded'] = le.fit_transform(datos_modelo['tipo_cultivo'])
        
        # Crear características de temporada a partir de la fecha
        datos_modelo['mes'] = datos_modelo['fecha'].dt.month
        datos_modelo['trimestre'] = datos_modelo['fecha'].dt.quarter
        datos_modelo['es_temporada_alta'] = datos_modelo['mes'].isin([3, 4, 5, 9, 10, 11]).astype(int)
        
        # Seleccionar características
        caracteristicas = [
            'temperatura_promedio', 'humedad_suelo', 'ph_suelo',
            'nitrogeno', 'fosforo', 'potasio', 'precipitacion',
            'area_cultivada', 'uso_agua', 'tipo_cultivo_encoded',
            'mes', 'trimestre', 'es_temporada_alta'
        ]
        
        X = datos_modelo[caracteristicas]
        y = datos_modelo['rendimiento']
        
        print(f"✅ Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        return X, y, caracteristicas
    
    def entrenar_modelos(self, X, y):
        """Entrena múltiples modelos y compara su rendimiento"""
        print("\n🤖 Entrenando modelos de predicción...")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X['tipo_cultivo_encoded']
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Definir modelos
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # Entrenar y evaluar modelos
        for nombre, modelo in modelos.items():
            if nombre == 'Linear Regression':
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            
            # Calcular métricas
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            self.resultados[nombre] = {
                'modelo': modelo,
                'y_pred': y_pred,
                'y_test': y_test,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\n📊 {nombre}:")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R²: {r2:.4f}")
        
        # Seleccionar mejor modelo
        self.mejor_modelo = max(self.resultados.items(), 
                               key=lambda x: x[1]['r2'])
        print(f"\n🏆 Mejor modelo: {self.mejor_modelo[0]} (R²: {self.mejor_modelo[1]['r2']:.4f})")
        
        return X_train, X_test, y_train, y_test
    
    def optimizar_hiperparametros(self, X, y):
        """Optimiza hiperparámetros del mejor modelo"""
        print("\n⚙️ Optimizando hiperparámetros...")
        
        # Usar Random Forest para optimización
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"✅ Mejor score CV: {grid_search.best_score_:.4f}")
        
        self.mejor_modelo = ('Random Forest Optimizado', {
            'modelo': grid_search.best_estimator_,
            'mejores_parametros': grid_search.best_params_
        })
        
        return grid_search.best_estimator_
    
    def analizar_importancias(self, X, caracteristicas):
        """Analiza la importancia de las características"""
        print("\n🔍 Analizando importancia de características...")
        
        if self.mejor_modelo[0].startswith('Random Forest'):
            modelo = self.mejor_modelo[1]['modelo']
            importancias = modelo.feature_importances_
            
            self.importancias = pd.DataFrame({
                'caracteristica': caracteristicas,
                'importancia': importancias
            }).sort_values('importancia', ascending=False)
            
            # Visualizar importancias
            plt.figure(figsize=(10, 8))
            sns.barplot(data=self.importancias, x='importancia', y='caracteristica')
            plt.title('Importancia de Características - Random Forest', fontsize=14, fontweight='bold')
            plt.xlabel('Importancia')
            plt.tight_layout()
            plt.show()
            
            print("\n📋 Top 10 características más importantes:")
            print(self.importancias.head(10))
        
        return self.importancias
    
    def visualizar_predicciones(self):
        """Visualiza las predicciones vs valores reales"""
        print("\n📈 Visualizando resultados de predicción...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (nombre, resultados) in enumerate(list(self.resultados.items())[:4]):
            if idx >= 4:  # Solo mostrar primeros 4 modelos
                break
                
            ax = axes[idx // 2, idx % 2]
            y_test = resultados['y_test']
            y_pred = resultados['y_pred']
            
            # Scatter plot predicciones vs real
            ax.scatter(y_test, y_pred, alpha=0.6, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Valores Reales')
            ax.set_ylabel('Predicciones')
            ax.set_title(f'{nombre}\nR² = {resultados["r2"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generar_recomendaciones(self, datos, caracteristicas):
        """Genera recomendaciones basadas en el análisis del modelo"""
        print("\n💡 Generando recomendaciones...")
        
        # Análisis por cultivo
        cultivos = datos['tipo_cultivo'].unique()
        
        print("\n🌾 CONDICIONES ÓPTIMAS POR CULTIVO:")
        print("-" * 50)
        
        for cultivo in cultivos:
            datos_cultivo = datos[datos['tipo_cultivo'] == cultivo]
            alto_rendimiento = datos_cultivo[datos_cultivo['rendimiento'] > datos_cultivo['rendimiento'].quantile(0.75)]
            
            print(f"\n📊 {cultivo}:")
            print(f"   Rendimiento promedio: {datos_cultivo['rendimiento'].mean():.1f} kg/ha")
            print(f"   Condiciones de alto rendimiento:")
            print(f"   - Temperatura: {alto_rendimiento['temperatura_promedio'].mean():.1f}°C")
            print(f"   - Humedad suelo: {alto_rendimiento['humedad_suelo'].mean():.1f}%")
            print(f"   - pH: {alto_rendimiento['ph_suelo'].mean():.2f}")
            print(f"   - Nitrógeno: {alto_rendimiento['nitrogeno'].mean():.1f} kg/ha")

def ejecutar_modelado(datos):
    """Función principal para ejecutar el modelado"""
    predictor = ModeloPrediccionRendimiento()
    X, y, caracteristicas = predictor.preparar_datos(datos)
    X_train, X_test, y_train, y_test = predictor.entrenar_modelos(X, y)
    predictor.visualizar_predicciones()
    
    # Solo optimizar si el dataset es suficientemente grande
    if len(datos) > 1000:
        modelo_optimizado = predictor.optimizar_hiperparametros(X, y)
    
    importancias = predictor.analizar_importancias(X, caracteristicas)
    predictor.generar_recomendaciones(datos, caracteristicas)
    
    return predictor

if __name__ == "__main__":
    # Para ejecutar independientemente, necesitamos datos
    from analisis_exploratorio import AnalizadorAgricola
    analizador = AnalizadorAgricola()
    datos = analizador.cargar_datos_ejemplo()
    predictor = ejecutar_modelado(datos)