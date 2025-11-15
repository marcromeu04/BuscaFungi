# Configuración de API Key para Open-Meteo

## ⚠️ Problema: Error 403 Forbidden

Si ves errores como:
```
403 Client Error: Forbidden for url: https://archive-api.open-meteo.com/v1/archive...
```

Esto significa que la **API archive de Open-Meteo requiere autenticación** en tu entorno.

## ✅ Solución: Obtener API Key Gratuita

### Paso 1: Registro

1. Ve a https://open-meteo.com/en/pricing
2. Haz clic en **"Register for free"**
3. Completa el formulario (email + contraseña)
4. Verifica tu email

### Paso 2: Obtener API Key

1. Inicia sesión en https://open-meteo.com
2. Ve a tu dashboard
3. Copia tu **API Key** (algo como: `abc123xyz456...`)

### Paso 3: Configurar API Key

Tienes 2 opciones:

#### Opción A: Variable de Entorno (Recomendado)

```bash
# Linux/Mac
export OPENMETEO_API_KEY="tu_clave_aqui"

# Windows (PowerShell)
$env:OPENMETEO_API_KEY="tu_clave_aqui"

# Windows (CMD)
set OPENMETEO_API_KEY=tu_clave_aqui
```

Para que sea permanente:

```bash
# Linux/Mac - Añadir a ~/.bashrc o ~/.zshrc
echo 'export OPENMETEO_API_KEY="tu_clave_aqui"' >> ~/.bashrc
source ~/.bashrc

# Windows - Agregar a las variables de entorno del sistema
```

#### Opción B: Directamente en el Código

```python
from src.meteo import MeteoDataFetcher

meteo = MeteoDataFetcher(api_key="tu_clave_aqui")
```

### Paso 4: Verificar

Ejecuta el test:

```bash
python test_interpolation.py
```

Si funciona, verás:
```
✅ 15 puntos de muestra obtenidos
✅ Interpolación completada
✅ TEST COMPLETADO EXITOSAMENTE
```

## 📊 Límites del Plan Gratuito

- **10,000 requests/día** (más que suficiente para BuscaFungi)
- **5,000 cells/día** para interpolación
- **Histórico completo** desde 1940
- **Forecast 16 días**

Para BuscaFungi típico:
- Setup grid: ~100 requests
- Training: ~200-500 requests
- Prediction: ~100 requests/fecha

**Total: ~500-700 requests/día** ✅ Dentro del límite

## 🔧 Troubleshooting

### Sigue dando 403 después de configurar

1. Verifica que exportaste la variable:
   ```bash
   echo $OPENMETEO_API_KEY  # Linux/Mac
   echo %OPENMETEO_API_KEY%  # Windows CMD
   $env:OPENMETEO_API_KEY    # Windows PowerShell
   ```

2. Reinicia el terminal

3. Verifica que la API key es correcta (cópiala de nuevo del dashboard)

### La API key no se carga

El código lee automáticamente de `os.getenv('OPENMETEO_API_KEY')`.

Para forzarla manualmente:

```python
import os
os.environ['OPENMETEO_API_KEY'] = 'tu_clave_aqui'
```

### Límite de requests excedido

Si ves errores de límite:
- Espera 24 horas para reset
- O actualiza a plan de pago (€10-50/mes para uso intensivo)

## 🌐 Alternativas

Si no quieres usar API key:

1. **Usar solo Forecast API** (siempre gratuita):
   - Modifica scripts para usar `use_forecast=True`
   - Solo funciona para fechas futuras

2. **Usar datos pre-descargados**:
   - Descarga una vez con API key
   - Usa cache local (`.meteo_cache/`)

3. **APIs alternativas**:
   - WeatherAPI.com (500 requests/día gratis)
   - Meteomatics (prueba gratuita)
   - Visual Crossing (1000 requests/día gratis)

## 📚 Más Información

- Documentación Open-Meteo: https://open-meteo.com/en/docs
- Pricing: https://open-meteo.com/en/pricing
- Support: support@open-meteo.com
