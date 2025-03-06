# Generación de Resúmenes de Documentos Judiciales con LLMs

Este repositorio contiene el código y los datos utilizados en el desarrollo de un pipeline basado en Modelos de Lenguaje de Gran Escala (LLMs) para la generación automática de resúmenes de documentos judiciales. 

## 📂 Estructura del Repositorio

- **`pipeline2.py`**  
  Implementación del segundo diseño del pipeline, que sigue estos pasos:
  1. Carga un documento judicial completo. (Ya sea escogiendo el archivo de forma aleatoria o recibiendo el path de un archivo como input)
  2. Selecciona un conjunto de prompts predefinidos, uno para cada sección del resumen (Datos de la causa, Síntesis y Sumarios).
  3. Procesa cada sección del documento de manera independiente con su respectivo prompt, generando resúmenes individuales usando la API de Claude.
  4. **Evaluación automática con Claude**: Se utiliza Claude como evaluador para comparar el resumen generado con el original, evaluando coherencia, precisión, relevancia, concisión y fidelidad.

- **`resultados/`**  
  Carpeta con los resúmenes generados utilizando diferentes estrategias y modelos:
  - LLaMA y Claude como generadores.
  - Métodos Zero-shot, One-shot y Few-shot.

- **`base/`**  
  Contiene los documentos judiciales en texto plano junto con sus resúmenes originales.

- **`analisis/`**  
  - Archivos de texto con los documentos utilizados en entrenamiento, validación y prueba.
  - Resultados de la clusterización de documentos judiciales.

- **`prompts.text`**  
  Archivo con los prompts utilizados en cada estrategia de generación (`zero-shot`, `one-shot` y `few-shot`).  
  - La estrategia a usar se define con la variable `PROMPT_STRATEGY` (`"zero-shot"`, `"one-shot"` o `"few-shot"`).
  - Dependiendo de la estrategia seleccionada, se deben modificar:
    - El parámetro `prompt_strategy` en el archivo `pipeline2.py`, este dato es útil para el json de los resultados.
    - Los valores de `prompt_datos`, `prompt_sintesis` y `prompt_sumarios` en `pipeline2.py`.

---

## 🚀 Instalación

### 🔹 Requisitos Previos
- Python 3.8+
- API Key de **Claude** (Anthropic)

### 🔹 Instalación de Dependencias

Primero, instala las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt 
```

El pipeline requiere una API Key de Claude para la evaluación automática. Debes almacenarla en una variable de entorno llamada LLM_API_KEY:

```bash
export LLM_API_KEY="TU_CLAVE_AQUI"
```

(O agrégala en tu configuración de entorno si usas Windows o un entorno virtual.)

---

## ⚙️ Uso

Puedes ejecutar el pipeline con un archivo específico o dejar que seleccione uno aleatorio:

```bash
python3 pipeline2.py <ruta_al_documento>
```

Si no se proporciona una ruta, el script elegirá un documento judicial aleatoriamente de la carpeta `base/`.