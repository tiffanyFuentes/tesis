import os
import json
import random
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge_score import rouge_scorer
import requests  # Necesario para interactuar con la API de Claude

# --- Configuración ---
BASE_PATH = "BASE"
ENTRENAMIENTO_PATH = "Analisis/Entrenamiento.txt"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"  # URL de la API de Claude
CLAUDE_API_KEY = os.getenv("LLM_API_KEY")
CLAUDE_API_VERSION = "2023-06-01" 
MODELO = "CLAUDE"
PROMPT_STRATEGY = "FEW-SHOT"
RESULTADOS_DIR = "/home/tfuentes/resultados"

# --- Funciones auxiliares para lectura de archivos ---
def ajustar_ruta_servidor(ruta_original):
    if "/BASE/" in ruta_original:
        nueva_ruta = ruta_original.split("/BASE/")[-1]
        return os.path.join(BASE_PATH, nueva_ruta)
    else:
        return None

def seleccionar_archivo_aleatorio():
    if not os.path.exists(ENTRENAMIENTO_PATH):
        print(f"No se encontró el archivo de entrenamiento en {ENTRENAMIENTO_PATH}")
        return None

    with open(ENTRENAMIENTO_PATH, "r") as f:
        lineas = [linea.strip().replace(".txt.s.txt", ".txt") for linea in f.readlines()]

    if not lineas:
        print("No hay archivos válidos en el archivo de entrenamiento.")
        return None

    archivo_aleatorio = random.choice(lineas)
    return ajustar_ruta_servidor(archivo_aleatorio)

def leer_archivo(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f"No se encontró el archivo en la ruta: {ruta_archivo}")
        return None

    with open(ruta_archivo, "r") as archivo:
        return archivo.read()

# --- Configuración de Llama ---
def cargar_llama():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generar_resumen_claude(texto, prompt):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": f"{prompt}\n\nTexto judicial:\n{texto}"}
        ]
    }
    response = requests.post(CLAUDE_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_json = response.json()

        # Verificar si el contenido de la respuesta es una lista
        if isinstance(response_json, dict) and "content" in response_json:
            resumen = response_json["content"]
        elif isinstance(response_json, list):
            resumen = "\n".join(item.get("text", "") for item in response_json if "text" in item)
        else:
            resumen = "Resumen no disponible."
        return resumen[0].get("text", "").strip() 
    else:
        print(f"Error en la API de Claude: {response.status_code} - {response.text}")
        return "Error generando resumen"

def calcular_rouge(resumen_original, resumen_generado):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    puntajes = scorer.score(resumen_original, resumen_generado)
    return puntajes

def guardar_resultados_json(archivo, resumen_generado, resumen_original, coh, pre, rel, con, fid, pro):
    resultado = {
        "id": os.path.basename(archivo + "_" + MODELO + "_" + PROMPT_STRATEGY),
        "pipeline": 2,
        "modelo": MODELO,
        "estrategia": PROMPT_STRATEGY,
        "resumen_generado": resumen_generado,
        "resumen_original": resumen_original,
        "coherencia": coh,
        "precision": pre,
        "relevancia": rel,
        "concision": con,
        "fidelidad": fid,
        "puntuacion_promedio": pro 
    }
    carpeta_guardado = os.path.join(RESULTADOS_DIR, f"pipeline{2}", MODELO, PROMPT_STRATEGY)
    os.makedirs(carpeta_guardado, exist_ok=True)
    json_path = os.path.join(carpeta_guardado, os.path.basename(archivo).replace(".txt", ".json"))
    with open(json_path, "w") as json_file:
        json.dump(resultado, json_file, indent=4, ensure_ascii=False)
    print(f"Resultados guardados en {json_path}")

def aplicar_prompt(texto, prompt, llama_pipeline):
    entrada = f"{prompt}\n\nTexto judicial:\n{texto}"
    resultado = llama_pipeline(entrada, max_new_tokens=500, num_return_sequences=1)
    return resultado[0]["generated_text"].strip()
    
def evaluar_aspecto_claude(resumen_original, resumen_generado, aspecto, descripcion):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": f"""Evalúa el siguiente resumen generado en comparación con 
            el resumen original. Considera el aspecto '{aspecto}': {descripcion}. 
            Primero, proporciona una breve justificación de tu evaluación. Luego, asigna 
            una puntuación del 1 al 5 basada en tu análisis.
            \n\nResumen original:\n{resumen_original}\n\nResumen generado:\n{resumen_generado}\n"""}
        ]
    }
    response = requests.post(CLAUDE_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        response_json = response.json()

        # Verificar si el contenido de la respuesta es una lista
        if isinstance(response_json, dict) and "content" in response_json:
            evaluacion = response_json["content"]
        elif isinstance(response_json, list):
            evaluacion = "\n".join(item.get("text", "") for item in response_json if "text" in item)
        else:
            evaluacion = "Evaluación no disponible."

        return evaluacion[0].get("text", "").strip() 
    else:
        print(f"Error en la API al evaluar el aspecto '{aspecto}': {response.status_code} - {response.text}")
        return f"Error al evaluar el aspecto '{aspecto}'."

def calcular_rouge(resumen_original, resumen_generado):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    puntajes = scorer.score(resumen_original, resumen_generado)
    return puntajes

# --- Pipeline Principal ---
def main():
    #llama_pipeline = cargar_llama()
    #Colapsar coherencia y concision en uno, precision y fidelidad en uno y relevancia
    aspectos = {
        "Coherencia": "Determina si las ideas del resumen están conectadas lógicamente y no se contradicen entre sí.",
        "Precisión": "Verifica si el resumen refleja fielmente los datos y eventos del original, sin agregar información incorrecta.",
        "Relevancia": "Evalúa si el resumen incluye las partes más importantes y omite detalles irrelevantes.",
        "Concisión": "Mide si el resumen es breve y directo, evitando redundancias innecesarias.",
        "Fidelidad": "Asegura que el resumen mantiene la intención y el tono del original sin distorsionar su significado."
    }

    if len(sys.argv) > 1:
        print(sys.argv[1])
        archivo_seleccionado = ajustar_ruta_servidor(sys.argv[1])
    else:
        archivo_seleccionado = seleccionar_archivo_aleatorio()

    contenido = leer_archivo(archivo_seleccionado)
    if not contenido:
        print("No se pudo leer el archivo.")
        return

    # Leer los resúmenes asociados
    resumen_path = archivo_seleccionado.replace(".txt", ".txt.s.txt")
    resumen_original = leer_archivo(resumen_path)
    if not resumen_original:
        print(f"No se encontró un resumen asociado en: {resumen_path}")
        return

    print("Generando resultados para las tres secciones del resumen...\n")

    # Prompts para cada sección
    prompt_datos = """Eres un asistente legal que debe extraer los datos principales de una sentencia judicial. Los únicos datos que debes obtener son los siguientes:

    Fecha
    Sede
    Dependencia
    Autos
    Resolución
    Jueces
    Formato del resultado (obligatorio):

    DATOS DE LA CAUSA
    Fecha: [Fecha]
    Sede: [Sede]
    Dependencia: [Dependencia]
    Autos: [Autos]
    Resolución: [Resolución]
    Jueces: [Jueces]

    Ejemplo correcto de los datos de una causa:

    DATOS DE LA CAUSA
    Fecha: 9/3/2021
    Sede: Ciudad de Córdoba
    Dependencia: Cámara de Apelaciones en lo Civil y Comercial de Séptima Nominación
    Autos: “Banco de la Provincia SA c/ Gaz SA - Ordinario - Consignación”, expediente n.° 5781418
    Resolución: Auto n.° 32
    Jueces: María Rosa Molina, Rubén Atilio Remigio y Miguel Jorge Flores

    Ejemplo incorrecto:

    DATOS DE LA CAUSA
    Fecha: marzo
    Sede: Córdoba
    Dependencia: CÁMARA APEL CIV. Y COM 4a
    Autos: caso n.° 5781418
    Resolución: Resuelto favorablemente
    Jueces: Flores, Molina y otros

    Errores en este ejemplo:

    La fecha debe ser completa (día/mes/año).
    La sede debe ser más precisa, incluyendo la ciudad.
    La dependencia debe contener el nombre completo.
    Los autos deben ser detallados, incluyendo el tipo de caso y las partes involucradas.
    La resolución debe ser clara y específica (número de auto).
    Los nombres de los jueces y de las dependencias deben estar completos, sin abreviaturas.

    Aquí está el texto judicial:
    """

    prompt_sintesis = """Eres un asistente legal que debe crear una síntesis concisa
    y clara de los actos procesales más importantes de una sentencia judicial. 
    A continuación, te proporcionaré el texto de la sentencia. Quiero que 
    redactes la **síntesis de la causa**, resaltando los puntos procesales 
    clave que llevaron al fallo final, utilizando verbos en pretérito. Esta 
    síntesis debe ser objetiva, sin opiniones, y centrarse en las instancias 
    judiciales y decisiones tomadas a lo largo del proceso.
    
    Instrucciones:
    1. Incluye sólo los actos o instancias procesales más importantes e 
    imprescindibles para entender la causa, omitiendo detalles superfluos.
    2. Debes utilizar un tono formal y preciso, sin adjetivos innecesarios.
    3. Si la causa es simple y su fallo se comprende sin secuencia procesal, 
    omite instancias menores, pero siempre incluye las decisiones clave.
    4. Usa siempre verbos en pretérito, ya que describirás eventos que ya 
    ocurrieron.
    5. La síntesis no debe exceder las 100-150 palabras, siendo breve pero 
    informativa.
    6. Si hay apelaciones o recursos, menciona brevemente cómo influyeron en 
    la decisión final.

    Ejemplo de síntesis de un fallo jurídico para tener en cuenta: 
    "En el marco de un juicio ordinario en el que se perseguía el resarcimiento de los 
    daños ocasionados por un accidente de tránsito, el juez a quo admitió 
    parcialmente la pretensión en contra del titular registral y, en cambio, la rechazó 
    respecto del sindicado como conductor del rodado al acoger la defensa de falta de 
    legitimación pasiva interpuesta por éste. Para así resolver tuvo en cuenta la 
    negativa del hecho brindada por éste, la circunstancia que el único testigo no pudo 
    identificarlo y que no se diligenció prueba informativa a la citada en garantía, que 
    hubiera podido corroborar los dichos de la actora. Interpuesto recurso de 
    apelación, la cámara hizo lugar parcialmente al medio impugnativo articulado y 
    determinó la responsabilidad del conductor del vehículo embistente en el siniestro". 

    Ejemplo de lo que **no** debes hacer en la síntesis:
    "La sentencia dictada fue muy larga y complicada, pero básicamente, el 
    juez decidió que uno de los demandados era culpable y el otro no. La 
    actora presentó mucha evidencia, aunque no toda fue convincente. En el 
    proceso hubo muchas apelaciones y los testigos dijeron cosas diferentes. 
    El fallo fue parcialmente a favor de la actora y en contra del otro 
    demandado, con muchos argumentos legales complicados que llevaron mucho 
    tiempo en ser resueltos. Finalmente, el conductor del coche fue 
    responsabilizado de lo que sucedió y tuvo que pagar, aunque hubo varias 
    discusiones sobre cómo se debía calcular el dinero." 

    Aquí está el texto judicial:"""

    prompt_sumarios = """Eres un asistente legal que debe generar un sumario jurisprudencial a partir del siguiente documento judicial, capturando la doctrina clave del fallo. El sumario debe cumplir con las siguientes pautas:

    Doctrina única: El sumario debe centrarse en una sola cuestión jurídica, evitando mezclar múltiples temas o doctrinas.
    Autonomía o autosuficiencia: El sumario debe ser comprensible por sí solo, sin necesidad de consultar el texto completo de la sentencia. Asegúrate de incluir los elementos necesarios para que el sumario sea autosuficiente, como las normas jurídicas relevantes o la conducta en cuestión.
    Abstracción y generalidad: Evita detalles irrelevantes o específicos del caso, como nombres propios o fechas, salvo que sean esenciales para la comprensión de la doctrina. El objetivo es que el sumario sea aplicable a casos similares.
    Claridad y concreción: Redacta el sumario de manera clara, precisa y concreta, utilizando un lenguaje jurídico sencillo y directo.
    Fidelidad: El sumario debe reflejar fielmente el contenido doctrinal de la sentencia, sin interpretaciones personales ni desviaciones del sentido original del fallo.
    Formato obligatorio de salida: El sumario debe seguir estrictamente este formato:

    SUMARIO:

    [Tema principal de la doctrina]

    [Resumen abstracto de la doctrina jurídica, aplicable de forma general]

    ### Ejemplos de referencia para ayudarte (no deben incluirse en la respuesta):

    **Ejemplo correcto** de un sumario:

    SUMARIO:

    RESPONSABILIDAD CIVIL. ACCIDENTE DE TRÁNSITO.

    La denuncia de siniestro ante la compañía de seguros no ostenta, en principio, valor probatorio pleno. Sin embargo, la cercanía temporal entre el choque y la denuncia, y la falta de pruebas de la contraria, generan una presunción a favor del asegurado, basada en las reglas de la sana crítica y la experiencia, que permite condenar al demandado como responsable del accidente. (Art. 1102 del Código Civil).

    **Ejemplo incorrecto** de un sumario:

    SUMARIO:

    RESPONSABILIDAD CIVIL Y PENAL. ACCIDENTE DE TRÁNSITO Y HURTO.

    En este caso, el tribunal determinó que el conductor fue responsable del accidente de tránsito, pero también mencionó que hubo una sospecha de hurto por parte de uno de los testigos. Además, la compañía de seguros no actuó rápidamente y la persona afectada por el accidente buscó reparación por daños psicológicos. No se presentaron evidencias sólidas en relación al hurto, pero el tribunal lo mencionó para futuras investigaciones. (Art. 1102 del Código Civil y Código Penal).

    **Errores en el ejemplo incorrecto**:
    - **Mezcla de doctrinas**: Aborda tanto responsabilidad civil como una acusación de hurto, violando el principio de doctrina única.
    - **Falta de autonomía**: Incluye detalles que no son relevantes y dificultan la comprensión autónoma del sumario.
    - **Exceso de detalles específicos**: Menciona aspectos irrelevantes, como “daños psicológicos” y la “acusación de hurto”.
    - **Falta de claridad**: Mezcla ideas sin centrarse en una doctrina.
    - **Falta de fidelidad**: Introduce temas adicionales que distorsionan la doctrina central del fallo.

    ### Genera el sumario sin incluir los ejemplos anteriores ni sus explicaciones:

    Aquí está el texto judicial:
    """

    #largo_datos = len(prompt_datos) + len(contenido)
    #largo_sintesis = len(prompt_sintesis) + len(contenido)
    #largo_sumarios = len(prompt_sumarios) + len(contenido)

    # Aplicar prompts y obtener selección del usuario
    datos_finales = generar_resumen_claude(contenido, prompt_datos)
    time.sleep(60)
    sintesis_final = generar_resumen_claude(contenido, prompt_sintesis)
    time.sleep(60)
    sumarios_finales = generar_resumen_claude(contenido, prompt_sumarios)

    # Crear documento final
    documento_final = (
        f"DATOS:\n{datos_finales}\n\n"
        f"SÍNTESIS:\n{sintesis_final}\n\n"
        f"SUMARIOS:\n{sumarios_finales}\n"
    )

    time.sleep(60)

    # Evaluar cada aspecto con Claude
    puntajes_aspectos = []
    for aspecto, descripcion in aspectos.items():
        evaluacion = evaluar_aspecto_claude(resumen_original, documento_final, aspecto, descripcion)
        print(f"Evaluación del aspecto '{aspecto}':\n{evaluacion}\n")

        # Extraer puntuación numérica del resultado
        import re
        match = re.search(f"(\\d+)/5", evaluacion)
        if match:
            puntuacion = int(match.group(1))
            puntajes_aspectos.append(puntuacion)
        else:
            print(f"No se pudo extraer una puntuación para el aspecto '{aspecto}'.")

    # Calcular promedio de los puntajes
    if puntajes_aspectos:
        promedio_score = sum(puntajes_aspectos) / len(puntajes_aspectos)
        print(f"Promedio de los puntajes: {promedio_score:.2f}\n")

    # Guardar en un archivo
    #output_path = archivo_seleccionado.replace(".txt", "_resumen_p2_claude_prompt2.txt")
    #with open(output_path, "w") as output_file:
    #    output_file.write(documento_final)

    #print(f"\nResumen final guardado en: {output_path}")

    # Guardar en un json 
    guardar_resultados_json(archivo_seleccionado, documento_final, resumen_original, 
                            puntajes_aspectos[0], puntajes_aspectos[1], puntajes_aspectos[2],
                            puntajes_aspectos[3],  puntajes_aspectos[4], sum(puntajes_aspectos) / len(puntajes_aspectos))

if __name__ == "__main__":
    main()
