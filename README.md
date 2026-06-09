# Generacion de mensajes bajo restricciones estructurales

Proyecto hibrido de Inteligencia Artificial y Simulacion para el tema 8: **Generacion de mensajes bajo restricciones estructurales**.

El sistema genera mensajes en espanol que deben cumplir restricciones de longitud, contenido, forma, tono, terminos prohibidos y estructura. El componente generativo usa Groq; despues, una evaluacion hibrida verifica si el mensaje cumple todas las restricciones duras.
Las restricciones objetivas se comprueban con codigo determinista. Las restricciones linguisticas abiertas, como saludo, cuerpo, cierre y agradecimiento final, se evaluan con Groq por defecto porque no pueden enumerarse de forma completa. La calidad semantica tambien puede evaluarse con Groq. Para la entrega final se recomienda `--open-constraint-evaluator llm --semantic-evaluator llm`.

## Modo de generacion

### `groq`

Es el unico generador del proyecto. Usa la API de Groq, por lo que no descarga modelos locales ni requiere ejecutar un servidor en la computadora.

## Instalacion del proyecto

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Configurar Groq

Crea un archivo `.env` en la raiz del proyecto:

```bash
GROQ_API_KEY=tu_api_key_de_groq
GROQ_MODEL=llama-3.1-8b-instant
```

El archivo `.env` queda ignorado por git. Tambien puedes exportar esas variables en la terminal si no quieres usar `.env`.

## Verificar que el LLM responde

```bash
PYTHONPATH=src python3 -m messagegen.cli check-llm
```

Debe aparecer un mensaje indicando que Groq respondio.

## Generar un mensaje

```bash
PYTHONPATH=src python3 -m messagegen.cli generate-one \
  --strategy unified \
  --open-constraint-evaluator llm \
  --semantic-evaluator llm
```

Para probar otro modelo sin editar `.env`:

```bash
PYTHONPATH=src python3 -m messagegen.cli generate-one --model otro-modelo --strategy unified
```

## Crear dataset simulado

```bash
PYTHONPATH=src python3 -m messagegen.cli make-dataset --per-difficulty 5 --out data/groq_final_instances.jsonl --seed 7
```

`--per-difficulty 5` genera 15 instancias balanceadas: 5 baja, 5 media y 5 alta.

## Ejecutar experimento con Groq

Recomendacion: empezar con pocas instancias para no gastar llamadas innecesarias de API.

```bash
PYTHONPATH=src python3 -m messagegen.cli run-experiment \
  --dataset data/groq_final_instances.jsonl \
  --out results/groq_final_results.csv \
  --generator groq \
  --model llama-3.1-8b-instant \
  --open-constraint-evaluator llm \
  --semantic-evaluator llm \
  --max-attempts 3 \
  --population-size 3 \
  --resume
```

Analizar resultados:

```bash
python3 scripts/analyze_results.py \
  --csv results/groq_final_results.csv \
  --out results/groq_final_analysis.md
```

`--resume` solo conserva filas compatibles con los evaluadores actuales. Si el CSV viene de una corrida vieja sin `open_constraint_evaluator=llm`, el experimento nuevo se regenerara en el formato actualizado.

Durante `run-experiment`, la CLI muestra progreso por cada combinacion `instancia + estrategia`: fila actual, barra de avance, validez, violaciones, intentos y tiempo. Si necesitas una salida silenciosa para scripts, agrega `--no-progress`.

La estrategia por defecto es `unified`. Primero genera varios candidatos (`--population-size`), selecciona el mejor por la funcion objetivo y, si todavia no es valido, lo repara hasta `--max-attempts` veces antes de devolver el mejor resultado encontrado.

El CSV guarda trazabilidad para depuracion:

- `query`: solicitud textual original de la instancia.
- `prompt_version`: version de los prompts usados para generar y reparar.
- `instance_json`: instancia completa con restricciones.
- `attempt_trace`: candidatos y reparaciones intentadas, con conteo de palabras y restricciones fallidas.
- `failed_constraints`: restricciones fallidas en JSON.
- `invalid_reason`: explicacion legible de por que no fue valido.
- `error_type` y `error_message`: excepcion registrada si una fila fallo por API, red o parsing.
- `evaluation_notes`: notas del evaluador semantico.

Si una fila falla por error, se registra en el CSV como `valid=False` y el experimento continua con la siguiente combinacion.
Si cambia `prompt_version`, `--resume` no reutiliza filas antiguas, para evitar mezclar resultados de prompts diferentes.

## Estrategias disponibles

- `unified`: genera varios candidatos, selecciona el mejor y repara ese candidato hasta cumplir o agotar el tope configurable.
- `direct`: genera un solo mensaje y lo evalua.
- `verify`: genera varios intentos y se detiene cuando encuentra uno valido.
- `repair`: genera, verifica y repara usando retroalimentacion del verificador.
- `multi`: genera varios candidatos y selecciona el mejor por funcion objetivo; se conserva como estrategia heredada.

## Archivos principales

```text
src/messagegen/models.py        Modelos de datos
src/messagegen/simulator.py     Simulacion de instancias
src/messagegen/constraints.py   Checks deterministas: longitud, conceptos y prohibidos
src/messagegen/evaluator.py     Restricciones abiertas LLM, funcion objetivo y evaluacion
src/messagegen/llm.py           Generador Groq
src/messagegen/strategies.py    Estrategias de generacion
src/messagegen/experiment.py    Comparacion experimental
src/messagegen/cli.py           Interfaz por terminal
scripts/analyze_results.py      Analisis agregado de resultados
```

## Descripcion para el informe

> El sistema usa un LLM mediante la API de Groq como componente generativo, reparador, juez de restricciones abiertas y evaluador semantico. El codigo determinista comprueba longitud, conceptos obligatorios y terminos prohibidos. Groq evalua como restricciones duras los aspectos linguisticos abiertos: saludo, cuerpo, cierre y agradecimiento final. La estrategia unificada genera una poblacion inicial de candidatos, selecciona el mejor mediante una funcion objetivo y, si no es valido, lo repara con retroalimentacion estructurada del verificador hasta alcanzar un tope configurable.
