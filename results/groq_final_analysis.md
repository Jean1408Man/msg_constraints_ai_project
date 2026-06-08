# Analisis de resultados: results/groq_final_results.csv

Filas analizadas: 38

Evaluador de restricciones abiertas: llm
Evaluador semantico: llm
Version de prompt: exact-concepts-target-length-v3

## Resumen por estrategia

| strategy | n | validez | hard_score | semantic_score | total_score | violaciones | intentos | errores | generacion_ms | restricciones_ms | semantica_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | 13 | 61.54% | 0.9450 | 0.7942 | 0.8998 | 0.38 | 1.00 | 0 | 5892.48 | 11840.03 | 4582.27 | 22314.89 |
| multi | 12 | 91.67% | 0.9881 | 0.8125 | 0.9354 | 0.08 | 2.00 | 0 | 17113.71 | 16291.98 | 13352.81 | 46758.75 |
| repair | 13 | 92.31% | 0.9890 | 0.8240 | 0.9395 | 0.08 | 1.31 | 0 | 7006.19 | 9177.47 | 8973.01 | 25156.86 |

## Resumen por dificultad

| difficulty | n | validez | hard_score | semantic_score | total_score | violaciones | intentos | errores | generacion_ms | restricciones_ms | semantica_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alta | 8 | 50.00% | 0.9285 | 0.7594 | 0.8778 | 0.50 | 1.50 | 0 | 8221.65 | 10908.02 | 5575.80 | 24705.68 |
| baja | 15 | 93.33% | 0.9905 | 0.8233 | 0.9403 | 0.07 | 1.40 | 0 | 6790.52 | 7140.53 | 6097.41 | 20028.65 |
| media | 15 | 86.67% | 0.9809 | 0.8242 | 0.9339 | 0.13 | 1.40 | 0 | 13694.42 | 18290.61 | 13358.98 | 45344.18 |

## Resumen por estrategia y dificultad

| strategy | difficulty | n | validez | hard_score | semantic_score | total_score | violaciones | intentos | errores | generacion_ms | restricciones_ms | semantica_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct | alta | 3 | 33.33% | 0.9047 | 0.7417 | 0.8558 | 0.67 | 1.00 | 0 | 2860.56 | 6176.25 | 3114.77 | 12151.68 |
| direct | baja | 5 | 80.00% | 0.9714 | 0.8050 | 0.9215 | 0.20 | 1.00 | 0 | 5053.12 | 5996.36 | 5617.69 | 16667.27 |
| direct | media | 5 | 60.00% | 0.9428 | 0.8150 | 0.9045 | 0.40 | 1.00 | 0 | 8551.01 | 21081.97 | 4427.34 | 34060.44 |
| multi | alta | 2 | 50.00% | 0.9285 | 0.7500 | 0.8750 | 0.50 | 2.00 | 0 | 20222.09 | 20221.13 | 12116.78 | 52560.25 |
| multi | baja | 5 | 100.00% | 1.0000 | 0.8350 | 0.9505 | 0.00 | 2.00 | 0 | 8201.64 | 8552.16 | 6968.93 | 23723.00 |
| multi | media | 5 | 100.00% | 1.0000 | 0.8150 | 0.9445 | 0.00 | 2.00 | 0 | 24782.44 | 22460.13 | 20231.10 | 67473.91 |
| repair | alta | 3 | 66.67% | 0.9524 | 0.7833 | 0.9017 | 0.33 | 1.67 | 0 | 5582.44 | 9431.06 | 3676.18 | 18689.96 |
| repair | baja | 5 | 100.00% | 1.0000 | 0.8300 | 0.9490 | 0.00 | 1.20 | 0 | 7116.82 | 6873.06 | 5705.62 | 19695.68 |
| repair | media | 5 | 100.00% | 1.0000 | 0.8425 | 0.9527 | 0.00 | 1.20 | 0 | 7749.82 | 11329.71 | 15418.51 | 34498.19 |

## Restricciones fallidas

| constraint | source | count | example |
| --- | --- | --- | --- |
| longitud | deterministic | 6 | 66 palabras; esperado entre 35 y 60 |
| conceptos_obligatorios | deterministic | 1 | faltan: confirmación |

## Errores registrados

Sin datos.

## Candidatos no validos

| instance_id | strategy | difficulty | error_type | reason | trace | query |
| --- | --- | --- | --- | --- | --- | --- |
| inst_0005 | direct | baja |  | longitud[deterministic]: 66 palabras; esperado entre 35 y 60 | generate#1 valid=False words=66 failed=longitud: 66 palabras; esperado entre 35 y 60; faltante=rango 35-60 | Genera un mensaje de felicitacion para profesor, con tono formal, entre 35 y 60 palabras. Debe incluir ['éxito'] y evitar ['inútil']. |
| inst_0007 | direct | media |  | conceptos_obligatorios[deterministic]: faltan: confirmación | generate#1 valid=False words=40 failed=conceptos_obligatorios: faltan: confirmación; faltante=confirmación | Genera un mensaje de invitacion para cliente, con tono persuasivo, entre 40 y 55 palabras. Debe incluir ['actividad', 'confirmación'] y evitar ['inútil', 'culpa']. |
| inst_0009 | direct | media |  | longitud[deterministic]: 80 palabras; esperado entre 40 y 60 | generate#1 valid=False words=80 failed=longitud: 80 palabras; esperado entre 40 y 60; faltante=rango 40-60 | Genera un mensaje de felicitacion para compañero, con tono formal, entre 40 y 60 palabras. Debe incluir ['esfuerzo', 'reconocimiento'] y evitar ['mentira']. |
| inst_0011 | direct | alta |  | longitud[deterministic]: 50 palabras; esperado entre 55 y 63 | generate#1 valid=False words=50 failed=longitud: 50 palabras; esperado entre 55 y 63; faltante=rango 55-63 | Genera un mensaje de solicitud para profesor, con tono persuasivo, entre 55 y 63 palabras. Debe incluir ['respuesta', 'plazo', 'apoyo'] y evitar ['amenaza', 'insulto', 'inútil']. |
| inst_0011 | multi | alta |  | longitud[deterministic]: 47 palabras; esperado entre 55 y 63 | candidate#1 valid=False words=47 failed=longitud: 47 palabras; esperado entre 55 y 63; faltante=rango 55-63 / candidate#2 valid=False words=50 failed=longitud: 50 palabras; esperado entre 55 y 63; faltante=rango 55-63 | Genera un mensaje de solicitud para profesor, con tono persuasivo, entre 55 y 63 palabras. Debe incluir ['respuesta', 'plazo', 'apoyo'] y evitar ['amenaza', 'insulto', 'inútil']. |
| inst_0013 | direct | alta |  | longitud[deterministic]: 44 palabras; esperado entre 45 y 60 | generate#1 valid=False words=44 failed=longitud: 44 palabras; esperado entre 45 y 60; faltante=rango 45-60 | Genera un mensaje de recordatorio para profesor, con tono formal, entre 45 y 60 palabras. Debe incluir ['tiempo', 'fecha', 'revisión'] y evitar ['fracaso', 'culpa', 'inútil']. D... |
| inst_0013 | repair | alta |  | longitud[deterministic]: 62 palabras; esperado entre 45 y 60 | generate#1 valid=False words=62 failed=longitud: 62 palabras; esperado entre 45 y 60; faltante=rango 45-60 / repair#2 valid=False words=65 failed=longitud: 65 palabras; esperado entre 45 y 60; faltante=rango 45-60 | Genera un mensaje de recordatorio para profesor, con tono formal, entre 45 y 60 palabras. Debe incluir ['tiempo', 'fecha', 'revisión'] y evitar ['fracaso', 'culpa', 'inútil']. D... |
