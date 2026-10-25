# coder-aiengineering-e2

**Pre-entrega 2: Pipeline de extracción de entidades técnicas.** Recibe un párrafo de texto sin procesar (una descripción de arquitectura, un log de error) y devuelve un objeto **validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico. Está construido con **LangChain (LCEL)** y funciona con **OpenAI** o **Anthropic**: el proveedor se elige desde el `.env`, igual que en la [Pre-entrega 1](https://github.com/perezjuansq-hue/coder-aiengineering-e1).

## Cómo funciona

```
texto ──► ChatPromptTemplate ──► LLM.with_structured_output(ExtraccionTecnica) ──► validar_respuesta ──► ExtraccionTecnica
          (instrucciones +        (JSON según el esquema Pydantic)                 (finish_reason,
           {texto})                                                                  errores de parseo)
          └──────────────────────────────── .with_retry(stop_after_attempt=3) ──────────────────────────┘
```

| Archivo | Contenido |
|---|---|
| `schemas.py` | Contrato de salida: `ExtraccionTecnica` (Pydantic), el enum `NivelCriticidad` y la excepción `RespuestaIncompletaError`. |
| `chain.py` | Cliente LLM, `ChatPromptTemplate`, cadena LCEL con reintentos y la función asíncrona `process_text()`. |
| `test_pipeline.py` | Mini-script de prueba asíncrono: casos normales, un texto ambiguo y una prueba de estrés. |

### Esquema de salida (`schemas.py`)

| Campo | Tipo | Restricciones |
|---|---|---|
| `tecnologias` | `list[str]` | Al menos un elemento. Sin duplicados (sin distinguir mayúsculas) ni strings vacíos. |
| `nivel_de_criticidad` | enum | Solo `baja`, `media` o `alta`. |
| `resumen_tecnico` | `str` | Entre 20 y 400 caracteres. |

Todos los strings se recortan (*strip*) antes de validarse. Las descripciones de cada campo (`Field(description=...)`) viajan al modelo como parte del JSON schema, así que también funcionan como instrucciones.

### Prompt template

El prompt es un `ChatPromptTemplate` con dos variables: `{texto}` (la entrada) y `{format_instructions}` (las reglas de formato, cargadas con `.partial()`). No se usan f-strings: LangChain gestiona las variables, y las reglas se pueden cambiar sin tocar la cadena.

### Cadena LCEL

```python
modelo_estructurado = modelo.with_structured_output(ExtraccionTecnica, include_raw=True)

cadena = (
    RunnableLambda(_log_intento)
    | prompt
    | modelo_estructurado
    | RunnableLambda(validar_respuesta)
).with_retry(
    retry_if_exception_type=debe_reintentar,
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

Es la forma `prompt | model.with_structured_output(Schema)` de la consigna, con un paso de validación al final. `include_raw=True` hace que el modelo devuelva también el mensaje crudo, y gracias a eso `validar_respuesta` puede revisar el `finish_reason` **antes** de aceptar el objeto.

### Resiliencia

`validar_respuesta` lanza una excepción en estos casos, y `.with_retry()` vuelve a ejecutar la cadena (hasta 3 intentos, con espera exponencial):

| Situación | Cómo se detecta |
|---|---|
| Respuesta cortada por límite de tokens | `finish_reason == "length"` (OpenAI) o `stop_reason == "max_tokens"` (Anthropic) → `RespuestaIncompletaError`. El SDK de OpenAI también puede lanzar `LengthFinishReasonError` por su cuenta. |
| JSON mal formado o que no cumple el esquema | `parsing_error` de `with_structured_output` (`OutputParserException` / `ValidationError`). |
| El modelo no devolvió ningún objeto | `parsed is None` → `RespuestaIncompletaError`. |

Solo se reintentan esos errores. Los errores de autenticación o de configuración se propagan enseguida, porque repetir la llamada no los arregla. Si después de 3 intentos no hay una respuesta válida, `process_text()` registra el error y relanza la excepción.

### Logs

Cada paso deja un log: el texto que se procesa, cada llamada al LLM, el `finish_reason` recibido, el resultado de la validación, cada intento fallido y el resultado final.

## Requisitos

- Python **3.12**
- Una API key de OpenAI y/o de Anthropic.

## Instalación

```bash
git clone https://github.com/perezjuansq-hue/coder-aiengineering-e2.git
cd coder-aiengineering-e2

py -3.12 -m venv .venv          # Windows
# python3.12 -m venv .venv      # macOS / Linux

source .venv/Scripts/activate   # Windows (Git Bash)
# .venv\Scripts\Activate.ps1    # Windows (PowerShell)
# source .venv/bin/activate     # macOS / Linux

python -m pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | Sí | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si `LLM_PROVIDER=openai` | API key de OpenAI |
| `ANTHROPIC_API_KEY` | Si `LLM_PROVIDER=anthropic` | API key de Anthropic |
| `OPENAI_MODEL` | No | Por defecto: `gpt-6-luna` |
| `ANTHROPIC_MODEL` | No | Por defecto: `claude-haiku-4-5-20251001` |

> El archivo `.env` está en `.gitignore` y no se sube al repositorio.

## Uso

### Mini-script de prueba

```bash
python test_pipeline.py
LLM_PROVIDER=anthropic python test_pipeline.py    # cambia de proveedor solo para esta ejecución
```

El script hace dos cosas:
1. Procesa **en paralelo** (`asyncio.gather`) tres textos: una descripción de arquitectura, un log de error y un texto ambiguo.
2. Corre una **prueba de estrés**: arma la cadena con `max_tokens=15` para forzar respuestas cortadas y comprobar que el pipeline las detecta, reintenta y termina con un error controlado.

### Desde tu código

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("Nuestra API en FastAPI usa Redis como caché..."))
print(resultado.nivel_de_criticidad)   # NivelCriticidad.ALTA
print(resultado.model_dump_json())
```

## Ejemplo de salida

Entrada:

> Nuestra API está hecha con FastAPI y corre en contenedores Docker sobre Kubernetes. Usamos Redis como caché de sesiones y PostgreSQL para la persistencia. Desde el último deploy, en horas pico las conexiones concurrentes a PostgreSQL llegan al máximo del pool y algunas requests terminan en timeout.

Salida validada (OpenAI, `gpt-6-luna`):

```json
{
  "tecnologias": ["FastAPI", "Docker", "Kubernetes", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "En horas pico, el pool de conexiones de PostgreSQL alcanza su máximo y algunas solicitudes terminan en timeout tras el último despliegue."
}
```

Logs de una ejecución (casos normales en paralelo y luego la prueba de estrés):

```
17:33:48 | INFO    | Proveedor: openai | Modelo: gpt-6-luna
17:33:48 | INFO    | Procesando texto (299 caracteres)
17:33:48 | INFO    | Llamando al LLM...
17:33:51 | INFO    | Respuesta recibida (finish_reason=stop)
17:33:51 | INFO    | Validación Pydantic OK
17:33:51 | INFO    | Extracción completada: 5 tecnologías, criticidad=alta
...
17:33:52 | INFO    | Llamando al LLM...
17:33:53 | WARNING | Intento fallido (LengthFinishReasonError): Could not parse response content as the length limit was reached ...
17:33:54 | INFO    | Llamando al LLM...
17:33:55 | WARNING | Intento fallido (LengthFinishReasonError): ...
17:33:58 | INFO    | Llamando al LLM...
17:33:59 | WARNING | Intento fallido (LengthFinishReasonError): ...
17:33:59 | ERROR   | Sin respuesta válida tras 3 intentos: ...
```

Con Anthropic, la prueba de estrés se detecta en `validar_respuesta`:

```
17:33:06 | INFO    | Respuesta recibida (finish_reason=max_tokens)
17:33:06 | WARNING | Respuesta cortada por límite de tokens -> el objeto está incompleto, se descarta
17:33:06 | WARNING | Intento fallido (RespuestaIncompletaError): respuesta incompleta (finish_reason=max_tokens)
```

### Resultado de la prueba de estrés con texto ambiguo

Entrada: *"Desde ayer el sistema anda raro, a veces tarda y otras no. Lo de la base quizá tenga que ver."*

Los dos proveedores se recuperaron en el primer intento y devolvieron un objeto válido. Como el texto no nombra ninguna tecnología concreta, el modelo completó la lista con una categoría genérica (`"Base de datos"`) para cumplir la restricción de lista no vacía, y asignó criticidad `media`:

```json
{
  "tecnologias": ["Base de datos"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "El sistema presenta demoras intermitentes y un comportamiento irregular desde ayer. Se sospecha que la base de datos podría estar relacionada, pero no se identifica una tecnología específica."
}
```
