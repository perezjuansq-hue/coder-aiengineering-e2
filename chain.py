import logging
import os
from enum import Enum
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import ValidationError

from schemas import ExtraccionTecnica, RespuestaIncompletaError

logger = logging.getLogger("pipeline")


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Modelo que se usa si OPENAI_MODEL / ANTHROPIC_MODEL están vacías o no existen (igual que en el Módulo 1)
MODELOS_POR_DEFECTO = {
    Provider.OPENAI: "gpt-6-luna",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
}

# Valores de finish_reason (OpenAI) / stop_reason (Anthropic) que indican que la respuesta se cortó
MOTIVOS_DE_CORTE = {"length", "max_tokens"}

# Errores que justifican un reintento: JSON mal formado, objeto que no pasa la validación o respuesta cortada.
# LengthFinishReasonError lo lanza el SDK de OpenAI cuando corta la salida estructurada por max_tokens.
ERRORES_REINTENTABLES = (
    OutputParserException,
    ValidationError,
    RespuestaIncompletaError,
    LengthFinishReasonError,
)

MAX_INTENTOS = 3


# --- Prompt template -------------------------------------------------------------------------

INSTRUCCIONES_SISTEMA = """Sos un ingeniero de software senior que analiza textos técnicos.
Tu tarea es extraer las entidades técnicas del texto que te pasa el usuario.

Reglas de formato:
{format_instructions}"""

REGLAS_DE_FORMATO = """- tecnologias: lista NO vacía con los nombres propios de las tecnologías mencionadas \
(lenguajes, frameworks, bases de datos, servicios, herramientas). Sin duplicados.
- nivel_de_criticidad: exactamente uno de "baja", "media" o "alta".
  * alta: caída del servicio, pérdida de datos, error en producción o problema de seguridad.
  * media: degradación de rendimiento o un riesgo concreto que todavía no causó una falla.
  * baja: descripción informativa, sin problema aparente.
- resumen_tecnico: una o dos oraciones en español (entre 20 y 400 caracteres).
- Si el texto es ambiguo, inferí la criticidad a partir de las pistas disponibles; no inventes \
tecnologías que no estén mencionadas."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCCIONES_SISTEMA),
        ("human", "Texto a analizar:\n\n{texto}"),
    ]
).partial(format_instructions=REGLAS_DE_FORMATO)


# --- Cliente LLM (misma lógica de configuración que el Módulo 1) ------------------------------

def crear_modelo(max_tokens: int = 1024) -> BaseChatModel:
    """Lee el .env y crea el cliente de chat de LangChain para el proveedor configurado."""
    load_dotenv()

    valor_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not valor_provider:
        raise ValueError("Falta LLM_PROVIDER (valores aceptados: openai, anthropic)")
    provider = Provider(valor_provider)

    if provider == Provider.OPENAI:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Falta OPENAI_API_KEY en el .env")
        modelo = os.getenv("OPENAI_MODEL") or MODELOS_POR_DEFECTO[provider]
        logger.info("Proveedor: openai | Modelo: %s", modelo)
        return ChatOpenAI(model=modelo, max_tokens=max_tokens)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Falta ANTHROPIC_API_KEY en el .env")
    modelo = os.getenv("ANTHROPIC_MODEL") or MODELOS_POR_DEFECTO[provider]
    logger.info("Proveedor: anthropic | Modelo: %s", modelo)
    return ChatAnthropic(model=modelo, max_tokens=max_tokens)


# --- Validación de la respuesta ----------------------------------------------------------------

def validar_respuesta(resultado: Dict[str, Any]) -> ExtraccionTecnica:
    """Revisa la salida de with_structured_output(include_raw=True) antes de aceptarla.

    1. Si el modelo cortó la respuesta por falta de tokens, el objeto está incompleto: se descarta.
    2. Si el JSON no se pudo parsear o no pasó la validación de Pydantic, se relanza el error.
    Cualquiera de estas excepciones hace que .with_retry() vuelva a ejecutar la cadena.
    """
    raw = resultado["raw"]
    metadata = raw.response_metadata
    motivo = metadata.get("finish_reason") or metadata.get("stop_reason")
    logger.info("Respuesta recibida (finish_reason=%s)", motivo)

    if motivo in MOTIVOS_DE_CORTE:
        logger.warning("Respuesta cortada por límite de tokens -> el objeto está incompleto, se descarta")
        raise RespuestaIncompletaError(f"respuesta incompleta (finish_reason={motivo})")

    error = resultado.get("parsing_error")
    if error is not None:
        logger.warning("JSON inválido o fuera del esquema")
        raise error

    parsed = resultado.get("parsed")
    if parsed is None:
        logger.warning("El modelo no devolvió ningún objeto estructurado")
        raise RespuestaIncompletaError("el modelo no devolvió un objeto estructurado")

    logger.info("Validación Pydantic OK")
    return parsed


def _log_intento(entrada: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Llamando al LLM...")
    return entrada


def debe_reintentar(error: BaseException) -> bool:
    """Criterio de .with_retry(): reintenta solo ante errores de formato o respuestas incompletas.

    Errores como una API key inválida no se reintentan: repetir la llamada no los arregla.
    """
    reintentable = isinstance(error, ERRORES_REINTENTABLES)
    if reintentable:
        logger.warning("Intento fallido (%s): %s", type(error).__name__, str(error)[:200])
    return reintentable


# --- Cadena LCEL --------------------------------------------------------------------------------

def construir_cadena(max_tokens: int = 1024) -> Runnable:
    """prompt | LLM con salida estructurada | validación, envuelto en una política de reintentos."""
    modelo = crear_modelo(max_tokens=max_tokens)

    # include_raw=True devuelve {"raw", "parsed", "parsing_error"} en lugar de solo el objeto:
    # así podemos mirar el finish_reason del mensaje crudo antes de aceptar el resultado.
    modelo_estructurado = modelo.with_structured_output(ExtraccionTecnica, include_raw=True)

    cadena = (
        RunnableLambda(_log_intento)
        | prompt
        | modelo_estructurado
        | RunnableLambda(validar_respuesta)
    )

    return cadena.with_retry(
        retry_if_exception_type=debe_reintentar,
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=True,
    )


_cadena: Runnable | None = None


def _obtener_cadena() -> Runnable:
    """Crea la cadena la primera vez que se usa y la reutiliza en las siguientes llamadas."""
    global _cadena
    if _cadena is None:
        _cadena = construir_cadena()
    return _cadena


async def process_text(text: str, cadena: Runnable | None = None) -> ExtraccionTecnica:
    """Ejecuta el pipeline de forma asíncrona y devuelve un objeto ExtraccionTecnica validado.

    Lanza la última excepción si después de MAX_INTENTOS el modelo no logra una respuesta válida.
    """
    if not text or not text.strip():
        raise ValueError("El texto de entrada está vacío")

    cadena = cadena or _obtener_cadena()
    logger.info("Procesando texto (%d caracteres)", len(text))
    try:
        resultado = await cadena.ainvoke({"texto": text})
    except ERRORES_REINTENTABLES as e:
        logger.error("Sin respuesta válida tras %d intentos: %.200s", MAX_INTENTOS, e)
        raise
    logger.info("Extracción completada: %d tecnologías, criticidad=%s",
                len(resultado.tecnologias), resultado.nivel_de_criticidad.value)
    return resultado
