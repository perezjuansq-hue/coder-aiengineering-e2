"""Mini-script de prueba asíncrono del pipeline de extracción.

Uso:
    python test_pipeline.py
"""
import asyncio
import logging
import sys

from chain import construir_cadena, process_text

EJEMPLOS = {
    "arquitectura": (
        "Nuestra API está hecha con FastAPI y corre en contenedores Docker sobre Kubernetes. "
        "Usamos Redis como caché de sesiones y PostgreSQL para la persistencia. Desde el último "
        "deploy, en horas pico las conexiones concurrentes a PostgreSQL llegan al máximo del pool "
        "y algunas requests terminan en timeout."
    ),
    "log_de_error": (
        "2026-10-25 03:14:07 ERROR [payments-worker] celery.exceptions.WorkerLostError: Worker "
        "exited prematurely: signal 9 (SIGKILL). RabbitMQ connection reset by peer. "
        "Task charge_customer[8f1c] lost; Stripe webhook not acknowledged. OOMKilled in pod "
        "payments-worker-7d9f (memory limit 512Mi)."
    ),
    # Prueba de estrés: texto ambiguo, sin tecnologías claras ni gravedad explícita
    "ambiguo": "Desde ayer el sistema anda raro, a veces tarda y otras no. Lo de la base quizá tenga que ver.",
}


async def main() -> None:
    # 1) Casos normales, procesados en paralelo
    print("\n=== Casos normales (en paralelo) ===")
    nombres = list(EJEMPLOS)
    resultados = await asyncio.gather(
        *(process_text(EJEMPLOS[nombre]) for nombre in nombres),
        return_exceptions=True,
    )
    for nombre, resultado in zip(nombres, resultados):
        print(f"\n--- {nombre} ---")
        if isinstance(resultado, Exception):
            print(f"[Falló la validación] {type(resultado).__name__}: {resultado}")
        else:
            print(resultado.model_dump_json(indent=2))

    # 2) Prueba de estrés: max_tokens muy bajo para forzar respuestas cortadas.
    #    El pipeline tiene que detectar el finish_reason, reintentar y finalmente fallar de forma controlada.
    print("\n=== Prueba de estrés: respuesta cortada por max_tokens=15 ===")
    cadena_limitada = construir_cadena(max_tokens=15)
    try:
        resultado = await process_text(EJEMPLOS["arquitectura"], cadena=cadena_limitada)
        print(resultado.model_dump_json(indent=2))
    except Exception as e:
        print(f"[Error controlado] {type(e).__name__}: {e}")


if __name__ == "__main__":
    # En Windows la consola puede no usar UTF-8 (mismo arreglo que en la Pre-entrega 1)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silencia los logs de cada request HTTP para que se vean solo los del pipeline
    for nombre in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(nombre).setLevel(logging.WARNING)
    # LangChain loguea un traceback cuando la salida se corta por max_tokens; el pipeline ya lo
    # detecta y lo informa con su propio log, así que se oculta para no duplicar el mensaje.
    logging.getLogger("langchain_core.output_parsers.openai_tools").setLevel(logging.CRITICAL)
    asyncio.run(main())
