from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (descripción de arquitectura, log de error, etc.)."""

    # Quita espacios al inicio y al final de todos los strings antes de validar longitudes
    model_config = ConfigDict(str_strip_whitespace=True)

    tecnologias: List[str] = Field(
        min_length=1,
        description=(
            "Tecnologías mencionadas en el texto: lenguajes, frameworks, bases de datos, "
            "servicios cloud, herramientas. Con su nombre propio, sin repetir."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Gravedad técnica de la situación descrita. 'alta': caída, pérdida de datos o "
            "error en producción; 'media': degradación o riesgo concreto; 'baja': informativo."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=400,
        description="Resumen técnico en español, en una o dos oraciones.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, v: List[str]) -> List[str]:
        """Quita duplicados (sin distinguir mayúsculas) y strings vacíos."""
        vistas = set()
        limpias = []
        for nombre in v:
            if nombre and nombre.lower() not in vistas:
                vistas.add(nombre.lower())
                limpias.append(nombre)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede estar vacía")
        return limpias


class RespuestaIncompletaError(ValueError):
    """El LLM cortó la respuesta (finish_reason = length / max_tokens) o no devolvió un objeto."""
