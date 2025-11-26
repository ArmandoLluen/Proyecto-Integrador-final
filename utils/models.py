# utils/models.py
from dataclasses import dataclass

@dataclass
class MessageResult:
    def __init__(
        self,
        nro_imagen: int,
        texto: str,

        # Señales individuales
        hostilidad: bool,
        insultos: bool,
        discriminacion: bool,
        lenguaje_soez: bool,
        contenido_sexual: bool,
        amenazas: bool,
        humillacion: bool,
        coercion: bool,

        # Resultado final
        es_acoso: bool,
        razon: str,
        probabilidad: float = 0.0
    ):
        self.nro_imagen = nro_imagen
        self.texto = texto

        self.hostilidad = hostilidad
        self.insultos = insultos
        self.discriminacion = discriminacion
        self.lenguaje_soez = lenguaje_soez
        self.contenido_sexual = contenido_sexual
        self.amenazas = amenazas
        self.humillacion = humillacion
        self.coercion = coercion

        self.es_acoso = es_acoso
        self.razon = razon
        self.probabilidad = probabilidad

    def to_dict(self):
        return {
            "nro_imagen": self.nro_imagen,
            "texto": self.texto,
            "hostilidad": self.hostilidad,
            "insultos": self.insultos,
            "discriminacion": self.discriminacion,
            "lenguaje_soez": self.lenguaje_soez,
            "contenido_sexual": self.contenido_sexual,
            "amenazas": self.amenazas,
            "humillacion": self.humillacion,
            "coercion": self.coercion,
            "es_acoso": self.es_acoso,
            "razon": self.razon,
            "probabilidad": self.probabilidad
        }