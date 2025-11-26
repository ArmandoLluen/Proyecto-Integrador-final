# services/excel_service.py
import pandas as pd
import os
from datetime import datetime

import os
import pandas as pd
from datetime import datetime

class ExcelService:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def bool_to_si_no(self, value):
        if isinstance(value, str):
            return value  # si ya es 'sí'/'no', lo deja tal cual
        return "sí" if value else "no"

    def export_messages(self, message_results):
        """
        message_results: list of MessageResult dataclass instances or dicts
        Excel columns:
        Nro Imagen | texto | hostilidad | discriminacion | insultos | lenguaje soez | contenido sexual | amenazas | humillacion | coercion | es acoso | razon | probabilidad
        Todos los booleanos se muestran como 'sí'/'no'.
        """

        rows = []
        for m in message_results:
            if hasattr(m, '__dict__'):
                d = m.__dict__
            else:
                d = dict(m)
            rows.append({
                "Nro Imagen": d.get("nro_imagen"),
                "texto": d.get("texto"),
                "hostilidad": self.bool_to_si_no(d.get("hostilidad")),
                "discriminacion": self.bool_to_si_no(d.get("discriminacion")),
                "insultos": self.bool_to_si_no(d.get("insultos")),
                "lenguaje soez": self.bool_to_si_no(d.get("lenguaje_soez")),
                "contenido sexual": self.bool_to_si_no(d.get("contenido_sexual")),
                "amenazas": self.bool_to_si_no(d.get("amenazas")),
                "humillacion": self.bool_to_si_no(d.get("humillacion")),
                "coercion": self.bool_to_si_no(d.get("coercion")),
                "es acoso": self.bool_to_si_no(d.get("es_acoso")),
                "razon": d.get("razon", ""),
                "probabilidad": d.get("probabilidad", 0.0)
            })
        df = pd.DataFrame(rows)
        filename = f"analisis_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path = os.path.join(self.output_dir, filename)
        df.to_excel(path, index=False)
        return path

