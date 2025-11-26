# services/detection_service.py
from turtle import color
from typing import List, Dict, Tuple
import re
import statistics
import cv2
import numpy as np
from sklearn.cluster import KMeans

class DetectionService:
    """
    Detección de burbujas (mensajes) usando:
    - color promedio del fondo de la burbuja (gris/blanco)
    - distancia vertical
    - cambio de alineación horizontal
    - detección de timestamps como separadores
    - blacklist
    """

    BLACKLIST = [
        "hora de últ. vez",
        "los mensajes y las llamadas están cifrados",
        "toca para obtener más información",
        "activaste los mensajes temporales",
        "desaparecerán de este chat",
        "mensajes temporales",
        "de extremo a extremo",
        "los mensajes y las llamadas están cifrados de extremo a extremo",
        "toca para obtener más información sobre el cifrado de extremo a extremo",
        "este chat está cifrado de extremo a extremo",
        "este grupo está cifrado de extremo a extremo",
        "toca para cambiar esta opción",
        "nadie fuera de este chat, ni siquiera whatsapp",
        "puede leerlos ni escucharlos",
    ]

    def __init__(self,
                 min_chars: int = 3,
                 gray_diff: int = 40,   # más permisivo
                 padding: int = 8):
        
        self.min_chars = min_chars
        self.timestamp_re = re.compile(
                                r"\b\d{1,2}:\d{2}(?:\s*(?:a|p)\.?m\.?)?(?:\s*V[\/\.]*)?\b",
                                re.IGNORECASE
                            )
        self.gray_diff = gray_diff
        self.padding = padding
        self._only_symbols_re = re.compile(r'^[\d\W_]+$')

    # -------------------- UTILIDADES --------------------
    def _is_blacklist(self, text: str) -> bool:
        if not text or len(text.strip()) <= 1:
            return True
        tl = text.lower().strip()
        for bad in self.BLACKLIST:
            if bad.lower() in tl:
                return True
        return False

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        t = text

        # 1. Remover timestamps
        t = re.sub(r"\b\d{1,2}:\d{2}(?:\s*(?:a|p)\.?m\.?)?(?:\s*V[\/\.]*)?\b", "", t, flags=re.IGNORECASE)

        # 2. Remover estados tipo V/, V//, V., ✓, etc
        t = re.sub(r"\bV[\/\.\-]*\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bJ[\/\.\-]*\b", "", t, flags=re.IGNORECASE)
        t = t.replace("✓", "").replace("✔", "")

        # 3. Remover puntos repetidos (.. ... ....)
        t = re.sub(r"\.{2,}", ".", t)

        # 4. Remover puntos o comas iniciales generadas por join
        t = re.sub(r"^[\.\,\-\_]+", "", t).strip()

        # 5. Normalizar espacios múltiples
        t = re.sub(r"\s{2,}", " ", t)

        # 6. Remover espacios antes de puntuación
        t = re.sub(r"\s+([,.!?])", r"\1", t)

        # 7. Recortar espacios finales / inicio
        return t.strip()

    def _is_noise_text(self, text: str) -> bool:

        if not text or not text.strip():
            return True

        cleaned = self._clean_text(text).strip()
        if not cleaned:
            return True

        cleaned_low = cleaned.lower()

        # 0. Basura tipo vectores / RGB → (0 0 0), 0 0 0, (1,1,1)
        if re.fullmatch(r"[\(\[\{]?\s*\d\s+\d\s+\d\s*[\)\]\}]?", cleaned_low):
            return True

        if re.fullmatch(r"[\(\[\{]?\s*\d\s*[,\s]\s*\d\s*[,\s]\s*\d\s*[\)\]\}]?", cleaned_low):
            return True

        if re.fullmatch(r"(\d\s+){1,4}\d", cleaned_low):
            return True

        if re.fullmatch(r"\d[\.\, ]+\d[\.\, ]+\d", cleaned_low):
            return True

        # 1. solo puntuación
        if all(c in ".·,;:¡¿" for c in cleaned_low):
            return True

        # 2. OCR puntos
        if cleaned_low in {".", "..", "...", "...."}:
            return True

        tokens = cleaned_low.split()

        # 3. **Palabras válidas incluso si son cortas**
        allowed_short = {
            "si","sí","no","ok","ya","va","ey","hi","yo","eh","así","qué","qué?","de",
            "la","el","y","o","tu","tú","me","te","es","al","ja","ajá","ajá.","sí.","no.","ok.","ok!",
            "xq","pq","xd","jaj","jaja","jajaja","wtf","lol","brb","btw",
            "ahora","donde","dónde","cuando","cuándo",
            "quien","quién","como","cómo","q","x","xq","pq"
        }
        if any(t in allowed_short for t in tokens):
            return False

        # 4. Si NO hay ninguna palabra con >=3 letras PERO existe una palabra >=2 → válida
        if not any(len(t) >= 3 for t in tokens):
            # si hay una palabra de 2 letras → NO ruido (“ya”, “ok”, “ah”, etc.)
            if any(len(t) == 2 for t in tokens):
                return False
            # si hay una palabra de 1 letra → depende
            if any(len(t) == 1 for t in tokens):
                # una sola letra → ruido
                if len(tokens) == 1:
                    return True
                return False
            return True

        # 5. números extraños
        if cleaned_low.isdigit():
            if len(cleaned_low) == 1:
                return False
            if 2 <= len(cleaned_low) <= 3:
                return False
            if len(cleaned_low) > 4:
                return True

        # 6. mezcla número/letra “1a”, “a1”
        if re.fullmatch(r'\d+[a-zA-Z]+|[a-zA-Z]+\d+', cleaned_low):
            return True

        return False

    def _get_bubble_average_color(self, bubble, img):
        """Calcula el color promedio de una burbuja dada su región en la imagen."""
        if img is None or img.size == 0:
            return None

        # =============================
        # 1. Obtener ROI original
        # =============================
        tops = [ln["top"] for ln in bubble["lines"]]
        lefts = [ln["left"] for ln in bubble["lines"]]
        bottoms = [ln["top"] + ln["height"] for ln in bubble["lines"]]
        rights = [ln["left"] + ln["width"] for ln in bubble["lines"]]

        # Padding adaptativo
        h_vals = [ln["height"] for ln in bubble["lines"]]
        h_mean = np.mean(h_vals) if h_vals else 20
        pad = int(h_mean * 0.5)
        pad = max(5, min(pad, 20))

        top    = max(min(tops) - pad, 0)
        left   = max(min(lefts) - pad, 0)
        bottom = min(max(bottoms) + pad, img.shape[0])
        right  = min(max(rights) + pad, img.shape[1])

        roi = img[top:bottom, left:right]
        if roi.size == 0:
            return None

        # =============================
        # 2. Recorte central para eliminar texto y bordes 
        # =============================
        h, w = roi.shape[:2]

        # recorte central
        cy1 = int(h * 0.25)
        cy2 = int(h * 0.75)
        cx1 = int(w * 0.15)
        cx2 = int(w * 0.85)

        if cy2 <= cy1 or cx2 <= cx1:
            return None

        core = roi[cy1:cy2, cx1:cx2]
        if core.size == 0:
            return None

        # =============================
        # 3. Convertir a LAB para eliminar sombras blancas/negras
        # =============================
        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)

        # mantener solo fondo (ni muy oscuro ni muy brillante)
        mask = (lab[:,:,0] > 60) & (lab[:,:,0] < 240)
        clean_pixels = core[mask]

        if clean_pixels.size == 0:
            return None

        # =============================
        # 4. Promedio final del fondo
        # =============================
        mean_color = clean_pixels.mean(axis=0)

        return tuple(int(c) for c in mean_color)
    
    def is_pure_timestamp(self, text: str) -> bool:
        '''
        Comprueba si el texto es un timestamp puro según la expresión regular definida.
        '''
        txt = text.strip()
        return bool(self.timestamp_re.fullmatch(txt))
    
    #--------------------- AGRUPACIÓN DE BURBUJAS --------------------
    def _group_lines_to_bubbles(self, lines: List[Dict], img: np.ndarray) -> List[Dict]:
        """
        función principal para agrupar líneas OCR en burbujas de mensajes
        lines: lista de líneas OCR con 'text', 'top', 'left', 'height', 'width'
        img: imagen original en formato numpy array (BGR)
        returns: lista de burbujas detectadas con 'text', 'top', 'left', 'bottom', 'right', 'color'
        Cada burbuja contiene varias líneas agrupadas.
        """

        if not lines:
            return []

        # ------------------------------------------------------
        # PRINT DE DEPURACIÓN
        # ------------------------------------------------------
        print(f"Total líneas OCR a procesar: {len(lines)}")
        print("-" * 50)
        for l in lines:
            print(f"  - {l['text']} (top: {l['top']}, left: {l['left']}, h:{l.get('height',0)} w:{l.get('width',0)})")
        print("-" * 50)

        # ------------------------------------------------------
        # CALCULAR ALTURA MEDIA
        # ------------------------------------------------------
        heights = [max(6, l.get("height", 0)) for l in lines]
        median_h = statistics.median(heights)
        median_h = max(14, min(median_h, 32))

        # ------------------------------------------------------
        # CALCULAR TOLERANCIA GLOBAL DE COLOR
        # ------------------------------------------------------
        def is_white_bubble(c):
            if c is None:
                return False
            return c[0] > 180 and c[1] > 180 and c[2] > 180

        # obtener colores rápidos
        sample_colors = []
        for l in lines[:min(40, len(lines))]:
            c = self._get_bubble_average_color({"lines": [l]}, img)
            if c is not None:
                sample_colors.append(c)

        # TOLERANCIA BASE
        COLOR_TOL = 30
        WHITE_TOL = 50

        if len(sample_colors) > 4:
            arr = np.array(sample_colors, np.uint8).reshape(-1, 1, 3)
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
            diffs = [np.linalg.norm(lab[i] - lab[i+1]) for i in range(len(lab)-1)]
            base = np.median(diffs) if diffs else 10
            COLOR_TOL = max(18, min(base * 2.7, 32))

        # ------------------------------------------------------
        # ORDENAR LÍNEAS
        # ------------------------------------------------------
        lines_sorted = sorted(lines, key=lambda x: (x["top"], x["left"]))

        # ------------------------------------------------------
        # DETECTAR LADO
        # ------------------------------------------------------
        xs = np.array([l["left"] for l in lines_sorted], float)
        thr = (np.percentile(xs, 33) + np.percentile(xs, 66)) / 2 if len(xs) >= 4 else np.median(xs)
        labels = np.array([0 if x <= thr else 1 for x in xs])

        # ------------------------------------------------------
        # LOOP PRINCIPAL
        # ------------------------------------------------------
        bubbles = []
        cur = None
        cur_side = None
        cur_colors = []        # <-- acumulación de colores
        cur_color_lab = None

        for idx, l in enumerate(lines_sorted):

            l_top = l["top"]
            l_left = l["left"]
            l_h = max(6, l.get("height", median_h))
            l_bottom = l_top + l_h
            this_side = int(labels[idx])

            # ==========================================================
            # BLOQUE A — DETECTAR TIMESTAMP
            # ==========================================================
            raw = l["text"].strip().lower()
            raw_norm = re.sub(r"[^0-9apm:]", "", raw)
            raw_norm = raw_norm.replace("am","a.m.").replace("pm","p.m.")

            is_timestamp = False

            if re.fullmatch(r"\d{1,2}:\d{2}", raw_norm):
                is_timestamp = True
            elif re.fullmatch(r"\d{1,2}:\d{2}(a\.?m\.?|p\.?m\.?)", raw_norm):
                is_timestamp = True
            elif re.fullmatch(r"\d{1,2}:\d{2}\.*", raw_norm):
                is_timestamp = True

            if is_timestamp:
                if cur is not None:

                    # calcular color promedio final de la burbuja
                    if cur_colors:
                        avg = tuple(int(c) for c in np.mean(cur_colors, axis=0))
                        cur["color"] = avg

                    cur["text"] = " ".join(x["text"] for x in cur["lines"])
                    bubbles.append(cur)

                cur = None
                cur_side = None
                cur_colors = []
                cur_color_lab = None
                continue

            # ===================================================
            # BLOQUE B — INICIO DE NUEVA BURBUJA
            # ===================================================
            if cur is None:
                first_color = self._get_bubble_average_color({"lines": [l]}, img)

                cur = {
                    "top": l_top, "left": l_left,
                    "bottom": l_bottom, "right": l_left + l.get("width", 0),
                    "lines": [l]
                }

                if first_color is not None:
                    cur_colors = [first_color]
                    cur_color_lab = cv2.cvtColor(np.uint8([[first_color]]), cv2.COLOR_BGR2LAB)[0][0]
                else:
                    cur_colors = []
                    cur_color_lab = None

                cur_side = this_side
                continue

            # ===================================================
            # BLOQUE C — DECISIÓN: ¿NUEVA BURBUJA O MISMA?
            # ===================================================
            new_color = self._get_bubble_average_color({"lines": [l]}, img)
            if new_color is not None:
                new_lab = cv2.cvtColor(np.uint8([[new_color]]), cv2.COLOR_BGR2LAB)[0][0]
            else:
                new_lab = None

            deltaE = np.linalg.norm(cur_color_lab - new_lab) if (cur_color_lab is not None and new_lab is not None) else 0

            # Blancas
            if cur_colors and is_white_bubble(cur_colors[-1]) and is_white_bubble(new_color):
                if deltaE < WHITE_TOL:
                    cur["lines"].append(l)
                    cur["bottom"] = max(cur["bottom"], l_bottom)
                    cur["right"] = max(cur["right"], l_left + l.get("width", 0))
                    if new_color is not None:
                        cur_colors.append(new_color)
                    continue

            # Delta pequeño → misma burbuja
            if deltaE < COLOR_TOL:
                cur["lines"].append(l)
                cur["bottom"] = max(cur["bottom"], l_bottom)
                cur["right"] = max(cur["right"], l_left + l.get("width", 0))
                if new_color is not None:
                    cur_colors.append(new_color)
                continue

            # Cambio fuerte de lado
            if this_side != cur_side and deltaE > COLOR_TOL * 0.8:

                if cur_colors:
                    avg = tuple(int(c) for c in np.mean(cur_colors, axis=0))
                    cur["color"] = avg

                cur["text"] = " ".join(x["text"] for x in cur["lines"])
                bubbles.append(cur)

                cur = {
                    "top": l_top, "left": l_left,
                    "bottom": l_bottom, "right": l_left + l.get("width", 0),
                    "lines": [l]
                }

                cur_colors = [new_color] if new_color is not None else []
                cur_color_lab = new_lab
                cur_side = this_side
                continue

            # Salto horizontal

            cur_widths = [ln.get("width", 0) for ln in cur["lines"]]
            mean_w = np.mean(cur_widths) if cur_widths else 120

            # 45% del ancho promedio de la burbuja
            HORIZ_GAP = int(mean_w * 0.45)

            # límites razonables para evitar extremos
            HORIZ_GAP = max(60, min(HORIZ_GAP, 260))

            dx = abs(l_left - cur["left"])
            if dx > HORIZ_GAP:

                if cur_colors:
                    avg = tuple(int(c) for c in np.mean(cur_colors, axis=0))
                    cur["color"] = avg

                cur["text"] = " ".join(x["text"] for x in cur["lines"])
                bubbles.append(cur)

                cur = {
                    "top": l_top, "left": l_left,
                    "bottom": l_bottom, "right": l_left + l.get("width", 0),
                    "lines": [l]
                }

                cur_colors = [new_color] if new_color is not None else []
                cur_color_lab = new_lab
                cur_side = this_side
                continue

            # GAP vertical
            gap = l_top - cur["bottom"]
            GAP_TOL = median_h * (8.0 if (cur_colors and is_white_bubble(cur_colors[-1])) else 6.0)

            if gap > GAP_TOL:

                if cur_colors:
                    avg = tuple(int(c) for c in np.mean(cur_colors, axis=0))
                    cur["color"] = avg

                cur["text"] = " ".join(x["text"] for x in cur["lines"])
                bubbles.append(cur)

                cur = {
                    "top": l_top, "left": l_left,
                    "bottom": l_bottom, "right": l_left + l.get("width", 0),
                    "lines": [l]
                }

                cur_colors = [new_color] if new_color is not None else []
                cur_color_lab = new_lab
                cur_side = this_side
                continue

            # MISMA BURBUJA (default)
            cur["lines"].append(l)
            cur["bottom"] = max(cur["bottom"], l_bottom)
            cur["right"] = max(cur["right"], l_left + l.get("width", 0))
            if new_color is not None:
                cur_colors.append(new_color)


        # =====================================================
        # CERRAR ÚLTIMA BURBUJA
        # =====================================================
        if cur:
            if cur_colors:
                # Elegir el color más gris (menor diferencia entre R,G,B)
                best = None
                best_score = 999999

                for c in cur_colors:
                    r, g, b = int(c[0]), int(c[1]), int(c[2])
                    score = abs(r - g) + abs(r - b) + abs(g - b)  # qué tan gris es

                    if score < best_score:
                        best_score = score
                        best = c

                if best is not None:
                    cur["color"] = (int(best[0]), int(best[1]), int(best[2]))

            # Cerrar la burbuja
            cur["text"] = " ".join(x["text"] for x in cur["lines"])
            bubbles.append(cur)


        for b in bubbles:
            print("  >> Burbuja detectada:", b["text"])
        print("-"*50)

        # =====================================================
        # LIMPIEZA FINAL
        # =====================================================
        cleaned = []
        for b in bubbles:
            txt = self._clean_text(b["text"]).strip()
            if not txt:
                continue
            if self._is_noise_text(txt):
                continue
            if self._is_blacklist(txt):
                continue
            b["text"] = txt
            cleaned.append(b)
            print("  ** Burbuja limpia aceptada:", b["text"], "(top,left,bottom,right): ", b["top"], b["left"], b["bottom"], b["right"], "color:", b["color"])
        print("-"*50)

        return cleaned

    # -------------------- DETECCIÓN DE MENSAJE RECIBIDO --------------------
    # convierte RGB/BGR a HSV y LAB
    def to_hsv(self, color):
        """Convierte RGB/BGR a HSV (h, s, v)."""
        col = np.array([[tuple(int(c) for c in color)]], dtype=np.uint8)
        hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)[0][0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])

    def to_lab(self, color):
        """Convierte RGB/BGR a LAB (L, A, B)."""
        col = np.array([[tuple(int(c) for c in color)]], dtype=np.uint8)
        lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)[0][0]
        return float(lab[0]), float(lab[1]), float(lab[2])

    # Detecta si una burbuja es de color gris/blanco/beige claro
    def is_grayish(self, color):
        """
        Detecta si una burbuja es gris/blanca, excluyendo
        completamente burbujas verdes (incluso verde pastel).
        """

        r, g, b = [int(c) for c in color]
        h, s, v = self.to_hsv(color)

        # 1) Blanco o gris muy claro
        if v > 200 and s < 40:
            return True

        # 2) Gris medio
        if 120 < v <= 200 and s < 30:
            return True

        # 3) Confirmación por coherencia RGB (tonos grisáceos)
        rgb_diff = max(abs(r - g), abs(r - b), abs(g - b))
        if rgb_diff < 25 and s < 35:
            return True

        # ---------------------------------------------------
        # REGLAS PARA EXCLUIR VERDE WHATSAPP (SIEMPRE)
        # ---------------------------------------------------
        # Hue verde entre 60° y 170°
        if 60 <= h <= 170:
            return False

        # saturación moderada típicamente verde pastel
        if s >= 35:
            return False

        return False


    # -------------------- MÉTODOS PRINCIPALES --------------------
    # Determina si una burbuja es recibida (lado izquierdo)
    def get_received_messages(self, bubbles: List[Dict]) -> List[str]:
        """
        Devuelve la lista de textos de las burbujas clasificadas como recibidas.
        Clasifica cuáles burbujas son mensajes recibidos usando una lógica robusta multimétodo:
        
        PRIORIDAD:
            1. Saturación baja + luminosidad alta → recibido (blanco/gris/beige claro)
            2. Si hay mezcla neutro + color → neutro = recibido
            3. Si todo es neutro → todos recibidos
            4. Si nada es neutro → fallback por luminosidad (LAB L)
        
        Retorna SOLO el texto de las burbujas recibidas.
        """

        if not bubbles:
            return []

        # -----------------------------------------------------------
        # Extraer colores y métricas
        # -----------------------------------------------------------
        colors = []
        gray_flags = []
        L_values = []

        for b in bubbles:
            col = b.get("color") or (255, 255, 255)
            col = tuple(int(min(255, max(0, c))) for c in col)

            colors.append(col)
            gray_flags.append(self.is_grayish(col))

            L, A, B = self.to_lab(col)
            L_values.append(L)

        num_gray = sum(gray_flags)
        total = len(bubbles)

        has_gray = num_gray > 0
        has_colored = num_gray < total

        # -----------------------------------------------------------
        # CASO 1 — Mezcla neutro + color
        # -----------------------------------------------------------
        if has_gray and has_colored:
            return [b["text"] for b, flag in zip(bubbles, gray_flags) if flag]

        # -----------------------------------------------------------
        # CASO 2 — Todo neutro → todo recibido
        # -----------------------------------------------------------
        if has_gray and not has_colored:
            return [b["text"] for b in bubbles]

        # -----------------------------------------------------------
        # CASO 3 — Nada neutro → fallback por luminosidad (LAB)
        # -----------------------------------------------------------
        median_L = float(np.median(L_values))

        received_flags = [(L >= median_L) for L in L_values]

        return [b["text"] for b, flag in zip(bubbles, received_flags) if flag]
    
    # Obtener todas las burbujas detectadas
    def messages(self, ocr_result: Dict, img: np.ndarray) -> List[str]:
        """
        Retorna lista de mensajes filtrados por blacklist.
        Requiere el OCR y la imagen para calcular colores.
        """
        lines = ocr_result.get("lines", []) or []
        if not lines:
            return []

        # Agrupar todo primero (internamente limpia ruido)
        bubbles = self._group_lines_to_bubbles(lines, img)

        return bubbles