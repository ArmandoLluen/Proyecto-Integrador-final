# services/nlp_service.py
import math
import os
import json
import re
from typing import Dict
from git import List
import torch
from dotenv import load_dotenv
from transformers import pipeline
from services.common.common import logger, log_
import unicodedata
import nltk
from nltk.corpus import wordnet as wn
import requests

import re
import nltk
from nltk.stem import SnowballStemmer
# Descargar tokenizer de frases
nltk.download("punkt")
nltk.download("punkt_tab")
import unicodedata
from difflib import SequenceMatcher

import spacy

# Descargar si no existe
try:
    wn.synsets('hola', lang='spa')
except LookupError:
    nltk.download('omw-1.4')

class NLPService:
    def __init__(self, device=None):
        load_dotenv()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        log_("info", logger, f" nlp_service usando dispositivo: {self.device}")
        
        # ---------------- HuggingFace Pipelines ----------------
        self.hf_sent = os.getenv("HF_MODEL_SENT", "pysentimiento/robertuito-sentiment-analysis")
        self.hf_off = os.getenv("HF_MODEL_OFF", "pysentimiento/robertuito-offensive")
        self.hf_hate = os.getenv("HF_MODEL_HATE", "pysentimiento/robertuito-hate-speech")
        self.hf_pipes = {}
        try:
            self.hf_pipes['sent'] = pipeline("text-classification", model=self.hf_sent, device=0 if self.device=='cuda' else -1)
            self.hf_pipes['off'] = pipeline("text-classification", model=self.hf_off, device=0 if self.device=='cuda' else -1)
            self.hf_pipes['hate'] = pipeline("text-classification", model=self.hf_hate, device=0 if self.device=='cuda' else -1)
        except Exception:
            self.hf_pipes = {'sent': None, 'off': None, 'hate': None}
        
        # ---------------- Azure OpenAI ----------------
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        #----------------  Steemer ----------------
        self.stemmer = SnowballStemmer("spanish")
        self.stemmer.stem("insultos")

    def _remove_interspersed_chars(self, s):
        return re.sub(r'(?<=\w)[\.\-\_\*~]+(?=\w)', '', s)
    
    # Quitar acentos
    def _strip_accents(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
    
    def _similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio() >= 0.78
    
    def _detect_repetition(self, txt: str) -> bool:
        words = txt.split()
        if len(words) < 4:
            return False
        return len(set(words[:3])) < 3

    def _contains_imperative(self, txt: str) -> bool:
        imperativos = [
            "ven", "haz", "manda", "obedece", "pásame", "pasame", 
            "envia", "envíame", "muestra", "callate", "cállate"
        ]
        return any(imp in txt for imp in imperativos)


    def _detect_dominance(self, txt: str) -> bool:
        patrones = [
            "yo mando", "yo decido", "hazme caso", "tu haces lo que yo diga",
            "tienes que obedecer", "obedece"
        ]
        return any(p in txt for p in patrones)


    def _detect_sexual_implicature(self, txt: str) -> bool:
        frases = [
            "vamos tu y yo", "vamos tú y yo",
            "los dos solos", "los 2 solos",
            "ven a verme", "quiero verte", "te quiero ver",
            "a solas", "estar solos"
        ]
        return any(p in txt for p in frases)


    def _is_targeted(self, txt: str) -> bool:
        pronombres = ["tu ", "tú ", "te ", "contigo", "a ti", "para ti"]
        return any(p in txt for p in pronombres)


    def _sarcasm(self, txt: str, insultos_final: bool) -> bool:
        return ("jaja" in txt or "xd" in txt) and insultos_final

    def _local_heuristic(self, text: str) -> dict:
        # ---------------- PATRONES DE ACOSO REAL ----------------
        humillacion_list = [
            # Desprecio directo
            "me das pena",
            "das pena",
            "me das verguenza",
            "das verguenza",
            "eres patetico",
            "que patetico",
            "eres ridiculo",
            "que ridiculo",
            "que verguenza contigo",

            # Invalidación personal
            "nadie te quiere",
            "no le importas a nadie",
            "a nadie le importas",
            "todos se burlan de ti",
            "todos se rien de ti",
            "eres un estorbo",
            "no sirves para nada",
            "no sirves",
            "no vales nada",
            "no vales",

            # Menosprecio
            "no eres nadie",
            "no existes como persona",
            "no cuentas para nadie",
            "eres un cero a la izquierda",
            "eres irrelevante",
            "no importas a nadie",
            "no significas nada",
            "eres lo peor",

            # Ridiculización
            "hablas estupideces",
            "dices tonterias",
            "solo dices tonterias",
            "no sabes nada",
            "no tienes idea",
            "tu opinion no importa",
            "lo que dices no importa",
            "para que hablas",
            "para que opinas",

            # Mofa o burla ofensiva
            "eres un chiste",
            "das risa tu",
            "das risa contigo",

            # Silenciamiento humillante
            "callate",
            "mejor callate",
            "no hables",
            "no deberias hablar",
            "no abras la boca",

            # Cuestionamiento personal
            "quien te crees",
            "quien te crees que eres",
            "que te crees",
            "bajate de tu nube",
            "ponte en tu lugar",

            # Ataques emocionales
            "nadie te respeta",
            "nadie te toma en serio",
            "no te respetan",
            "no das la talla",
            "siempre fallas",
            "siempre te equivocas",
            "nunca haces nada bien",

            # Extrema
            "no entiendo como tu madre no te aborto",

            # Comparaciones ofensivas
            "cualquiera es mejor que tu",
            "todos son mejores que tu",
            "eres el peor de todos",
            "siempre quedas mal",
        ]


        intensificadores = [
            # Intensificadores comunes
            "muy",
            "bien",
            "tan",
            "demasiado",
            "bastante",
            "sumamente",
            "totalmente",
            "completamente",
            "extremadamente",
            "super",
            "re",
            "mega",
            "ultra",
            "hiper",

            # Intensificadores coloquiales del español
            "montón",
            "caleta",
            "harto",
            "full",
            "fuerte",

            # Intensificadores enfáticos usados en ataques
            "de verdad",
            "en serio",
            "pero en serio",
            "realmente",
            "literal",

            # Intensificador vulgar
            "puta madre",
        ]

        # ---------------- LISTAS PROCESADAS ----------------
        insultos_list = [
            "idiota","imbecil","estupido","inutil","gilipollas","tonto",
            "tarado","cabron","bastardo","maldito","hijo de puta","boludo","asno",
            "burro","tarada","perra","troll","mamon","gil","pajero","payaso",
            "estafador","caga","pendejo","zorra","pelotudo","baboso","cagon",
            "cagona","mierda","subnormal","anormal","inepto", "aborto", 
            "retardado", "retardada", "retrasado",
            "cretino","cretina","malparido","malparida",
            "parasito","parasita","loco de mierda","loca de mierda",
            "chupapijas","chupapollas","come pijas","come pollas","caga mierda"
        ]

        amenazas_list = [
            "te voy a matar","te mato","te golpeo","te hare dano","te voy a buscar",
            "te voy a encontrar","te voy a romper",
            "te va a costar","te vas a arrepentir","esto no queda asi",
            "me las vas a pagar","no te conviene","te vigilo","te sigo",
            "te estoy mirando","hackeo","hackear","atacar","romper","golpear",
            "lastimar","matar","asesinar","herir","golpearte","destrozar",
            "hackeare","te arrepentiras","me voy a encargar","vas a pagar",
            "te voy a hundir","te hundire","te destruire","te voy a arruinar",
            "te arruinare","te voy a cazar","te cazare","te voy a secuestrar"
        ]

        sexual = [
            r"\bsexo\b", r"\bs3xo\b", r"\bporn\b", r"\bporno\b",
            r"\berotico\b", r"\bvagina\b", r"\bpene\b",
            r"\bmasturbacion\b", r"\borgasmo\b", r"\bfollar\b",
            r"\bcono\b", r"\bcoger\b",
            r"\bsexo oral\b",
            r"\bpenetracion\b", r"\bdesnudo\b", r"\bdesnuda\b",
            r"\bnude\b", r"\btopless\b", r"\bsexy\b",
            r"\bdesvestirse\b", r"\bdesnudarse\b",
            r"\bmamada\b", r"\btetas\b",
            r"\bsenos\b", r"\bpezones\b", r"\bculo\b",
            r"\bpoto\b", r"\bchupar\b",
            r"\bme calientas\b", r"\bme excitas\b",
            r"\bfoto sexy\b", r"\bfoto hot\b",
            r"\bmandame fotos\b", r"\bnudes\b",
            r"\bmamacita\b",
            r"\bviolacion\b", r"\bviolar\b", r"\bme la pones dura\b"
            r"\bme la pones dura\b", r"\bpack\b"
            r"\bfollar\b", r"\bcoger\b"
            r"\bpussy\b", r"\bblowjob\b", r"\bdick\b", r"\bcock\b"
            r"\bslut\b", r"\bchimbo\b",
            r"\bviolar\b"
        ]

        soeces = [
            "mierda","puta","gil","cabron","pendejo","idiota","imbecil","boludo",
            "estupido","fuck","shit","bitch","asshole","concha de su madre",
            "conchadesumadre","hijo de puta","hijodeputa","reconchadetumadre",
            "puta madre","putamadre","malparido","malparida","mlp","hdp",
            "pndjo","pndja","tarado","asno","burro","mnm","zorra",
            "pelotudo","baboso","cagon","cagona","hueva","huevon","huevona",
            "subnormal","anormal","inepto","estafa","estafador","estafadora"
            "wtf", "fuck", "sucks", "sux", "damn", "crap"
        ]

        slurs = [
            # raza / etnia
            r'\bnegro\b', r'\bnegrata\b', r'\bmono\b', r'\bsimio\b', r'\bindio\b',
            r'\bserrano\b', r'\bcholo\b', r'\bchola\b', r'\bamarillo\b',
            r'\bchinito\b', r'\bgitano\b', r'\bromani\b',

            # religión
            r'\bjudio\b', r'\bjudío\b', r'\bmusulman\b', r'\bislamico\b',
            r'\bhereje\b',

            # orientación sexual / género
            r'\bmaricon\b', r'\bmarica\b', r'\bputo\b', r'\bgay\b',
            r'\blesbiana\b', r'\btrans\b', r'\btransexual\b',

            # discapacidad
            r'\bcojo\b', r'\bmongolico\b', r'\bdown\b',

            # apariencia
            r'\bgordo\b', r'\bgorda\b', r'\bobeso\b', r'\bfeo\b', r'\bfea\b',

            # nacionalidades despectivas
            r'\bveneco\b', r'\bbolita\b', r'\bchilote\b',
        ]

        acoso_sexual_frases = [
            r"se te ve.*(bonita|rica|hermosa).*escote",
            r"\blindo escote\b",
            r"quiero verte con esa blusa",
            r"quiero verte sin ropa",
            r"se marca.*(pecho|tetas|nalgas|trasero)",
            r"estas muy rica",
            r"me encantas desnuda",
            r"enviame fotos desnuda",
            r"enviame fotos sexy",
            r"me gustaria verte desnuda",
            r"a ver esa delantera",
            r"ponte esa ropa de nuevo"
            r"violacion"
        ]

        coercion_list = [
            "hazlo o","si no haces","obligarte","tienes que hacerme caso",
            "no tienes opción","debes obedecer"
        ]

        pasivo_agresivo = [
            r"no es insulto pero",
            r"con cariño pero",
            r"solo te digo la verdad",
            r"no te ofendas pero",
            r"no quiero ser malo pero",
        ]

        # ---------------- NORMALIZACIÓN AVANZADA ----------------
        t_orig = text.strip()
        t = t_orig.lower()
        t_noacc = self._strip_accents(t)

        # Quitar símbolos
        t_nosym = re.sub(r'[^a-zA-Z0-9áéíóúñ ]', '', t)
        t_nosym_noacc = self._strip_accents(t_nosym)

        # Quitar elongaciones: estuuuupido -> estupido
        t = re.sub(r'(.)\1{2,}', r'\1', t)
        t_noacc = re.sub(r'(.)\1{2,}', r'\1', t_noacc)

        # Compactar palabras separadas por espacios
        t_compact = re.sub(r'(\w)\s+(\w)', r'\1\2', t_noacc)

        # Tokens
        tokens = re.findall(r'\w+', t)
        tokens_noacc = re.findall(r'\w+', t_noacc)
        tokens_nosym = re.findall(r'\w+', t_nosym_noacc)
        tokens_compact = re.findall(r'\w+', t_compact)

        # Deobfuscación
        t_deobf = self._remove_interspersed_chars(t_noacc)
        tokens_deobf = re.findall(r'\w+', t_deobf)

        # Leetspeak
        leet_map = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
        t_leet = t_noacc.translate(leet_map)
        tokens_leet = re.findall(r'\w+', t_leet)

        all_token_sets = (
            tokens + tokens_noacc + tokens_nosym +
            tokens_leet + tokens_compact + tokens_deobf
        )

        # ---------------- DETECCIONES ORIGINALES ----------------
        intensificador_final = any(w in t for w in intensificadores)

        insultos_detect = (
            any(w in t for w in insultos_list) or
            any(w in t_noacc for w in insultos_list) or
            any(w in t_nosym_noacc for w in insultos_list) or
            any(w in t_leet for w in insultos_list)
        )

        insultos_stem = any(
            self.stemmer.stem(tok) in [self.stemmer.stem(i) for i in insultos_list]
            for tok in all_token_sets
        )

        insultos_fuzzy = any(
            self._similar(tok, ins)
            for tok in all_token_sets
            for ins in insultos_list
        )

        insultos_final = insultos_detect or insultos_stem or insultos_fuzzy

        sexual_final = any(expr == tok for expr in sexual for tok in tokens_noacc)
        soez_final = any(w == tok for w in soeces for tok in tokens_noacc)
        amenazas_final = any(frase == tok for frase in amenazas_list for tok in tokens_noacc)
        discriminacion_final = any(re.search(pattern, t_noacc) for pattern in slurs)
        acoso_sexual_final = any(re.search(pattern, t_noacc) for pattern in acoso_sexual_frases)

        humillacion_final = any(frase in t_noacc for frase in humillacion_list)
        coercion_final = any(frase in t_noacc for frase in coercion_list)
        acoso_pasivo = any(re.search(pat, t_noacc) for pat in pasivo_agresivo)

        # ---------------- NUEVAS DETECCIONES NLP ----------------
        insistencia = self._detect_repetition(t_noacc)
        orden_autoritaria = self._contains_imperative(t_noacc)
        dominancia = self._detect_dominance(t_noacc)
        sexual_implicito = self._detect_sexual_implicature(t_noacc)
        tiene_objetivo = self._is_targeted(t_noacc)
        sarcasmo = self._sarcasm(t_noacc, insultos_final)

        # Intensidad emocional
        intensidad = 0
        if "!!" in t_orig or "??" in t_orig:
            intensidad += 1
        if re.search(r"(.)\1{2,}", t_orig):
            intensidad += 1

        # ---------------- SCORE TOTAL (versión científica) ----------------
        score = 0

        # Pesos basados en severidad de la tabla
        if insultos_final: score += 4
        if humillacion_final: score += 5
        if soez_final: score += 2
        if sexual_final: score += 3
        if sexual_implicito: score += 3
        if acoso_sexual_final: score += 7
        if coercion_final: score += 7
        if amenazas_final: score += 7
        if discriminacion_final: score += 6

        # Conductas asociadas
        if orden_autoritaria: score += 3
        if insistencia: score += 2
        if dominancia: score += 2
        if sarcasmo: score += 1

        # Intensidad emocional (caps, repetición, exclamaciones)
        score += intensidad * 1.5

        # Ajuste si NO hay objetivo humano (reduce falsos positivos)
        if score >= 7 and not tiene_objetivo and not sexual_final:
            score -= 3

        # ---------------- HOSTILIDAD Y ACOSO ----------------
        hostilidad = score >= 6
        es_acoso = score >= 9

        # ---------------- RAZÓN ----------------
        razones = []
        if insultos_final: razones.append("insultos")
        if amenazas_final: razones.append("amenazas")
        if discriminacion_final: razones.append("discriminación")
        if sexual_final or acoso_sexual_final: razones.append("contenido sexual")
        if sexual_implicito: razones.append("sexualización implícita")
        if soez_final: razones.append("lenguaje soez")
        if humillacion_final: razones.append("humillación")
        if coercion_final: razones.append("coerción")
        if insistencia: razones.append("insistencia")
        if orden_autoritaria: razones.append("orden autoritaria")
        if dominancia: razones.append("dominancia")
        if sarcasmo: razones.append("sarcasmo")
        if intensidad > 0: razones.append("intensidad emocional")

        razon = "Se detectó " + ", ".join(razones) if es_acoso else "No se detectó acoso"

        return {
            "hostilidad": "sí" if hostilidad else "no",
            "insultos": "sí" if insultos_final else "no",
            "discriminacion": "sí" if discriminacion_final else "no",
            "lenguaje_soez": "sí" if soez_final else "no",
            "contenido_sexual": "sí" if sexual_final or acoso_sexual_final else "no",
            "amenazas": "sí" if amenazas_final else "no",
            "humillacion": "sí" if humillacion_final else "no",
            "coercion": "sí" if coercion_final else "no",
            "es_acoso": "sí" if es_acoso else "no",
            "razon": razon,
            "score": score
        }
    
    def _normalize_unicode(self, text):
        # Quita acentos y pasa a minúsculas
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode().lower()
        

    # ---------------- Hugging Face / modelo fineturning ----------------
    def _hf(self, text):
        """
        Combina resultados de HF y heurística:
        - Si alguno detecta algo negativo (offense/hate), prevalece el "sí".
        - Si HF falla en un tag, usamos heurística.
        - El hostilidad final siempre refleja la peor señal detectada:
            hostil > neutral > positivo
        """
        
        sent, off, hate = None, None, None
        try:
            if self.hf_pipes.get('sent'):
                out = self.hf_pipes['sent'](text)[0]
                sent = self._map_sent_label(out['label'])  # 'hostil' / 'neutral' / 'positivo'
        except:
            pass

        try:
            if self.hf_pipes.get('off'):
                out = self.hf_pipes['off'](text)[0]
                off = self._map_bool_label(out['label'])   # 'sí' / 'no'
        except:
            pass

        try:
            if self.hf_pipes.get('hate'):
                out = self.hf_pipes['hate'](text)[0]
                hate = self._map_bool_label(out['label'])  # 'sí' / 'no'
        except:
            pass

        return sent, off, hate

    # ---------------- Mapear etiquetas ----------------
    def _map_sent_label(self, label: str):
        lab = label.lower()
        if any(k in lab for k in ['neg', 'negative', 'negativo', 'bad','hostil','malo']): return 'hostil'
        if any(k in lab for k in ['pos','positivo','positive', 'positivo', 'good', 'bueno']): return 'positivo'
        return 'neutral'

    def _map_bool_label(self, label: str):
        lab = label.lower()
        return lab in ['yes','true','toxic','offensive','insult','abuse','hate']
    
    def _sanitize_azure_value(self, key, value):
        """
        Normaliza cualquier salida del LLM:
        - Categorías booleanas → True/False
        - Probabilidad → float entre 0.0 y 1.0
        - Cualquier valor roto → False / 0.0
        """

        # -------- PROBABILIDAD --------
        if key == "probabilidad":
            # Número
            if isinstance(value, (int, float)):
                v = float(value)
                if v > 1:
                    v = v / 10 if v <= 10 else 1
                return max(0.0, min(1.0, v))

            # String con número
            if isinstance(value, str):
                try:
                    v = float(value.replace(",", "."))
                    if v > 1:
                        v = v / 10 if v <= 10 else 1
                    return max(0.0, min(1.0, v))
                except:
                    return 0.0

            # Boolean
            if isinstance(value, bool):
                return 1.0 if value else 0.3

            return 0.3

        # -------- BOOLEANOS (insultos, amenazas, humillación…) --------
        if isinstance(value, bool):
            return value

        # Strings tipo "true"/"si"/"yes"
        if isinstance(value, str):
            if value.lower().strip() in ["true", "sí", "si", "yes"]:
                return True
            return False

        # Si el modelo devolvió un número → interpretarlo como bool
        if isinstance(value, (int, float)):
            return value > 0

        # Valor desconocido
        return False
    
    def extract_json(self, text):
        # captura el primer {...} válido
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        if not matches:
            return None
        for m in matches:
            try:
                return json.loads(m)
            except:
                continue
        return None

    def normalize_sexual_abuse_terms(self, text: str) -> str:
        """
        Detecta expresiones de abuso sexual evitando falsos positivos.
        Tokeniza con spaCy (sin depender de punkt/punkt_tab).
        """

        # Crear tokenizer español SIN MODELO
        nlp_es = spacy.blank("es")

        # --- Tokenización con spaCy (estable, sin descargas) ---
        doc = nlp_es(text.lower())
        tokens = [t.text for t in doc]

        stemmer = SnowballStemmer("spanish")
        stems = [stemmer.stem(t) for t in tokens]

        # --- Palabras que sí implican abuso sexual ---
        sexual_stems = {
            "viol", "abus", "manos", "toc", "forz", "acos"
        }

        # --- Palabras clave de contexto sexual ---
        sexual_context = {
            "sexo", "sexual", "relacion", "relaciones",
            "cuerpo", "intimo", "íntimo", "forzado"
        }

        # --- Exclusiones explícitas (falsos positivos) ---
        blacklist_exact = {
            "violencia", "violento", "violín", "violin",
            "biologia", "biológico", "biologica",
            "violentando", "violenta", "violente",
            "violador"   # insulto, pero no abuso sexual automáticamente
        }

        def has_sexual_context(idx):
            """Mira 2 palabras antes/después."""
            for offset in [-2, -1, 1, 2]:
                j = idx + offset
                if 0 <= j < len(tokens):
                    if tokens[j] in sexual_context:
                        return True
                    if stemmer.stem(tokens[j]) in sexual_context:
                        return True
            return False

        out_tokens = list(tokens)

        for i, (tok, st) in enumerate(zip(tokens, stems)):

            # 1. Excluir falsos positivos explícitos
            if tok in blacklist_exact:
                continue

            # 2. Detectar raíces peligrosas
            if st in sexual_stems:

                # Evitar palabras inocuas: "violencia", "violin", etc.
                if tok.startswith("violen") or tok in blacklist_exact:
                    continue

                # 3. Reemplazar si hay contexto sexual claro
                if has_sexual_context(i):
                    out_tokens[i] = "(palabra que contiene contenido sobre abuso sexual)"
                    continue

                # 4. Verbos de violar → abuso sexual directo
                if st == "viol":
                    if tok not in ["violencia", "violin", "violento"]:
                        out_tokens[i] = "(palabra que contiene contenido sobre abuso sexual)"
                    continue

                # 5. Abusar + "de", "a"...
                if st == "abus":
                    if i+1 < len(tokens) and tokens[i+1] in ["de", "a", "del", "la"]:
                        out_tokens[i] = "(palabra que contiene contenido sobre abuso sexual)"
                    continue

                # 6. Tocó / tocar → sexual si menciona cuerpo
                if st == "toc":
                    nearby = tokens[max(0, i-2): i+3]
                    cuerpo = {"ella", "el", "cuerpo", "pecho", "trasero", "parte", "pantalón"}
                    if any(w in cuerpo for w in nearby):
                        out_tokens[i] = "(palabra que contiene contenido sobre abuso sexual)"
                    continue

        return " ".join(out_tokens)

    # ---------------- Azure OpenAI ----------------
    def _ask_azure_openai(self, text):
        if not (self.azure_endpoint and self.azure_key and self.azure_deployment): 
            return None

        url = self.azure_endpoint.rstrip('/') + f"/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
        headers = {"api-key": self.azure_key, "Content-Type": "application/json"}

        criterios = {
            "insultos": "insultos directos, indirectos o disfrazados, vulgaridades dirigidas a la persona, ataques a la inteligencia, capacidad o valor personal.",
            "discriminacion": "ataques a raza, género, orientación, apariencia, cuerpo, clase social, discapacidad, nacionalidad.",
            "lenguaje_soez": "groserías, vulgaridades, lenguaje sexual explícito o sugerente.",
            "contenido_sexual": "insinuaciones sexuales hacia la persona, solicitudes de fotos íntimas, referencias al cuerpo sexual, violación o frases sexualizadas.",
            "amenazas": "amenazas físicas, digitales, sociales, sexuales o emocionales, frases intimidantes o que implican daño, persecucion, vigilancia o hackeo.",
            "humillacion": "bullying, burla, ridiculización, desprecio, cosificación, sexualización hacia la persona, bromas hirientes o de superioridad.",
            "coercion": "presión de todo tipo, insistencia, manipulación emocional, chantaje o frases que obligan a la persona.",
            "hostilidad" : "si hay indicios de humillacion, maltrato, violencia, coercion, intimidación fisica, emocional, digital o sexual", 
            "es_acoso": "si el mensaje contiene indicios de: acoso emocional, fisico, cibernetico o sexual, coercion, amenazas o maltrato.",
            "probabilidad": "indique la probabilidad de que el mensaje sea acoso o no."
        }

        resultados = {}

        # ---- Evaluación individual de criterios ----
        for key, desc in criterios.items():

            if key == "probabilidad":
                json_format = '{"probabilidad": NUMBER}'
            else:
                json_format = f'{{"{key}": true/false}}'

            prompt = (
                "Responde únicamente en JSON. No agregues texto antes ni después.\n"
                f"Formato estricto: {json_format}\n\n"
                f"Evalúa el mensaje según el criterio: {key}.\n"
                f"descripcion del criterio: {desc}\n\n"
                f"Mensaje a analizar: \"{text}\""
            )

            body = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200
            }

            try:
                r = requests.post(url, headers=headers, json=body)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]

                data = self.extract_json(content)
                if data is None:
                    print("JSON invalido")
                    raise Exception("JSON inválido")

                valor = data.get(key)
                resultados[key] = self._sanitize_azure_value(key, valor)

            except Exception:
                print("no se genero", key)

                if key == "probabilidad":
                    resultados[key]= 1 if resultados["es_acoso"] else 0.3
                else: 
                    resultados[key] = False

        # ---- Generar razonamiento global ----
        prompt_razon = (
            "Responde únicamente en JSON. No agregues texto fuera del JSON.\n"
            "Formato exacto: {\"razon\": \"texto\"}\n\n"
            "Da una razon breve (1 o 2 líneas) basada en los resultados parciales.\n"
            f"Mensaje: \"{text}\"\n"
            f"Resultados: {json.dumps(resultados)}"
        )

        body = {
            "messages": [{"role": "user", "content": prompt_razon}],
            "temperature": 0.7,
            "max_tokens": 150
        }

        try:
            r = requests.post(url, headers=headers, json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

            data = self.extract_json(content)
            if data:
                resultados["razon"] = data.get("razon", "")
            else:
                raise Exception()

        except Exception:
            print("no se genero la razon")
            resultados["razon"] = ""

        if resultados["es_acoso"]:
            if resultados["probabilidad"] <= 0.5:
                resultados["probabilidad"] = 1.0
        return resultados

    # ---------------- Analizar contexto de conversación ----------------
    def analyze_conversation_context(self, messages: List[str]) -> Dict:
        """
        Analiza la conversación completa y devuelve un resumen global,
        incluyendo opinión/reflexión sobre posibles ciberacoso o bullying.
        """
        print("Analizando contexto de conversación")

        # ---------------- Caso vacío ----------------
        if not messages:
            return self._resultado_vacio()

        # ---------------- Juntar todo el texto ----------------
        full_text = " \n".join(messages)

        # ---------------- 1. Analizar conversación completa ----------------
        full_text_safe = self.normalize_sexual_abuse_terms(full_text) 
        analysis = self.analyze_text(full_text_safe, True)

        # ---------------- 2. Preparar prompt de reflexión ----------------
        prompt = (
            "Eres un experto en ciberacoso, dinámica social y análisis conversacional entre personas."
            f"Analiza el SIGUIENTE BLOQUE DE CONVERSACIÓN (todo el chat concatenado y censurado) \n{full_text_safe}\n "
            f"se tiene el siguiente análisis previo: {json.dumps(analysis)}.\n\n"
            "Dependiendo del análisis previo, di si hay acoso o no, y justifica tu respuesta."
            "La evaluacion debe ser robusta pero concisa, de preferencia 1 o dos parrafos maximo."
            "NO resumas el contenido de los mensajes: analiza el comportamiento y las dinámicas sociales presentes.\n\n"
        )

        # ---------------- 3. Llamada a Azure OpenAI ----------------
        reflexion = "No se pudo generar reflexión."

        try:
            if self.azure_endpoint and self.azure_key and self.azure_deployment:
                url = self.azure_endpoint.rstrip('/') + f"/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"

                headers = {
                    "api-key": self.azure_key,
                    "Content-Type": "application/json"
                }

                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 1000
                }

                r = requests.post(url, headers=headers, json=body)
                r.raise_for_status()
                content = r.json()['choices'][0]['message']['content']
                reflexion = content.strip()

        except Exception as e:
            reflexion = f"No se pudo generar reflexión: {e}"

        # ---------------- 4. Formato final coherente ----------------
        return {
                "hostilidad": analysis.get("hostilidad"),
                "insultos": analysis.get("insultos"),
                "discriminacion": analysis.get("discriminacion"),
                "lenguaje_soez": analysis.get("lenguaje_soez"),
                "contenido_sexual": analysis.get("contenido_sexual"),
                "amenazas": analysis.get("amenazas"),
                "humillacion": analysis.get("humillacion"),
                "coercion": analysis.get("coercion"),
                "es_acoso": analysis.get("es_acoso"),
                "razon": analysis.get("razon"),
                "probabilidad": analysis.get("probabilidad"),
                "reflexion": reflexion
            }

    # ---------------- Resultado vacío ----------------
    def _resultado_vacio(self):
        return {
                "hostilidad": "neutral",
                "insultos": False,
                "discriminacion": False,
                "lenguaje_soez": False,
                "contenido_sexual": False,
                "amenazas": False,
                "humillacion": False,
                "coercion": False,
                "es_acoso": False,
                "razon": "No hay mensajes para analizar.",
                "probabilidad": 0.0,
                "reflexion": "No se encontraron interacciones que permitan evaluar acoso o riesgos."
            }

    
    # ---------------- normalizar “sí/no/hostil/positivo/etc” a boolean ----------------
    def _to_bool(self, v):
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        v = str(v).lower().strip()
        if v in ["sí", "si", "yes", "true", "hostil", "ofensivo", "hate"]:
            return True
        return False
    
    # ---------------- Convertir score a probabilidad ----------------
    def _logistic_probability(self, score, midpoint=7, scale=1.2):
        """
        Convierte el score heurístico (0–30 aprox) en una probabilidad 0–1.
        midpoint = punto donde P=0.5
        scale = qué tan rápido sube la curva
        """
        return 1 / (1 + math.exp(-(score - midpoint) / scale))

    # ---------------- Analizar texto ----------------
    def analyze_text(self, text: str, texto_seguro: bool = False):
        # ---------------- Normalización ----------------
        text_clean = text.strip()
        text_norm = self._normalize_unicode(text_clean)

        trivial_msgs = ['hola','buenos dias','buen dia','ok','si','no','gracias','jaja','jajaja','xd']
        if not text_clean or text_norm in trivial_msgs or len(text_norm) < 4:
            return self._resultado_vacio()

        # ---------------- Llamadas externas ----------------
        hf_sent, hf_off, hf_hate = None, None, None
        try:
            hf_sent, hf_off, hf_hate = self._hf(text_clean)
        except:
            pass

        local = self._local_heuristic(text_clean)

        if not texto_seguro:
                texto_seguro = self.normalize_sexual_abuse_terms(text)
        else:
            texto_seguro = text

        try:
            azure = self._ask_azure_openai(texto_seguro) if (self.azure_endpoint and self.azure_key and self.azure_deployment) else {}
        except Exception as e:
            print("cayo en la excepcion", e.__context__)
            azure = {}

        print("respuestas azure:\n", text_clean, " -> ", azure)
        print("-"*50)

        # ---------------- Helper ----------------
        def is_true(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ["true", "sí", "si", "hostil", "ofensivo", "True", "Sí", "Si", "TRUE"]
            return False

        # ---------------- Señales unitarias (Azure > Local > HF) ----------------
        hostilidad       = is_true(azure.get("hostilidad"))       or is_true(local.get("hostilidad"))       or (hf_sent == "hostil")
        insultos         = is_true(azure.get("insultos"))         or is_true(local.get("insultos"))         or is_true(hf_off)
        discriminacion   = is_true(azure.get("discriminacion"))   or is_true(local.get("discriminacion"))   or is_true(hf_hate)
        lenguaje_soez    = is_true(azure.get("lenguaje_soez"))    or is_true(local.get("lenguaje_soez"))
        contenido_sexual = is_true(azure.get("contenido_sexual")) or is_true(local.get("contenido_sexual"))
        amenazas         = is_true(azure.get("amenazas"))         or is_true(local.get("amenazas"))
        humillacion      = is_true(azure.get("humillacion"))      or is_true(local.get("humillacion"))
        coercion         = is_true(azure.get("coercion"))         or is_true(local.get("coercion"))

        score_local      = local.get("score", 0)
        razon            = azure.get("razon", "")

        # ---------------- probabilidad base ----------------
        p_base = (
            contenido_sexual * 0.20 +
            amenazas         * 0.25 +
            coercion         * 0.22 +
            humillacion      * 0.15 +
            discriminacion   * 0.15 +
            insultos         * 0.10 +
            hostilidad       * 0.08 +
            lenguaje_soez    * 0.05
        )

        p_local = self._logistic_probability(score_local)

        # ---------------- probabilidad combinada de las razones ----------------
        p_fusion = p_base * 0.7 + p_local * 0.3

        # ---------------- probabilidad combinada con el modelo LLM ----------------
        p_azure = None
        try:
            val = azure.get("probabilidad", None)

            if isinstance(val, (float, int)):
                p_azure = float(val)

            elif isinstance(val, str):
                # Acepta "0.87", "0,87", "true", "false"
                v = val.strip().lower()
                if v in ["true", "sí", "si"]:
                    p_azure = 1.0
                elif v in ["false", "no"]:
                    p_azure = 0.0
                else:
                    p_azure = float(v.replace(",", "."))  # números en texto

            elif isinstance(val, bool):
                p_azure = 1.0 if val else 0.0

        except:
            p_azure = None

        # Si Azure NO provee probabilidad → ignorarlo
        if p_azure is not None:
            prob_acoso_final = (p_azure * 0.65) + (p_fusion * 0.35)
        else:
            prob_acoso_final = p_fusion

        # Decisión final
        es_acoso = prob_acoso_final >= 0.5
        # ---------------- Motivo ----------------
        razones = []
        if hostilidad: razones.append("hostilidad")
        if insultos: razones.append("insultos")
        if discriminacion: razones.append("discriminación")
        if lenguaje_soez: razones.append("lenguaje soez")
        if contenido_sexual: razones.append("contenido sexual")
        if amenazas: razones.append("amenazas")
        if humillacion: razones.append("humillación")
        if coercion: razones.append("coerción")

        if razones:
            if azure.get("razon"):
                razon = f"Se detecta: {', '.join(razones)}.\n {azure.get('razon', '')}"
            else:
                razon = f"Se detecta: {', '.join(razones)}"
        else:
            razon = f"No se detecta acoso."
        respuesta = {
            "texto": text_clean,
            "hostilidad": "sí" if hostilidad else "no",
            "insultos": "sí" if insultos else "no",
            "discriminacion": "sí" if discriminacion else "no",
            "lenguaje_soez": "sí" if lenguaje_soez else "no",
            "contenido_sexual": "sí" if contenido_sexual else "no",
            "amenazas": "sí" if amenazas else "no",
            "humillacion": "sí" if humillacion else "no",
            "coercion": "sí" if coercion else "no",
            "es_acoso": "sí" if es_acoso else "no",
            "razon": razon,
            "probabilidad": prob_acoso_final
        }

        print("Análisis final:", respuesta)
        print("-"*50)

        return respuesta

