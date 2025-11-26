# services/ocr_service.py
import os
import time
import io
import requests
from PIL import Image
from dotenv import load_dotenv
import numpy as np
load_dotenv()

class OCRService:
    """
    Extrae líneas con bounding boxes desde una imagen.
    Usa Azure Read (Computer Vision) si está configurado; fallback a pytesseract.
    Resultado: {"width": int, "height": int, "lines": [{text,left,top,width,height,center_x,center_y}]}
    """
    def __init__(self):
        self.endpoint = os.getenv("AZURE_CV_ENDPOINT")   # e.g. https://<resource>.cognitiveservices.azure.com
        self.key = os.getenv("AZURE_CV_KEY")

    def image_to_lines(self, image_bytes: bytes):
        if not self.endpoint or not self.key:
            print(f'"AZURE_CV_ENDPOINT"{self.endpoint}, "AZURE_CV_KEY"{self.key}')
            raise RuntimeError("Faltan claves Azure CV.")
        
        try:
            return self._azure_read(image_bytes)
        except Exception as e:
            print(f"OCR Azure falló: {e}")
            raise RuntimeError(f"OCR Azure falló: {e}")

    def _azure_read(self, image_bytes: bytes, language: str = "es"):
        read_url = self.endpoint.rstrip("/") + "/vision/v3.2/read/analyze"
        headers = {"Ocp-Apim-Subscription-Key": self.key, "Content-Type": "application/octet-stream"}
        r = requests.post(read_url, headers=headers, data=image_bytes)
        r.raise_for_status()
        operation_url = r.headers.get("operation-location")
        if not operation_url:
            raise RuntimeError("Azure Read: no operation-location returned")
        # Poll
        for _ in range(60):
            rr = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": self.key})
            rr.raise_for_status()
            j = rr.json()
            status = j.get("status", "")
            if status.lower() in ("succeeded", "failed"):
                break
            time.sleep(0.5)
        if j.get("status", "").lower() != "succeeded":
            raise RuntimeError("Azure Read failed or timed out")
        pages = j.get("analyzeResult", {}).get("readResults", [])
        if not pages:
            return {"width": 0, "height": 0, "lines": []}
        p = pages[0]
        width = p.get("width", 0)
        height = p.get("height", 0)
        lines = []
        for line in p.get("lines", []):
            text = line.get("text", "").strip()
            bbox = line.get("boundingBox", [])
            if not text or len(bbox) < 6:
                continue
            xs = bbox[0::2]; ys = bbox[1::2]
            left = int(min(xs)); top = int(min(ys))
            right = int(max(xs)); bottom = int(max(ys))
            w = right - left; h = bottom - top
            cx = left + w/2; cy = top + h/2
            lines.append({"text": text, "left": left, "top": top, "width": w, "height": h, "center_x": cx, "center_y": cy})
        return {"width": width, "height": height, "lines": lines}
