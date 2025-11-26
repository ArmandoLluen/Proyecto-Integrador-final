# Chat Analyzer — Detector de Acoso en Capturas de Chat

**Chat Analyzer** es una aplicación web interactiva desarrollada para analizar capturas de chat y detectar posibles casos de **acoso, ciberbullying, insultos o lenguaje inapropiado**.

## 📌 Descripción General
Este proyecto implementa una solución basada en Inteligencia Artificial para **automatizar el análisis de capturas de pantalla de conversaciones**, con el objetivo de identificar **indicadores tempranos de acoso digital**. 

El sistema procesa imágenes, extrae texto con OCR, organiza los mensajes en burbujas, aplica modelos de NLP y genera un reporte estructurado, reduciendo significativamente el tiempo de evaluación y apoyando la toma de decisiones de equipos especializados.

---

## 🎯 Objetivo General
Desarrollar una herramienta que permita **agilizar el análisis preliminar de conversaciones** para la detección de posible acoso digital, reduciendo la carga manual y facilitando la priorización de casos.

### 📌 Objetivos Específicos
- Automatizar la extracción de texto desde imágenes mediante OCR.  
- Organizar las conversaciones en mensajes identificables y estructurados.  
- Detectar señales de hostilidad, acoso, coerción, amenazas o contenido inapropiado mediante NLP.  
- Generar reportes interpretativos y exportar resultados a Excel.  
- Incrementar la capacidad operativa del análisis de casos.

---

## 🧪 Hipótesis del Proyecto
> *Si se automatiza la lectura, organización y análisis preliminar de capturas de conversación mediante IA, el tiempo de evaluación disminuirá sustancialmente sin comprometer la calidad, permitiendo detectar de manera más eficiente posibles casos de acoso digital.*

Los resultados obtenidos validaron completamente esta hipótesis.

---

## 🏗️ Arquitectura del Sistema

1. **Interfaz Web (Streamlit)**  
   Permite subir imágenes, ejecutar el análisis y visualizar resultados.

2. **Backend**
   Coordina el flujo completo: OCR → procesamiento → NLP → exportación.

3. **Módulo OCR (Azure Computer Vision + fallback AI Foundry)**
   Extrae texto incluso en imágenes complejas o con baja resolución.

4. **Módulo de Detección de Conversaciones**
   Agrupa texto en burbujas de chat e identifica quién envía cada mensaje.

5. **Módulo NLP (Hugging Face + reglas lingüísticas)**
   Detecta categorías como:
   - amenazas explícitas  
   - insultos  
   - discriminación  
   - coerción  
   - contenido sexual explícito/implícito  
   - hostilidad emocional  
   - lenguaje agresivo o manipulativo  

6. **Módulo Excel**
   Genera un archivo estructurado con todos los mensajes analizados y su clasificación.

---

## 📊 Métricas y Resultados

### ⏱️ Impacto Operativo
| Caso | Antes (Manual) | Después (IA) | Mejora |
|------|----------------|--------------|---------|
| Caso estándar (20 imágenes) | 4 horas | 30 minutos | **–87.5%** |
| Caso complejo (60 imágenes) | 12 horas | 1.5 horas | **–87.5%** |

Capacidad operativa semanal en jornada de 40 h:
- **Manual:** 10 casos → IA: 80 casos (8× más)
- **Manual:** 3.3 casos complejos → IA: 26.6 casos complejos

### 🤖 Rendimiento del Modelo
**A nivel de conversación**
- Exactitud: 95%
- Precisión: 90%
- Recall: **100%**
- F1-score: 0.95

**A nivel de mensaje individual**
- Exactitud: 91%
- Precisión: 97%
- Recall: 80%
- F1-score: 0.88

---

## 📚 Criterios de Detección (basados en evidencia científica)
El modelo utiliza un conjunto de categorías y scores fundamentados en trabajos como:

- **Van Hee et al. (2018)** – ciberacoso, humillación, agresión relacional  
- **Waseem & Hovy (2016)** – diferenciación entre insultos comunes y discriminatorios  
- **Williams et al. (2013)** – grooming, coerción, contenido sexual no consensuado  
- **Hosseinmardi et al. (2015)** – hostilidad, agresividad lingüística  

---

## 🚀 Recomendaciones de Implementación
- Anonimizar manualmente nombres, números e información sensible previo al análisis.  
- Ejecutar un **piloto supervisado de 4–6 semanas** con casos reales.  
- Recalcular métricas con datos operativos del piloto.  
- Integrar la solución al flujo institucional:  
  *recepción → anonimización → análisis IA → revisión humana → acompañamiento.*  
- Establecer un **protocolo de revisión humana obligatoria**.  
- Capacitar al equipo en uso responsable de IA.

---

## 🧩 Conclusiones
- El proyecto confirma que la IA puede **reducir de forma drástica** los tiempos de análisis, pasando de horas a minutos.  
- Los objetivos planteados fueron alcanzados con éxito, demostrando un modelo eficiente, escalable y alineado a la hipótesis inicial.  
- La herramienta no reemplaza al profesional, sino que **optimiza el análisis preliminar**, permitiendo una atención más rápida y oportuna.  
- La solución es técnicamente robusta, fundamentada teóricamente y lista para un piloto supervisado y futura integración institucional.

---

## **Características principales**

- **Subida de imágenes de chat** en formatos PNG, JPG o JPEG.
- **Detección de texto mediante OCR** usando Azure Read.
- **Agrupación de líneas en burbujas de chat** y filtrado de mensajes válidos.
- **Detección de mensajes recibidos** y análisis individual.
- **Análisis de sentimiento, insultos, discriminación y lenguaje soez** por mensaje.
- **Análisis de contexto general** de la conversación incluyendo reflexión sobre posible acoso.
- **Almacenamiento temporal en buffer** para analizar varias imágenes antes de exportar.
- **Exportación a Excel** de resultados individuales y resumen global.

---

## **Flujo de trabajo**

```mermaid
flowchart TD
    A[Usuario sube imagen] --> B[OCRService: extrae texto y bounding boxes]
    B --> C[DetectionService: agrupa líneas en burbujas de chat]
    C --> D[Filtrado de burbujas válidas]
    D --> E[Extracción de mensajes recibidos (lado izquierdo)]
    E --> F[NLPService: analiza cada mensaje]
    F --> G[Resultado por mensaje: sentimiento, insultos, discriminación, lenguaje soez, acoso, razón]
    D --> H[NLPService: analiza contexto general de conversación]
    H --> I[Resultado global de conversación: sentimiento, acoso, razón + reflexión]
    G --> J[Almacenamiento en buffer (session_state)]
    I --> J
    J --> K[ExcelService: exporta resultados a Excel]
    K --> L[Usuario descarga archivo]
```

## **Servicios internos**

- **OCRService**
Extrae texto de la imagen con bounding boxes.
Usa Azure Computer Vision Read API.

- **DetectionService**
Agrupa líneas detectadas en burbujas de chat.
Identifica mensajes del lado izquierdo (recibidos) y derecho (enviados).

- **NLPService**
Analiza mensajes individualmente:
1. Sentimiento: hostil / neutral / positivo
2. Insultos o amenazas
3. Discriminación
4. Lenguaje soez o sexual
5. osible acoso
6. Razón o explicación
7. Analiza contexto general de la conversación:
8. Resumen global
9. Opinión / reflexión sobre interacción

- **ExcelService**
Exporta resultados del análisis a un archivo Excel.
Incluye análisis individual y resumen general.

## Instalación y ejecución

1. **Clonar repositorio**

```
git clone <repo-url>
cd chat-analyzer
```

2. **Instalar dependencias**

```
pip install -r requirements.txt
```

3. **Configurar variables de entorno (.env)**
```
AZURE_CV_ENDPOINT=<tu_endpoint>
AZURE_CV_KEY=<tu_key>
AZURE_OPENAI_ENDPOINT=<tu_endpoint>
AZURE_OPENAI_KEY=<tu_key>
AZURE_OPENAI_DEPLOYMENT=<tu_deployment>
HF_MODEL_SENT=pysentimiento/robertuito-sentiment-analysis
HF_MODEL_OFF=pysentimiento/robertuito-offensive
HF_MODEL_HATE=pysentimiento/robertuito-hate-speech
```

4. **Ejecutar la app**

```
streamlit run app.py
```

## Uso

1. Subir una imagen de chat.
2. Revisar los mensajes detectados.
3. Analizar mensajes y contexto.
4. Descargar los resultados en Excel.

## Referencia

Los datos y la metodología de prueba pueden basarse en datasets de código abierto como:

[Consorcio Madroño – Dataset de Código](http://edatos.consorciomadrono.es:8080/dataset.xhtml;jsessionid=b533c252d601e9be8cdb3bdf8b6d?persistentId=doi%3A10.21950%2FRXLJOH&version=&q=&fileTypeGroupFacet=%22C%C3%B3digo%22&fileAccess=&fileSortField=name&fileSortOrder=desc&tagPresort=false&folderPresort=true)
