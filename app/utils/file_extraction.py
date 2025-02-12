import io
import logging
import fitz
import docx
import pandas as pd
from PIL import Image
import pytesseract
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error("Error extracting PDF: %s", e)
        raise

def extract_text_from_doc(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in document.paragraphs)
    except Exception as e:
        logger.error("Error extracting DOCX: %s", e)
        raise

def extract_text_from_excel(file_bytes: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    except Exception as e:
        logger.error("Error extracting Excel: %s", e)
        raise

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error("Error performing OCR on image: %s", e)
        raise

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        logger.error("Error extracting TXT: %s", e)
        raise

def extract_text(file) -> str:
    ext = file.filename.split(".")[-1].lower()
    file_bytes = file.file.read()
    file.file.seek(0)
    
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ["doc", "docx"]:
        return extract_text_from_doc(file_bytes)
    elif ext in ["xls", "xlsx"]:
        return extract_text_from_excel(file_bytes)
    elif ext in ["jpg", "jpeg", "png", "bmp"]:
        return extract_text_from_image(file_bytes)
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        logger.error("Unsupported file type: %s", ext)
        raise HTTPException(status_code=400, detail="Unsupported file type")
