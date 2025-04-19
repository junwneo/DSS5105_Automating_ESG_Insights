import re  
import os
import json
import cv2
import numpy as np
import jieba
import langid
from pdf2image import convert_from_path
from PIL import Image
import easyocr
from google.cloud import translate_v2 as translate  # 引入 Google Translate API

# ----- 语言映射配置 -----
# EasyOCR 支持的语言：中文简体 "ch_sim", 英文 "en", 法语 "fr", 德语 "de", 西班牙语 "es"
LANG_MAP = {
    'zh': 'ch_sim',  # 中文
    'en': 'en',      # 英文
    'fr': 'fr',      # 法语
    'de': 'de',      # 德语
    'es': 'es',      # 西班牙语
}

# ----- 图像预处理部分 -----
def preprocess_image(image):
    """
    将 PIL 图像转换为二值图像：
    - 转换为灰度
    - 去噪（使用 fastNlMeansDenoising）
    - 增强对比度（线性拉伸）
    - 自适应二值化
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度

    # 去噪：使用 fastNlMeansDenoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # 增强对比度：拉伸对比度
    min_val, max_val = np.min(denoised), np.max(denoised)
    contrast_enhanced = np.uint8(255 * (denoised - min_val) / (max_val - min_val))

    # 自适应二值化处理
    binary = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(binary)

# ----- 表格区域检测与过滤 -----
def detect_table_regions(cv_img):
    """
    检测图像中的表格区域，并通过尺寸及宽高比进行过滤，
    返回表格区域的边界列表 [ (x1, y1, x2, y2), ... ]
    """
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 200, 255, cv2.THRESH_BINARY)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    table_mask = cv2.add(detect_horizontal, detect_vertical)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 20:
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.5:
                table_bboxes.append((x, y, x + w, y + h))
    return table_bboxes

def remove_table_regions(cv_img, table_bboxes):
    """
    将图像中检测到的表格区域用白色填充，
    从而减少 OCR 时将表格内的文字混入正文的情况。
    """
    img_no_table = cv_img.copy()
    for bbox in table_bboxes:
        x1, y1, x2, y2 = bbox
        if len(img_no_table.shape) == 2:
            cv2.rectangle(img_no_table, (x1, y1), (x2, y2), 255, -1)
        else:
            cv2.rectangle(img_no_table, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return img_no_table

# ----- OCR 文本提取部分（使用 EasyOCR） -----
def extract_text_from_pdf(pdf_path, dpi=300, lang='ch_sim+en'):
    """
    对扫描型 PDF 进行 OCR：
    - 将 PDF 页面转换为图像
    - 预处理图像（灰度、去噪、二值化）
    - 检测并抹去表格区域
    - 对剩余区域使用 EasyOCR 进行 OCR 识别
    返回每页提取的文本列表和页面图像列表。
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    extracted_texts = []
    langs = lang.split('+')
    reader = easyocr.Reader(langs, gpu=True)
    for i, page in enumerate(pages):
        processed_img = preprocess_image(page)
        processed_cv = np.array(processed_img)
        table_bboxes = detect_table_regions(processed_cv)
        processed_no_table = remove_table_regions(processed_cv, table_bboxes)
        results = reader.readtext(processed_no_table, detail=0, paragraph=True)
        text = "\n".join(results)
        extracted_texts.append(text)
    return extracted_texts, pages

# ----- 文本预处理部分 -----
def fullwidth_to_halfwidth(text):
    """将全角字符转换为半角字符"""
    result = ""
    for char in text:
        code = ord(char)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result += chr(code)
    return result

def clean_text(text):
    """
    清洗文本：
    - 转换全角为半角
    - 去除多余空格和换行符
    - 保留中英文及常见标点
    """
    text = fullwidth_to_halfwidth(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F\u4e00-\u9fff,.?!，。！？：:；;“”‘’\s]', '', text)
    return text.strip()

def tokenize_text(text):
    """使用 jieba 进行中文分词"""
    tokens = jieba.lcut(text)
    return tokens

def detect_language(text):
    """使用 langid 检测文本语言"""
    lang_code, _ = langid.classify(text)
    return lang_code

# ----- Google Translate 翻译模块 -----
def translate_text(text, target_lang='en'):
    """
    调用 Google Translate API 将文本翻译为目标语言（默认英文）。
    注意：请确保已安装 google-cloud-translate 并正确配置 Google Cloud 认证。
    """
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_lang)
    return result['translatedText']

# ----- 结构化处理文本 -----
def structure_and_preprocess_texts(raw_texts, pages):
    """
    将 OCR 得到的文本进行清洗、结构化，并分词。
    如果文本不是英文，则调用翻译模块将其翻译为英文。
    返回结构化后的 JSON 数据。
    """
    structured_data = {"report_pages": []}
    for i, text in enumerate(raw_texts):
        cleaned = clean_text(text)
        page_language = detect_language(cleaned)
        lang_setting = LANG_MAP.get(page_language, 'en')
        reader_local = easyocr.Reader([lang_setting], gpu=True)
        page_img = np.array(pages[i])
        results_local = reader_local.readtext(page_img, detail=0, paragraph=True)
        page_text = "\n".join(results_local)
        cleaned = clean_text(page_text)
        
        page_language = detect_language(cleaned)
        if page_language != 'en':
            cleaned = translate_text(cleaned, target_lang='en')
        
        page_data = {
            "page_number": i + 1,
            "content": cleaned,
            "tokens": tokenize_text(cleaned)
        }
        structured_data["report_pages"].append(page_data)
    return structured_data
