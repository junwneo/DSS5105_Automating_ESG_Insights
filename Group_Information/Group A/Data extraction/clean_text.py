import re
import json
from deepmultilingualpunctuation import PunctuationModel
import nltk
from nltk.tokenize import sent_tokenize

# 清洗函数（与之前一致）
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'[^\x00-\x7F\u4e00-\u9fff,.?!，。%！？：:；;-“”‘’\s]', '', text)
    text = re.sub(r'[^0-9A-Za-z\u4e00-\u9fa5\s,.!?，。%！？：:；;-“”‘’\'"]', '', text)
    text = re.sub(r'[T]+', 'T', text)
    text = re.sub(r'[|]+', ' ', text)
    text = re.sub(r'[\u200B\u200C]+', '', text)
    text = re.sub(r'(?<=\s)\.', '', text)
    text = re.sub(r'\b(\w+)\s+\1\b', '', text)
    text = re.sub(r'[A-Za-z]{15,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([a-zA-Z0-9])\s*,\s*([A-Za-z])', r'\1, \2', text)
    text = re.sub(r'(\d+)\s+([a-zA-Z]+)', r'\1\2', text)
    text = re.sub(r'(\d+\.)\s+([A-Za-z])', r'\1 \2', text)
    return text.strip()

def process_json_to_text(json_file, temp_output, catalog_keywords=["contents"]):
    """
    从 JSON 中读取各页文本，进行清洗与目录页过滤，
    将所有页面文本合并后保存到临时文件，并返回合并后的文本。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_text = []
    for page in data.get('report_pages', []):
        content = page.get('content', '')
        cleaned_text = clean_text(content)
        if any(keyword.lower() in cleaned_text.lower() for keyword in catalog_keywords):
            continue
        all_text.append(cleaned_text)
    
    merged_text = "\n".join(all_text)
    with open(temp_output, 'w', encoding='utf-8') as f:
        f.write(merged_text)
    return merged_text

def restore_punctuation(text, final_output):
    """
    使用 deepmultilingualpunctuation 模型恢复标点，并保存到最终文件中。
    """
    model = PunctuationModel()
    restored_text = model.restore_punctuation(text)
    with open(final_output, 'w', encoding='utf-8') as f:
        f.write(restored_text)
    return restored_text

def is_short_sentence(sentence, min_char=10, min_tokens=3):
    """
    判断句子是否过短（忽略标点符号），
    当句子字符数少于 min_char 或词数少于 min_tokens 时，认为句子太短。
    """
    sentence_no_punc = re.sub(r'[^\w\s]', '', sentence)
    if len(sentence_no_punc.strip()) < min_char:
        return True
    tokens = sentence_no_punc.split()
    if len(tokens) < min_tokens:
        return True
    return False

def filter_sentences(text, min_char=10, min_tokens=3):
    """
    使用 nltk 对输入文本分句，然后过滤掉过短句子，
    返回一个句子列表。
    """
    sentences = sent_tokenize(text)
    filtered = [s.strip() for s in sentences if not is_short_sentence(s, min_char, min_tokens)]
    return filtered

def write_sentences_to_file(sentences, output_file):
    """
    将句子列表写入 TXT 文件，每句占一行。
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n")
