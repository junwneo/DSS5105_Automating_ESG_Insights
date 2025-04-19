from flask import Flask, request, jsonify
import os
import tempfile
from readpdf import extract_text_from_pdf, structure_and_preprocess_texts

app = Flask(__name__)

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # 检查请求中是否包含文件
    if 'pdf' not in request.files:
        return jsonify({'error': 'No uploaded file'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'File name is empty'}), 400

    # 将上传的文件保存到临时目录
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # 调用你已有的OCR处理流程
        raw_texts, pages = extract_text_from_pdf(temp_path, dpi=300, lang='en')
        processed_data = structure_and_preprocess_texts(raw_texts, pages)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 处理完后删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 返回结构化后的 JSON 数据
    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True)
