from flask import Flask, request, jsonify
import os, tempfile
from clean_text import process_json_to_text, restore_punctuation, filter_sentences, write_sentences_to_file

app = Flask(__name__)

@app.route('/process_text', methods=['POST'])
def process_text():
    # 检查请求中是否包含文件，假设文件名为 json_file
    if 'json_file' not in request.files:
        return jsonify({'error': 'No uploaded file'}), 400

    file = request.files['json_file']
    if file.filename == '':
        return jsonify({'error': 'File name is empty'}), 400

    # 保存上传的 JSON 文件到临时位置
    temp_dir = tempfile.gettempdir()
    temp_json_path = os.path.join(temp_dir, file.filename)
    file.save(temp_json_path)
    
    try:
        # 设置临时输出文件和最终输出文件路径
        temp_output = os.path.join(temp_dir, "cleaned_text.txt")
        final_output = os.path.join(temp_dir, "restored_text.txt")
        filtered_output = os.path.join(temp_dir, "final_output.txt")
        
        # 1. 处理 JSON，合并文本（过滤目录页）
        merged_text = process_json_to_text(temp_json_path, temp_output, catalog_keywords=["contents", "Alignment", "Page Number", "Glossary", "Annex"])
        
        # 2. 恢复标点
        restored_text = restore_punctuation(merged_text, final_output)
        
        # 3. 分句并过滤短句
        filtered_sentences = filter_sentences(restored_text, min_char=10, min_tokens=3)
        
        # 4. 将过滤后的句子写入最终输出文件
        write_sentences_to_file(filtered_sentences, filtered_output)
        
        # 可选：返回处理后的文本作为响应（或返回处理后文件的路径）
        return jsonify({filtered_sentences})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 清理临时文件（根据需求决定是否删除）
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        # 注意：其他临时文件可视情况处理
        

if __name__ == '__main__':
    app.run(debug=True)
