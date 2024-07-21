# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# import os
# import subprocess
# import pandas as pd

# app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# def parse_log_file(file_path):
#     log_entries = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             columns = line.strip().split(',')
#             # Skip header row
#             if columns[0].lower() == 'lineid':
#                 continue
#             if len(columns) == 11:
#                 lineId, month, date, time, user, component, pid, address, content, eventId, eventTemplate = columns
#                 log_entries.append({
#                     'lineid': int(lineId),
#                     'month': month,
#                     'date': int(date),
#                     'time': time,
#                     'user': user,
#                     'component': component,
#                     'pid': int(pid),
#                     'address': address,
#                     'content': content,
#                     'eventid': eventId,
#                     'eventtemplate': eventTemplate,
#                 })
#     return log_entries

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'logFile' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file = request.files['logFile']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if file:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         feedback = request.form['feedback']
#         additional_comments = request.form['additionalComments']

#         # Parse log file
#         log_entries = parse_log_file(file_path)

#         try:
#             # Write the parsed log entries to a temporary CSV file
#             temp_log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_log.csv')
#             temp_log_df = pd.DataFrame(log_entries)
#             temp_log_df.to_csv(temp_log_path, index=False)

#             # Run prediction script
#             process = subprocess.Popen(['python3', 'predict.py', temp_log_path, feedback, additional_comments],
#                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = process.communicate()

#             if process.returncode != 0:
#                 raise Exception(stderr.decode('utf-8'))

#             predictions = stdout.decode('utf-8').strip().split('\n')
#             return jsonify({'predictions': predictions})
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'File not allowed'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import subprocess
import pandas as pd
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

def parse_log_file(file_path):
    log_entries = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            columns = line.strip().split(',')
            # Skip header row
            if columns[0].lower() == 'lineid':
                continue
            if len(columns) == 11:
                lineId, month, date, time, user, component, pid, address, content, eventId, eventTemplate = columns
                log_entries.append({
                    'lineid': int(lineId),
                    'month': str(month),
                    'date': int(date),
                    'time': str(time),
                    'user': str(user),
                    'component': str(component),
                    'pid': int(pid),
                    'address': str(address),
                    'content': str(content),
                    'eventid': str(eventId),
                    'eventtemplate': str(eventTemplate),
                })
    return log_entries

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'logFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['logFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        feedback = request.form['feedback']
        problem_type = request.form['problemType']

        # Parse log file
        log_entries = parse_log_file(file_path)

        # Prepare input for LLM
        input_text = f"""
        <log_file>
        {"".join([str(entry) for entry in log_entries])}
        </log_file>
        <problem_type>{problem_type}</problem_type>
        <feedback>{feedback}</feedback>
        """

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors='pt')

        # Generate a response
        outputs = model.generate(inputs.input_ids, max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'response': response})

    return jsonify({'error': 'File not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
