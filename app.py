from flask import Flask, render_template, request, jsonify
import os
from packages.main import predict

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model.html',  methods=['POST', 'GET'])
def model():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = 'uploads/'+image_file.filename
        image_file.save(image_path)
        gender = request.form.get('gender')
        view = request.form.get('view')
        age = '0' + request.form.get('age') + 'Y'
        pre = predict(image_path, age, gender, view)[0][0] * 100
        os.remove(image_path)
        pre = round(pre, 2)
        return jsonify(pre), 201
    return render_template('model.html')

if __name__ == '__main__':
    app.run()