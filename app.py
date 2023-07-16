from flask import Flask, render_template, request, jsonify
from packages.main import predict

app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model.html')
def model():
    return render_template('model.html')

if __name__ == '__main__':
    app.run()