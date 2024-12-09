from flask import Flask, jsonify
from flask_cors import CORS  # 导入 CORS

app = Flask(__name__)
CORS(app)  # 启用 CORS


@app.route('/api/data')
def get_data():
    data = {
        'message': 'Hello from Python!',
        'timestamp': '2024-12-09 10:00:00'
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
