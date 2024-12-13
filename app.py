import json
import subprocess
import time
import random

from flask import Flask, jsonify, Response, request
from flask_cors import CORS  # 导入 CORS

import config

app = Flask(__name__)
CORS(app)  # 启用 CORS

# 用来保存进程对象
process_execute_flag = False


@app.route('/api/start-location', methods=['GET'])
def start_location():
    return Response(run_location_script(), content_type='text/event-stream')


# 路由：停止当前正在运行的脚本
@app.route('/api/stop-location', methods=['POST'])
def stop_location():
    global process_execute_flag
    if process_execute_flag is True:
        process_execute_flag = False
        return jsonify({"message": "Location calculation stopped."})
    else:
        return jsonify({"message": "No process is currently running."}), 400


# 路由：停止当前正在运行的脚本
@app.route('/api/set-config', methods=['POST'])
def set_config():
    configuration = request.get_json()
    print("接收到的配置:", configuration)

    with open('config.json', 'w') as f:
        json.dump(configuration, f)

    # 读取配置文件并赋值到全局变量中
    with open('config.json', 'r') as f:
        variables = json.load(f)
        config.set_config(variables)

    return jsonify({"message": "Configuration has saved."})


# 前端获取配置
@app.route('/api/get-config', methods=['GET'])
def get_config():
    with open('config.json', 'r') as f:
        variables = json.load(f)
        return jsonify({"message": variables})


# 这是一个用来执行定位计算的函数
def run_location_script():
    global process_execute_flag
    try:
        # 假设 calculate_location.py 是你要执行的脚本
        process = subprocess.Popen(
            ['python', 'main.py'],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获错误输出
            text=True
        )

        process_execute_flag = True

        # 实时读取标准输出并返回
        for line in process.stdout:
            if process_execute_flag is False:
                break
            # print(line)
            yield f"data: {line}\n\n"

        # # 处理错误输出（如果有）
        # for line in process.stderr:
        #     yield f"data: {line}\n\n"

        process.wait()  # 等待子进程完成

    except Exception as e:
        yield f"data: An error occurred: {str(e)}\n\n"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
