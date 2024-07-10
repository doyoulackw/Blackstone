#coding:gbk
from turtle import width
import gradio as gr
import pandas as pd
import os
import subprocess
import webbrowser

logo = "./figures/LOGIN_LOGO.png"

# 创建一个本地存储用户数据的Excel文件
def create_user_data_file():
    if not os.path.exists('user_data.xlsx'):
        df = pd.DataFrame(columns=['username', 'password'])
        df.to_excel('user_data.xlsx', index=False)

# 检查登录
def check_login(username, password):
    df = pd.read_excel('user_data.xlsx')
    user_row = df[(df['username'] == username) & (df['password'] == password)]
    if not user_row.empty:
        return "成功登陆。", True
    else:
        return "无效的用户名或密码。", False

# 保存新用户
def save_user(username, password):
    df = pd.read_excel('user_data.xlsx')
    if username in df['username'].values:
        return "Username already exists!"
    new_user = pd.DataFrame({'username': [username], 'password': [password]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_excel('user_data.xlsx', index=False)
    return "User registered successfully!"

# 登录界面
def login_interface():
    with gr.Blocks() as login:
        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=3):
                gr.Image(value=logo, height=300, show_label=False, show_download_button=False, container=False)
                username = gr.Textbox(label="Username")
                password = gr.Textbox(label="Password", type="password")
                with gr.Row():
                    login_button = gr.Button("Login", icon="./figures/登录.png")
                    enroll_button = gr.Button("Enroll", icon="./figures/注册.png")
                output = gr.Textbox(label="Result")
            gr.Column(scale=1)
        
        def handle_login(username, password):
            message, success = check_login(username, password)
            if success:
                # 在当前页面进行重定向
                #redirect_script = "<script>window.location.href = 'http://127.0.0.1:7860/';</script>"
                webbrowser.open_new_tab("http://127.0.0.1:7860/")
                return "正在打开个人知识库助手。"
            return message

        def handle_enroll(username, password):
            message = save_user(username, password)
            return message

        login_button.click(handle_login, [username, password], output)
        enroll_button.click(handle_enroll, [username, password], output)

    return login

create_user_data_file()

# 启动 run_gradio.py 文件
subprocess.Popen(["python", "serve/run_gradio.py"])

# 启动 Gradio 登录界面
with gr.Blocks() as demo:
    login_interface()

demo.launch(server_port=7850)
