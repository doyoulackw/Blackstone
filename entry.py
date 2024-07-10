#coding:gbk
from turtle import width
import gradio as gr
import pandas as pd
import os
import subprocess
import webbrowser

logo = "./figures/LOGIN_LOGO.png"

# ����һ�����ش洢�û����ݵ�Excel�ļ�
def create_user_data_file():
    if not os.path.exists('user_data.xlsx'):
        df = pd.DataFrame(columns=['username', 'password'])
        df.to_excel('user_data.xlsx', index=False)

# ����¼
def check_login(username, password):
    df = pd.read_excel('user_data.xlsx')
    user_row = df[(df['username'] == username) & (df['password'] == password)]
    if not user_row.empty:
        return "�ɹ���½��", True
    else:
        return "��Ч���û��������롣", False

# �������û�
def save_user(username, password):
    df = pd.read_excel('user_data.xlsx')
    if username in df['username'].values:
        return "Username already exists!"
    new_user = pd.DataFrame({'username': [username], 'password': [password]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_excel('user_data.xlsx', index=False)
    return "User registered successfully!"

# ��¼����
def login_interface():
    with gr.Blocks() as login:
        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=3):
                gr.Image(value=logo, height=300, show_label=False, show_download_button=False, container=False)
                username = gr.Textbox(label="Username")
                password = gr.Textbox(label="Password", type="password")
                with gr.Row():
                    login_button = gr.Button("Login", icon="./figures/��¼.png")
                    enroll_button = gr.Button("Enroll", icon="./figures/ע��.png")
                output = gr.Textbox(label="Result")
            gr.Column(scale=1)
        
        def handle_login(username, password):
            message, success = check_login(username, password)
            if success:
                # �ڵ�ǰҳ������ض���
                #redirect_script = "<script>window.location.href = 'http://127.0.0.1:7860/';</script>"
                webbrowser.open_new_tab("http://127.0.0.1:7860/")
                return "���ڴ򿪸���֪ʶ�����֡�"
            return message

        def handle_enroll(username, password):
            message = save_user(username, password)
            return message

        login_button.click(handle_login, [username, password], output)
        enroll_button.click(handle_enroll, [username, password], output)

    return login

create_user_data_file()

# ���� run_gradio.py �ļ�
subprocess.Popen(["python", "serve/run_gradio.py"])

# ���� Gradio ��¼����
with gr.Blocks() as demo:
    login_interface()

demo.launch(server_port=7850)
