# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "wenxin": ["ERNIE-Bot"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}


LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "Spark-1.5"
EMBEDDING_MODEL_LIST = ['zhipuai',  'm3e']
INIT_EMBEDDING_MODEL = "zhipuai"
DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db/chroma"
AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
LOGO1_PATH = "./figures/LOGO1.png"
LOGO2_PATH = "./figures/LOGO2.png"


# 存放历史对话
history_conversation = []


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")
class Model_center():
    """
    存储问答 Chain 的对象 

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}
        

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding: str = "openai", temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        调用带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(model=model, temperature=temperature,
                                                                                    top_k=top_k, chat_history=chat_history, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding="openai", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model=model, temperature=temperature,
                                                                       top_k=top_k, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append(
                (question, chain.answer(question, temperature, top_k)))
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()

    def create_conversation(self, message, chat_history, state):
        if chat_history!=[]:
            if chat_history not in history_conversation:
                history_conversation.append(chat_history)
                return "", [], gr.update(visible=state), gr.update(visible=not state), gr.update(visible=not state)
            else:
                return "", [], gr.update(visible=not state), gr.update(visible=state), gr.update(visible=state)
        else:
            return "", [], gr.update(visible=not state), gr.update(visible=state), gr.update(visible=state)
        
    
    def trigger_function(self, message, chat_history):
        return "", history_conversation[0]
    def trigger_function1(self, message, chat_history):
        return "", history_conversation[1]
    def trigger_function2(self, message, chat_history):
        return "", history_conversation[2]
    def trigger_function3(self, message, chat_history):
        return "", history_conversation[3]
    def trigger_function4(self, message, chat_history):
        return "", history_conversation[4]
    def trigger_function4(self, message, chat_history):
        return "", history_conversation[5]
    def trigger_function4(self, message, chat_history):
        return "", history_conversation[6]
    def trigger_function4(self, message, chat_history):
        return "", history_conversation[7]

def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt



def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    if message == None or len(message) < 1:
            return "", chat_history
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history
    except Exception as e:
        return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image(value=LOGO1_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>个人知识库助手</center></h1>
                            <center>V1.0</center>
                            """)
        gr.Image(value=LOGO2_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=1):
            s1 = gr.State(False)
            s2 = gr.State(False)
            s3 = gr.State(False)
            s4 = gr.State(False)
            s5 = gr.State(False)
            s6 = gr.State(False)
            s7 = gr.State(False)
            s8 = gr.State(False)
            e1 = gr.Button("新建对话", visible=True)
            e2 = gr.Button("新建对话", visible=False)
            e3 = gr.Button("新建对话", visible=False)
            e4 = gr.Button("新建对话", visible=False)
            e5 = gr.Button("新建对话", visible=False)
            e6 = gr.Button("新建对话", visible=False)
            e7 = gr.Button("新建对话", visible=False)
            e8 = gr.Button("新建对话", visible=False)
            b1 = gr.Button("历史对话一",visible=False)
            b2 = gr.Button("历史对话二",visible=False)
            b3 = gr.Button("历史对话三",visible=False)
            b4 = gr.Button("历史对话四",visible=False)
            b5 = gr.Button("历史对话五",visible=False)
            b6 = gr.Button("历史对话六",visible=False)
            b7 = gr.Button("历史对话七",visible=False)
            b8 = gr.Button("历史对话八",visible=False)
            # 状态
            s = [s1, s2, s3, s4, s5, s6, s7, s8]
            # 新建
            e = [e1, e2, e3, e4, e5, e6, e7, e8] 
            # 对话
            b = [b1, b2, b3, b4, b5, b6, b7, b8]


        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("知识库交流(关联记录)", icon="./figures/历史记录.png")
                db_wo_his_btn = gr.Button("知识库交流(不关联记录)", icon="./figures/无历史记录.png")
                llm_btn = gr.Button("大语言模型交流", icon="./figures/大语言模型.png")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="清除历史内容", icon="./figures/清除历史.png")

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录', file_count='multiple',
                           file_types=['.txt', '.md', '.docx', '.pdf'])# file_count='directory'
            with gr.Row():
                init_db = gr.Button("知识库文件向量化", icon="./figures/向量化.png")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="大语言模型(Large language model)",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="文本向量化模型(Embedding model)",
                                         value=INIT_EMBEDDING_MODEL)

        # 设置初始化向量数据库按钮的点击事件。当点击时，调用 create_db_info 函数，并传入用户的文件和希望使用的 Embedding 模型。
        init_db.click(create_db_info,
                      inputs=[file, embeddings], outputs=[msg])

        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                              msg, chatbot,  llm, embeddings, temperature, top_k, history_len],
                              outputs=[msg, chatbot])
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])
        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(respond, inputs=[
                      msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="minimal")
        
        for i in range(4):
            e[i].click(model_center.create_conversation, inputs=[msg, chatbot, s[i]], outputs=[msg, chatbot, e[i], b[i], e[i+1]])
        b1.click(model_center.trigger_function, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b2.click(model_center.trigger_function1, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b3.click(model_center.trigger_function2, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b4.click(model_center.trigger_function3, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b5.click(model_center.trigger_function4, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b6.click(model_center.trigger_function4, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b7.click(model_center.trigger_function4, inputs=[msg, chatbot], outputs=[msg, chatbot])
        b8.click(model_center.trigger_function4, inputs=[msg, chatbot], outputs=[msg, chatbot])
        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(respond, inputs=[
                   msg, chatbot,  llm, history_len, temperature], outputs=[msg, chatbot], show_progress="hidden")
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
    2. 初始化数据库时间可能较长，请耐心等待。
    3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
