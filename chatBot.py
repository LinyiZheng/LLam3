# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import os
 
# 在侧边栏创建标题和链接
with st.sidebar:
    st.markdown("## LLaMA3 LLM")
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
 
# 创建标题和副标题
st.title("💬 LLaMA3 Chatbot")
st.caption("🚀 由 Self-LLM 提供支持的 Streamlit 聊天机器人")
 
# 定义模型路径
model_name_or_path = '/Users/linyi/CodeMan/opensource/Llama3/Meta-Llama-3-8B-Instruct'


 
# 定义函数以获取模型和 tokenizer
@st.cache_resource
def get_model():
    # 从预训练模型获取 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    #调试
    print("tokenizer.eos_token: ",tokenizer.eos_token)
    # 从预训练模型获取模型，并设置参数
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, device_map=torch.device("mps"))
  
    return tokenizer, model
 
# 构建用户输入函数
def build_input(prompt, history=[]):
    system_format = 'system\n\n{content}'
    user_format = 'user\n\n{content}'
    assistant_format = 'assistant\n\n{content}\n'
    history.append({'role': 'user', 'content': prompt})
    prompt_str = ''
    # 拼接历史对话
    for item in history:
        if item['role'] == 'user':
            prompt_str += user_format.format(content=item['content'])
        else:
            prompt_str += assistant_format.format(content=item['content'])
    return prompt_str + 'assistant\n\n'


# 定义gpu的内存水位线
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# 清理gpu申请的内存
torch.mps.empty_cache()

# 加载 LLaMA3 模型和 tokenizer
tokenizer, model = get_model()
 
# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
# 遍历 session_state 中的所有消息，并在聊天界面上显示
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
 
# 如果用户在聊天输入框中输入内容，则执行以下操作
if prompt := st.chat_input():
    
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    
    # 构建输入
    input_str = build_input(prompt=prompt, history=st.session_state["messages"])
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to("mps")
    print("开始生成反馈 ....")
    print("eos_token_id 的值: ",tokenizer.encode('')[0])
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=512, do_sample=True,
        top_p=0.9, temperature=0.5, repetition_penalty=1.1,eos_token_id=tokenizer.eos_token_id
        )
    print("准备输出反馈 ....")
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs,skip_special_token_true=True)
    response = response.strip().replace('', "").replace('assistant\n\n', '').strip()
 
    # 将模型的输出添加到 session_state 中的 messages 列表
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
    print(st.session_state)