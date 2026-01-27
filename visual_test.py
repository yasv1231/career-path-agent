import streamlit as st
import subprocess
import sys
import time

st.set_page_config(page_title="Career Path Agent — Visual Test", layout="wide")

st.title("Career Path Agent — 可视化测试")

st.markdown("""
使用此页面可以以可视化方式运行 `main.py` 并查看 stdout 输出。点击 **Start** 启动子进程，点击 **Stop** 尝试终止它。
""")

if 'proc' not in st.session_state:
    st.session_state.proc = None
    st.session_state.logs = ""

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Start"):
        if st.session_state.proc is None:
            st.session_state.logs = ""
            # 启动子进程使用当前 Python 可执行文件
            st.session_state.proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
    if st.button("Stop"):
        if st.session_state.proc is not None:
            try:
                st.session_state.proc.terminate()
            except Exception:
                pass

with col2:
    log_area = st.empty()

# 输入区域：允许将文本发送到子进程的 stdin
with st.expander("Send input to process"):
    user_input = st.text_area("Input (will be sent as a line)", height=80)
    if st.button("Send"):
        if st.session_state.proc is not None:
            try:
                proc = st.session_state.proc
                proc.stdin.write(user_input + "\n")
                proc.stdin.flush()
                st.session_state.logs += f"\n> {user_input}\n"
            except Exception as e:
                st.error(f"Failed to send input: {e}")

if st.session_state.proc is not None:
    p = st.session_state.proc
    # 读取并显示输出
    try:
        while True:
            line = p.stdout.readline()
            if line == "" and p.poll() is not None:
                break
            if line:
                st.session_state.logs += line
                log_area.code(st.session_state.logs)
                # 轻微暂停以允许 UI 更新
                time.sleep(0.01)
    except Exception:
        pass

    # 子进程结束后显示退出码并清理
    if p.poll() is not None:
        ret = p.returncode
        st.session_state.proc = None
        st.success(f"子进程退出，返回码: {ret}")
else:
    # 显示现有日志（如果有）
    if st.session_state.logs:
        log_area.code(st.session_state.logs)
