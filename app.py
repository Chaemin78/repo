import os
import time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
from openai import OpenAI


# ==============================
# 0) 설정
# ==============================
st.set_page_config(page_title="Solar Chat Room", page_icon="☀️", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "kcelectra-toxic-best")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 토크나이저는 기존처럼 디렉토리에서 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# 1) config로 "빈" 모델 생성 (아직 가중치 없음, 그냥 일반 nn.Module)
config = AutoConfig.from_pretrained(MODEL_DIR)
clf_model = AutoModelForSequenceClassification.from_config(config)

# 2) safetensors에서 state_dict 직접 로드
state_path = os.path.join(MODEL_DIR, "model.safetensors")
state_dict = load_file(state_path)  # <- safetensors.torch.load_file

missing, unexpected = clf_model.load_state_dict(state_dict, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)

# 3) 이제 진짜 텐서가 올라간 상태이니 .to() 해도 meta 이슈 없음
clf_model = clf_model.to(DEVICE).eval()


from dotenv import load_dotenv
load_dotenv()

SOLAR_API_KEY = os.getenv("UPSTAGE_API_KEY")
if SOLAR_API_KEY is None:
    raise ValueError("환경변수 UPSTAGE_API_KEY가 없습니다. .env에 설정하세요.")

client = OpenAI(api_key=SOLAR_API_KEY, base_url="https://api.upstage.ai/v1")

# 데모 파라미터
BOT_NAMES = ["민수", "지아"] 
USER_NAME = "나"
TOXIC_THRESHOLD = 0.50


# ==============================
# 1) 유틸 함수
# ==============================
@torch.inference_mode()
def classify_toxicity(text: str):
    # 모델이 올라간 첫 파라미터의 device로 보냄 (device_map=auto 대비)
    target_device = next(clf_model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(target_device)

    logits = clf_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    return float(probs[0]), float(probs[1])


def solar_reply(history_messages, speaker_name: str) -> str:
    transcript = "\n".join([f"{m['name']}: {m['text']}" for m in history_messages[-8:]])


    system_msg = {
    "role": "system",
    "content": (
        f"너는 10대 청소년 {speaker_name}이고, 자연스럽게 대화해야 해.\n\n"
        "규칙:\n"
        "- 답변에는 오직 한 문장만 출력할 것. (최대 40자 내외)\n"
        "- 설명, 해석, 메타코멘트, 예시, 생각 과정, 번역, 요약 등을 절대 쓰지 말 것.\n"
        "- 괄호() 안에 해설 쓰지 말 것.\n"
        "- <think>, </think> 같은 태그를 포함한 어떤 태그도 출력하지 말 것.\n"
        "- 본인 이름/인사/자기소개/서명/이모지/해시태그/따옴표도 절대 금지.\n"
        "- '민수:' 같은 화자 표기 금지. 내용만 출력.\n"
        "- 앞 사람 말을 그대로 따라하지 말 것.\n"
        "- 반말로, 바로 직전 사람의 말에 자연스럽게 이어지는 한 마디만 출력할 것.\n"
    )
}


    user_msg = {
    "role": "user",
    "content": (
        (transcript + "\n") if transcript else "" 
    ) + f"{speaker_name}의 다음 한 마디만 작성해. "
        "이름 없이 내용만, 한 문장만 출력해."
}


    stop_list = [f"\n{n}:" for n in [speaker_name, "나", "민수", "지아", "현우"]]

    resp = client.chat.completions.create(
        model="solar-pro2",
        messages=[system_msg, user_msg],
        stream=False,
        max_tokens=80,
        temperature=0.3,
        top_p=0.9,
        stop=stop_list,
    )
    return resp.choices[0].message.content.strip()


# ==============================
# 2) 세션 상태
# ==============================
if "chat" not in st.session_state:
    st.session_state.chat = []
if "init_done" not in st.session_state:
    st.session_state.init_done = False

# ==============================
# 4) 대화 영역
# ==============================
st.title("☀️ 청진기 채팅방")

chat_placeholder = st.empty()  # 채팅을 렌더링할 자리

def render_all_messages():
    with chat_placeholder.container():
        for msg in st.session_state.chat:
            who = USER_NAME if msg["role"] == "user" else msg["name"]
            st.markdown(f"**{who}** · *{msg['ts']}*  \n{msg['text']}")

# 처음 로드 시 현재까지의 대화 렌더
render_all_messages()
st.divider()


# ==============================
# 5) 입력 & 로직
# ==============================

user_text = st.text_input("내 메시지", placeholder="메세지를 입력하세요.")
send = st.button("보내기", type="primary")

if send and user_text.strip():
    user_text = user_text.strip()

    # 1) 사용자 메시지 기록 + 바로 렌더
    user_msg = {
        "role": "user",
        "name": USER_NAME,
        "text": user_text,
        "ts": time.strftime("%H:%M:%S"),
    }
    st.session_state.chat.append(user_msg)
    render_all_messages()

    # 2) 악성 판정
    p_tox, p_clean = classify_toxicity(user_text)
    # 디버깅용으로 보고 싶으면:
    # st.write("p_tox:", p_tox, "p_clean:", p_clean)

    if p_tox >= TOXIC_THRESHOLD:
        st.error("🚨 경고: 악성 발화 감지 — 친구와 대화할 때는 바르고 고운 말을 사용해요.")
        # 여기서 바로 종료
        st.stop()

    # 3) 정상일 때: 봇 2명 순차 응답 + 2초 텀
    for name in BOT_NAMES:
        with st.spinner("메세지 작성 중..."):
            reply = solar_reply(st.session_state.chat, name)
            bot_msg = {
                "role": "bot",
                "name": name,
                "text": reply,
                "ts": time.strftime("%H:%M:%S"),
            }
            st.session_state.chat.append(bot_msg)
            render_all_messages()   # 새 메시지까지 포함해서 다시 그림
        time.sleep(2)                # 각 봇 사이 2초 텀
