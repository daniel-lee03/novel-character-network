# -*- coding: utf-8 -*-
# 실행: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
import os, re, json, time, itertools
from typing import List, Set, Dict, Tuple
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import kss
import streamlit.components.v1 as components
import requests

# =============== 기본 UI ===============
st.set_page_config(page_title="소설 인물 관계 네트워크(초간단)", page_icon="📚", layout="wide")
st.title("📚 소설 인물 관계 네트워크 — 차미 / 오란 / 녹주")

with st.sidebar:
    st.header("옵션")
    st.caption("등장 인물은 고정: 차미, 오란, 녹주")
    window = st.slider("동시출현 윈도우(문장 단위)", 1, 3, 1)
    min_edge = st.slider("엣지 가중치 임계값", 1, 5, 1)
    use_hf = st.toggle("Hugging Face 감정 분석 사용(선택)", value=False,
                       help="문장 감정(긍/부정)을 평균 내서 엣지 색에 반영합니다.")
    hf_model = st.text_input("감정 모델 ID", value="nlpai-lab/klue-roberta-base-finetuned-nsmc",
                             help="한국어 감정(긍/부정) 분류 모델 예시")
    hf_token = st.text_input("HF_TOKEN", type="password",
                             help="토큰이 없으면 공개모델에서 느릴 수 있어요(선택).")

SAMPLE = """오늘을 얼마나 기다렸는지 모른다. 차미는 가방을 여미고 오란을 기다렸다.
오란은 늦게 도착했고, 녹주는 두 사람을 멀리서 바라보았다.
차미와 녹주는 짧게 인사를 나누고, 오란은 미안하다고 말했다.
그날 이후 차미와 오란, 녹주의 관계는 조금씩 달라졌다."""
text = st.text_area("소설 본문 붙여넣기", value=SAMPLE, height=220)
uploaded = st.file_uploader("또는 .txt 업로드", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")

run = st.button("분석 실행")

# =============== 유틸 ===============
NAMES = ["차미", "오란", "녹주"]

def normalize_name(s: str) -> str:
    s = s.strip()
    # 호칭/접미사 간단 제거
    s = re.sub(r"(씨|님|군|양|선생님?)$", "", s)
    return s

# ‘차미/오란/녹주’ 다양한 표기 허용(호칭 포함)
NAME_RX = re.compile(r"\b(차미|오란|녹주)(씨|님|군|양|선생님?)?\b")

def split_sentences(t: str) -> List[str]:
    # 매우 안정적인 한국어 문장 분리기
    return [s.strip() for s in kss.split_sentences(t) if s.strip()]

def sentence_window_indices(i: int, w: int, n: int) -> List[int]:
    if w <= 1: return [i]
    idxs = {i}
    for k in range(1, w):
        if i-k >= 0: idxs.add(i-k)
        if i+k < n: idxs.add(i+k)
    return sorted(idxs)

# ---- Hugging Face Inference API (선택) ----
def hf_sentiment(sentence: str, model_id: str, token: str = "", timeout: float = 15.0) -> float:
    """
    반환: [+1.0 ~ -1.0] 범위의 스코어. (긍정≈+1, 부정≈-1)
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"inputs": sentence}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code == 503:
            # 모델 로드 대기
            time.sleep(2.0)
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        # 예: [{"label":"positive","score":0.98},{"label":"negative","score":0.02}]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            scores = {d.get("label", "").lower(): d.get("score", 0.0) for d in out}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            return float(pos - neg)  # +면 긍정, -면 부정
    except Exception:
        pass
    return 0.0  # 실패 시 중립

# =============== 실행 로직 ===============
if run:
    if not text.strip():
        st.warning("본문을 입력해주세요.")
        st.stop()

    sentences = split_sentences(text)
    st.write(f"문장 수: {len(sentences)}")

    # 문장별 등장 인물 세트 + (선택) 문장 감정
    per_by_sent: List[Set[str]] = []
    sent_sentiment: List[float] = []

    for s in sentences:
        found = set(normalize_name(m.group(1)) for m in NAME_RX.finditer(s))
        # 화이트리스트 적용(고정 3인)
        found = {n for n in found if n in NAMES}
        per_by_sent.append(found)

        if use_hf:
            sent_sentiment.append(hf_sentiment(s, hf_model, hf_token))
        else:
            sent_sentiment.append(0.0)

    # 네트워크 그래프 구성
    G = nx.Graph()
    freq: Dict[str, int] = {n: 0 for n in NAMES}
    pair_sent_scores: Dict[Tuple[str, str], list] = {}

    n = len(per_by_sent)
    for i in range(n):
        idxs = sentence_window_indices(i, window, n)
        people = set().union(*[per_by_sent[j] for j in idxs])
        for p in people:
            freq[p] += 1
            if p not in G:
                G.add_node(p)
        for a, b in itertools.combinations(sorted(list(people)), 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
            # 감정 스코어 수집(선택)
            if use_hf:
                key = (a, b)
                pair_sent_scores.setdefault(key, []).append(sent_sentiment[i])

    # 엣지 임계값 필터
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_edge]
    G.remove_edges_from(to_remove)

    # 요약 표
    nodes_df = pd.DataFrame(
        [{"name": n, "degree": G.degree(n) if n in G else 0, "frequency": freq.get(n, 0)} for n in NAMES]
    ).sort_values(["degree", "frequency"], ascending=[False, False])

    # --- 엣지 요약 (안전 패치) ---
edges_data = [
    {"source": u, "target": v, "weight": d.get("weight", 0)}
    for u, v, d in G.edges(data=True)
]
edges_df = pd.DataFrame(edges_data)

if not edges_df.empty:
    edges_df = edges_df.sort_values("weight", ascending=False)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("노드 요약")
        st.dataframe(nodes_df, use_container_width=True, height=260)
        st.download_button("노드 CSV 다운로드", nodes_df.to_csv(index=False).encode("utf-8"),
                           file_name="nodes.csv", mime="text/csv")
    with col2:
        st.subheader("엣지 요약")
        st.dataframe(edges_df, use_container_width=True, height=260)
        st.download_button("엣지 CSV 다운로드", edges_df.to_csv(index=False).encode("utf-8"),
                           file_name="edges.csv", mime="text/csv")

    # Pyvis 시각화
    st.subheader("관계 네트워크(드래그/줌 가능)")
    net = Network(height="620px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=160, spring_strength=0.005, damping=0.6)

    # 노드 표시(빈도 기준 크기)
    fmax = max(1, max(freq.values()))
    for n in NAMES:
        if n in G.nodes:
            size = 20 + 30 * (freq[n] / fmax)
            label = f"{n} (deg={G.degree(n)}, f={freq[n]})"
            net.add_node(n, label=label, title=label, size=float(size))
        else:
            # 그래프에 안 뜨면 스킵
            pass

    # 엣지 표시(가중치 두께 + 선택 시 감정색)
    def color_from_score(s: float) -> str:
        # 부정쪽은 푸른색, 긍정은 붉은색 계열(간단)
        if s > 0.1: return "#c0392b"   # red-ish
        if s < -0.1: return "#2980b9"  # blue-ish
        return "#7f8c8d"               # neutral gray

    for u, v, d in G.edges(data=True):
        val = int(d.get("weight", 1))
        col = "#7f8c8d"
        title = f"{u}–{v} (w={val})"
        if use_hf:
            key = tuple(sorted([u, v]))
            scores = pair_sent_scores.get(key, [])
            if scores:
                avg = sum(scores)/len(scores)
                col = color_from_score(avg)
                title += f" | 감정 평균={avg:+.2f}"
        net.add_edge(u, v, value=val, title=title, color=col, width=1+val)

    html_path = "graph.html"
    net.show(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=640, scrolling=True)

    st.success("완료!")

st.markdown("---")
st.caption("고정 인물: 차미 · 오란 · 녹주 | 문장 동시출현 기반 | (선택) Hugging Face 감정 분류")
