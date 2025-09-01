# -*- coding: utf-8 -*-
import os, re
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import plotly.graph_objects as go

# ========= Pretendard 폰트 (repo 내 fonts/Pretendard-Bold.ttf) =========
FONT_PATH = "fonts/Pretendard-Bold.ttf"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
plt.rc("font", family="Pretendard")
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="인물 관계 곡선 + 네트워크", layout="wide", page_icon="📖")
st.title("📖 인물 관계 시각화: 곡선 + 네트워크 (Plotly)")

# ========= 조사 허용 정규식(이름 인식) =========
JOSA_GENERAL = (
    r"(?:이|가|은|는|을|를|과|와|랑|이랑|하고|도|만|부터|까지|에게|한테|께|께서|에서|으로|로|의|야|아|여)?"
)
# '나'는 '나가다' 오탐 방지 위해 '가' 제외
JOSA_FOR_NA = (
    r"(?:은|는|을|를|과|와|랑|이랑|하고|도|만|부터|까지|에게|한테|께|께서|에서|으로|로|의|야|아|여)?"
)

def whole_word_korean(name: str, josa_pat: str) -> re.Pattern:
    return re.compile(
        rf"(^|[^가-힣A-Za-z0-9_]){re.escape(name)}{josa_pat}($|[^가-힣A-Za-z0-9_])"
    )

PAT = {
    "차미": whole_word_korean("차미", JOSA_GENERAL),
    "오란": whole_word_korean("오란", JOSA_GENERAL),
    "나":   whole_word_korean("나",   JOSA_FOR_NA),
}

# ========= 문장 분리 =========
def split_sentences(text: str):
    parts = re.split(r"[.?!;]+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]

def has(name: str, s: str) -> bool:
    return PAT[name].search(s) is not None

# ========= 분석: 곡선 + 네트워크 가중치 =========
PAIRS = [("차미", "오란"), ("나", "차미"), ("나", "오란")]

def analyze_all(text: str):
    sents = split_sentences(text)
    # 누적 곡선용
    score = {f"{a}-{b}": 0 for a, b in PAIRS}
    timeline = []
    # 네트워크 가중치(총 동시 등장 횟수)
    weights = {(a, b): 0 for a, b in PAIRS}

    for i, s in enumerate(sents, start=1):
        present = {n: has(n, s) for n in ["나", "차미", "오란"]}
        for a, b in PAIRS:
            if present[a] and present[b]:
                score[f"{a}-{b}"] += 1
                weights[(a, b)] += 1
        timeline.append((i, score.copy()))
    return timeline, weights

# ========= 곡선 시각화 (matplotlib) =========
def plot_curves(timeline):
    if not timeline:
        return
    x = [i for i, _ in timeline]
    series = {k: [t[1][k] for t in timeline] for k in timeline[-1][1].keys()}

    fig = plt.figure(figsize=(9, 5))
    for k, y in series.items():
        plt.plot(x, y, marker="o", linewidth=2, label=k)
    plt.title("인물 관계 친밀도 흐름 (문장 동시 등장 누적)")
    plt.xlabel("소설 진행 (문장 순서)")
    plt.ylabel("친밀도 점수")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig)

# ========= 네트워크 시각화 (Plotly) =========
def plot_network(weights: dict, min_weight: int = 1, seed: int = 42):
    # NetworkX 그래프
    G = nx.Graph()
    for n in ["나", "차미", "오란"]:
        G.add_node(n)

    for (a, b), w in weights.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=w)

    # 배치 (고정 시드로 재현성)
    pos = nx.spring_layout(G, seed=seed, k=1.2)

    # 엣지 선(각 엣지를 별도 trace로 두께 반영)
    edge_traces = []
    for (u, v, data) in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = data.get("weight", 1)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(width=1 + 2*w),
                hoverinfo="text",
                text=[f"{u}–{v}: {w}", f"{u}–{v}: {w}"],
                showlegend=False,
            )
        )

    # 노드 점
    strengths = {}
    for n in G.nodes():
        strengths[n] = sum(G[n][nbr]["weight"] for nbr in G.neighbors(n)) if G.degree(n) > 0 else 0

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [20 + 10*strengths[n] for n in G.nodes()]
    node_text = [f"{n} (연결강도: {strengths[n]})" for n in G.nodes()]
    node_color_map = {"나": "#6C5CE7", "차미": "#00B894", "오란": "#0984E3"}
    node_color = [node_color_map.get(n, "#3498DB") for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(size=node_size, line=dict(width=1), color=node_color),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        width=900, height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="관계 네트워크 (동시 등장 가중치)",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

sample_text = """나와 차미는 오늘 처음으로 오란을 만났다. 차미가 오란과 오래 이야기했다!
다음 날, 나는 차미와 다시 만났고 오란도 잠깐 합류했다.
점심시간에 나와 오란은 우연히 마주쳤다. 저녁에는 차미와 오란이 또 함께 있었다.
밤이 되어 나와 차미, 오란까지 셋이서 메시지를 주고받았다."""

with st.sidebar:
    st.subheader("옵션")
    min_w = st.slider("네트워크: 엣지 최소 가중치", 1, 5, 1, 1)
    seed = st.number_input("레이아웃 시드", value=42, step=1)

text = st.text_area("소설 텍스트 입력", sample_text, height=240)

if st.button("분석하기"):
    timeline, weights = analyze_all(text)

    st.subheader("관계 곡선")
    plot_curves(timeline)

    st.subheader("관계 네트워크")
    plot_network(weights, min_weight=min_w, seed=seed)

