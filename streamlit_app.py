# -*- coding: utf-8 -*-
# 실행: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
import os, re, itertools, time
from typing import List, Set, Dict
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import kss
import streamlit.components.v1 as components

# =============================
# 기본 설정
# =============================
st.set_page_config(page_title="소설 인물 관계 네트워크", page_icon="📚", layout="wide")
st.title("📚 소설 인물 관계 네트워크 — 차미 / 오란 / 녹주")

with st.sidebar:
    st.header("옵션")
    st.caption("등장 인물은 고정: 차미, 오란, 녹주")
    window = st.slider("동시출현 윈도우(문장 단위)", 1, 3, 1)
    min_edge = st.slider("엣지 가중치 임계값", 1, 5, 1)

SAMPLE = """오늘을 얼마나 기다렸는지 모른다. 차미는 가방을 여미고 오란을 기다렸다.
오란은 늦게 도착했고, 녹주는 두 사람을 멀리서 바라보았다.
차미와 녹주는 짧게 인사를 나누고, 오란은 미안하다고 말했다.
그날 이후 차미와 오란, 녹주의 관계는 조금씩 달라졌다."""
text = st.text_area("소설 본문 붙여넣기", value=SAMPLE, height=220)
uploaded = st.file_uploader("또는 .txt 업로드", type=["txt"])
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")

run = st.button("분석 실행")

# =============================
# 유틸 함수
# =============================
NAMES = ["차미", "오란", "녹주"]
NAME_RX = re.compile(r"(차미|오란|녹주)(씨|님|군|양|선생님?)?")

def normalize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"(씨|님|군|양|선생님?)$", "", s)
    return s

def split_sentences(t: str) -> List[str]:
    return [s.strip() for s in kss.split_sentences(t) if s.strip()]

def sentence_window_indices(i: int, w: int, n: int) -> List[int]:
    if w <= 1:
        return [i]
    idxs = {i}
    for k in range(1, w):
        if i-k >= 0: idxs.add(i-k)
        if i+k < n: idxs.add(i+k)
    return sorted(idxs)

# =============================
# 실행
# =============================
if run:
    if not text.strip():
        st.warning("본문을 입력해주세요.")
        st.stop()

    sentences = split_sentences(text)
    st.write(f"문장 수: {len(sentences)}")

    # 문장별 등장 인물 집합
    per_by_sent: List[Set[str]] = []
    for s in sentences:
        found = set(normalize_name(m.group(1)) for m in NAME_RX.finditer(s))
        found = {n for n in found if n in NAMES}
        per_by_sent.append(found)

    # 그래프 생성
    G = nx.Graph()
    freq: Dict[str, int] = {n: 0 for n in NAMES}

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

    # 엣지 필터
    to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_edge]
    G.remove_edges_from(to_remove)

    # =============================
    # 요약 테이블 생성
    # =============================
    nodes_df = pd.DataFrame([
        {"name": nname,
         "degree": G.degree(nname),
         "frequency": freq.get(nname, 0)}
        for nname in G.nodes()
    ])
    if not nodes_df.empty:
        nodes_df = nodes_df.sort_values(["degree", "frequency"], ascending=[False, False])

    edges_data = [
        {"source": u, "target": v, "weight": d.get("weight", 0)}
        for u, v, d in G.edges(data=True)
    ]
    edges_df = pd.DataFrame(edges_data)
    if not edges_df.empty:
        edges_df = edges_df.sort_values("weight", ascending=False)

    # =============================
    # Streamlit 출력
    # =============================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("노드 요약")
        if nodes_df.empty:
            st.info("⚠️ 노드가 없습니다. 인물이 등장하지 않았습니다.")
        else:
            st.dataframe(nodes_df, use_container_width=True, height=260)
            st.download_button(
                "노드 CSV 다운로드",
                nodes_df.to_csv(index=False).encode("utf-8"),
                file_name="nodes.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("엣지 요약")
        if edges_df.empty:
            st.info("⚠️ 엣지가 없습니다. 인물들이 같은 문장에 함께 등장하지 않았습니다.")
        else:
            st.dataframe(edges_df, use_container_width=True, height=260)
            st.download_button(
                "엣지 CSV 다운로드",
                edges_df.to_csv(index=False).encode("utf-8"),
                file_name="edges.csv",
                mime="text/csv"
            )

    # =============================
    # 네트워크 시각화 (Pyvis)
    # =============================
    st.subheader("관계 네트워크 (드래그/줌 가능)")
    net = Network(height="620px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=160, spring_strength=0.005, damping=0.6)

    # 노드 추가
    fmax = max(1, max(freq.values()))
    for n in NAMES:
        if n in G.nodes:
            size = 20 + 30 * (freq[n] / fmax)
            label = f"{n} (deg={G.degree(n)}, f={freq[n]})"
            net.add_node(n, label=label, title=label, size=float(size))

    # 엣지 추가
    for u, v, d in G.edges(data=True):
        val = int(d.get("weight", 1))
        title = f"{u}–{v} (w={val})"
        net.add_edge(u, v, value=val, title=title, width=1+val)

    html_path = "graph.html"
    net.show(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=640, scrolling=True)

    st.success("완료!")

st.markdown("---")
st.caption("고정 인물: 차미 · 오란 · 녹주 | 문장 동시출현 기반 관계망")
