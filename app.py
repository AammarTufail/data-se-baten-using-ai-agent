"""Data Se Baten — an AI agent that lets you chat with your data.

Workflow:  Data  ->  Data Profiling  ->  OpenAI LLM (LangGraph agent)  ->  Output & Answers

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_se_baten import APP_NAME, __version__
from data_se_baten.profiling import SUPPORTED_TYPES, clean_columns, load_data, profile_data, profile_to_text

ROOT = Path(__file__).parent
SAMPLE_FILE = ROOT / "sample_data" / "pk_ecommerce_sales_2025.csv"
MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"]
PALETTE = ["#4F46E5", "#0EA5E9", "#14B8A6", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B"]
USER_AVATAR, BOT_AVATAR = ":material/person:", ":material/smart_toy:"
VIEWS = {
    "chat": ":material/forum: Chat with data",
    "profile": ":material/analytics: Data profile",
    "preview": ":material/table_view: Data preview",
}

st.set_page_config(
    page_title=f"{APP_NAME} · Chat with your data",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="auto",  # expanded on desktop, collapsed on phones
)

# Custom HTML is used only for pieces with no native equivalent: the brand block,
# hero banner, workflow stepper and landing cards. Everything else is themed via
# .streamlit/config.toml.
st.html(
    """
<style>
.block-container { padding-top: 3.5rem; padding-bottom: 5rem; max-width: 1280px; }

.dsb-brand { display:flex; align-items:center; gap:.75rem; margin:.25rem 0 .5rem; }
.dsb-logo { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:1.35rem;
            background: linear-gradient(135deg,#6366F1,#06B6D4); box-shadow:0 6px 18px rgba(99,102,241,.45); }
.dsb-brand h2 { margin:0; padding:0; font-size:1.15rem; font-weight:800; color:#fff; letter-spacing:-.01em; }
.dsb-brand p { margin:0; font-size:.75rem; color:#A5B4FC; }
.dsb-side-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
                  color:#A5B4FC; margin:.5rem 0 -.25rem; }

.dsb-hero { border-radius:20px; padding:1.6rem 2rem; color:#fff; margin-bottom:.25rem;
            background: radial-gradient(1200px 300px at 10% -20%, rgba(255,255,255,.18), transparent),
                        linear-gradient(120deg,#4338CA 0%,#6366F1 45%,#0891B2 100%);
            box-shadow:0 18px 40px -18px rgba(67,56,202,.6); }
.dsb-hero h1 { margin:0; padding:0; font-size:2rem; font-weight:800; letter-spacing:-.02em; color:#fff; }
.dsb-hero p { margin:.35rem 0 0; font-size:1rem; opacity:.92; max-width:760px; }
.dsb-chip { display:inline-block; margin:.9rem .4rem 0 0; padding:.22rem .7rem; border-radius:999px;
            font-size:.75rem; font-weight:600; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25); }

.dsb-steps { display:flex; gap:.5rem; align-items:stretch; flex-wrap:wrap; }
.dsb-step { flex:1 1 170px; display:flex; align-items:center; gap:.7rem; padding:.75rem 1rem; border-radius:14px;
            background:#fff; border:1px solid #E2E8F0; }
.dsb-step .n { width:30px; height:30px; flex:none; border-radius:50%; display:grid; place-items:center;
               font-weight:700; font-size:.85rem; background:#F1F5F9; color:#64748B; }
.dsb-step .t { font-weight:600; font-size:.88rem; color:#0F172A; line-height:1.2; }
.dsb-step .s { font-size:.74rem; color:#64748B; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px; }
.dsb-step.done { border-color:#A7F3D0; background:#F0FDF4; }
.dsb-step.done .n { background:#10B981; color:#fff; }
.dsb-step.active { border-color:#A5B4FC; background:#EEF2FF; box-shadow:0 0 0 3px rgba(99,102,241,.12); }
.dsb-step.active .n { background:#4F46E5; color:#fff; }
.dsb-arrow { align-self:center; color:#94A3B8; }
@media (max-width: 900px) { .dsb-arrow { display:none; } }

.dsb-card { height:100%; padding:1.25rem; border-radius:16px; background:#fff; border:1px solid #E2E8F0; }
.dsb-card .i { font-size:1.5rem; }
.dsb-card h4 { margin:.5rem 0 .25rem; padding:0; font-size:1rem; font-weight:700; color:#0F172A; }
.dsb-card p { margin:0; font-size:.88rem; color:#475569; }

.dsb-alert { padding:.5rem .8rem; border-radius:10px; margin-bottom:.4rem; font-size:.86rem; border-left:4px solid; }
.dsb-alert.warning { background:#FFFBEB; border-color:#F59E0B; color:#78350F; }
.dsb-alert.info { background:#EFF6FF; border-color:#3B82F6; color:#1E3A8A; }
</style>
"""
)


# --------------------------------------------------------------------------- #
# State & helpers
# --------------------------------------------------------------------------- #
for _k, _v in {"file_name": None, "file_hash": None, "raw": None, "messages": [], "agent": None, "agent_key": None}.items():
    st.session_state.setdefault(_k, _v)


def secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml present
        pass
    return os.environ.get(name)


@st.cache_data(show_spinner=False, max_entries=8)
def load_and_profile(file_name: str, raw: bytes):
    df = clean_columns(load_data(file_name, raw))
    profile = profile_data(df)
    return df, profile, profile_to_text(df, profile)


def set_dataset(file_name: str, raw: bytes) -> None:
    file_hash = hashlib.md5(raw).hexdigest()
    if file_hash == st.session_state.file_hash:
        return
    st.session_state.update(file_name=file_name, file_hash=file_hash, raw=raw, messages=[], agent=None, agent_key=None)
    st.session_state.pop("suggestion", None)


def get_agent(api_key: str, model: str, df: pd.DataFrame, profile_text: str):
    from data_se_baten.agent import build_agent  # heavy import, only needed once chatting starts

    key = (st.session_state.file_hash, model, hashlib.sha256(api_key.encode()).hexdigest())
    if st.session_state.agent_key != key:
        st.session_state.agent = build_agent(df, profile_text, api_key, model)
        st.session_state.agent_key = key
    return st.session_state.agent


def suggested_questions(p) -> list[str]:
    """Build starter questions from the profile, skipping ID-like and free-text columns."""
    info = p.columns.set_index("column")
    groups = [c for c in p.categorical_cols if 2 <= info.loc[c, "unique"] <= 30]
    measures = [c for c in p.numeric_cols if info.loc[c, "unique"] > 2]
    keywords = ("revenue", "sales", "amount", "total", "profit", "price", "value", "income", "cost")
    measures.sort(key=lambda c: next((i for i, k in enumerate(keywords) if k in c.lower()), len(keywords)))

    qs = ["Give me an executive summary of this dataset"]
    if measures and groups:
        qs.append(f"Show total {measures[0]} by {groups[0]} as a bar chart")
    if measures and p.datetime_cols:
        qs.append(f"How does {measures[0]} trend by month?")
    if len(groups) > 1:
        qs.append(f"What are the top 5 {groups[1]} values by count?")
    if len(measures) > 1:
        qs.append("Which numeric columns are most strongly correlated?")
    qs.append("Which columns have data quality issues and how do I fix them?")
    return qs[:5]


def chat_to_markdown(messages: list[dict]) -> str:
    out = [f"# {APP_NAME} — chat export", f"_Dataset: {st.session_state.file_name} · {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in messages:
        out.append(f"## {'You' if m['role'] == 'user' else APP_NAME}\n\n{m['content']}\n")
        for step in m.get("steps", []):
            out.append(f"```python\n{step['code']}\n```\n")
    return "\n".join(out)


def profile_to_markdown(p) -> str:
    out = [
        f"# Data profile — {st.session_state.file_name}",
        "",
        f"- Rows: {p.n_rows:,}\n- Columns: {p.n_cols}\n- Missing cells: {p.n_missing:,} ({p.missing_pct}%)",
        f"- Duplicate rows: {p.n_duplicates:,}\n- Quality score: {p.quality_score}/100",
        "",
        "## Columns",
        p.columns.to_markdown(index=False),
    ]
    if not p.numeric_stats.empty:
        out += ["", "## Numeric statistics", p.numeric_stats.to_markdown()]
    if p.alerts:
        out += ["", "## Alerts"] + [f"- **{a['column']}** — {a['message']}" for a in p.alerts]
    return "\n".join(out)


def friendly_error(exc: Exception, model: str) -> str:
    name, text = type(exc).__name__, str(exc)
    if name == "AuthenticationError":
        return ":material/key_off: **Your OpenAI API key was rejected.** Check the key in the sidebar and try again."
    if name == "RateLimitError":
        return ":material/hourglass_top: **OpenAI rate limit or quota reached.** Wait a moment, or check billing at platform.openai.com."
    if name in ("NotFoundError", "BadRequestError", "PermissionDeniedError") and "model" in text.lower():
        return f":material/smart_toy: **The model `{model}` isn't available for your key.** Pick another model in the sidebar."
    if name == "GraphRecursionError":
        return ":material/replay: **The analysis took too many steps.** Try a more specific question."
    if name in ("APIConnectionError", "APITimeoutError"):
        return ":material/wifi_off: **Could not reach OpenAI.** Check your internet connection and try again."
    return f":material/error: **Something went wrong:** `{name}: {text[:300]}`"


# --------------------------------------------------------------------------- #
# Sidebar: connect + upload
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.html(
        f"""<div class="dsb-brand"><div class="dsb-logo">💬</div>
        <div><h2>{APP_NAME}</h2><p>Chat with your data · v{__version__}</p></div></div>"""
    )

    st.html('<div class="dsb-side-label">① Connect OpenAI</div>')
    secret_key = secret("OPENAI_API_KEY")
    if secret_key:
        api_key = secret_key
        st.success("API key loaded from secrets", icon=":material/lock:")
    else:
        api_key = st.text_input(
            "OpenAI API key", type="password", placeholder="sk-...",
            help="Create one at platform.openai.com/api-keys. It is kept only in this browser session.",
        ).strip()
    default_model = secret("OPENAI_MODEL") or MODELS[0]
    model = st.selectbox(
        "Model", MODELS, index=MODELS.index(default_model) if default_model in MODELS else 0,
        accept_new_options=True, help="Pick a model or type any OpenAI chat model name that supports tool calling.",
    )

    st.html('<div class="dsb-side-label">② Upload data</div>')
    uploaded = st.file_uploader(
        "Dataset", type=SUPPORTED_TYPES, label_visibility="collapsed",
        help="CSV, TSV, Excel, JSON or Parquet — up to 200 MB.",
    )
    if uploaded is not None:
        set_dataset(uploaded.name, uploaded.getvalue())
    if st.button("Try the sample dataset", icon=":material/auto_awesome:", width="stretch",
                 disabled=not SAMPLE_FILE.exists()):
        set_dataset(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes())

    if st.session_state.file_hash:
        st.divider()
        with st.container(horizontal=True):
            if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch",
                         disabled=not st.session_state.messages):
                st.session_state.messages = []
                st.session_state.pop("suggestion", None)
                st.rerun()
            st.download_button(
                "Export", chat_to_markdown(st.session_state.messages), file_name="data_se_baten_chat.md",
                mime="text/markdown", icon=":material/download:", width="stretch",
                disabled=not st.session_state.messages,
            )

    st.divider()
    st.caption(
        ":material/shield: Your file stays in memory for this session only. "
        "Your questions and a summary of the data are sent to OpenAI."
    )


# --------------------------------------------------------------------------- #
# Workflow steps 1 & 2: load + profile
# --------------------------------------------------------------------------- #
df = profile = profile_text = None
load_error = None
if st.session_state.file_hash:
    try:
        with st.spinner("Profiling your data…"):
            df, profile, profile_text = load_and_profile(st.session_state.file_name, st.session_state.raw)
    except Exception as exc:
        load_error = str(exc)


# --------------------------------------------------------------------------- #
# Header + workflow stepper
# --------------------------------------------------------------------------- #
st.html(
    f"""<div class="dsb-hero">
      <h1>💬 {APP_NAME}</h1>
      <p>Upload a dataset, get an instant profile, then ask questions in English or Urdu.
      The AI agent writes and runs the analysis for you and answers with real numbers and charts.</p>
      <span class="dsb-chip">OpenAI</span><span class="dsb-chip">LangGraph agent</span>
      <span class="dsb-chip">pandas</span><span class="dsb-chip">Plotly</span>
    </div>"""
)

has_data, has_key = df is not None, bool(api_key)
n_questions = sum(m["role"] == "user" for m in st.session_state.messages)
steps = [
    ("Data", st.session_state.file_name or "Upload a file", has_data),
    ("Data profiling", f"{profile.n_rows:,} rows · {profile.n_cols} columns" if has_data else "Automatic", has_data),
    ("OpenAI LLM", model if has_key else "Add your API key", has_key and has_data),
    ("Output & answers", f"{n_questions} question{'s' if n_questions != 1 else ''} answered" if n_questions else "Ask anything",
     n_questions > 0),
]
first_pending = next((i for i, s in enumerate(steps) if not s[2]), None)
parts = []
for i, (title, sub, done) in enumerate(steps):
    cls = "done" if done else ("active" if i == first_pending else "")
    parts.append(f'<div class="dsb-step {cls}"><div class="n">{"✓" if done else i + 1}</div>'
                 f'<div><div class="t">{title}</div><div class="s">{sub}</div></div></div>')
    if i < len(steps) - 1:
        parts.append('<div class="dsb-arrow">→</div>')
st.html(f'<div class="dsb-steps">{"".join(parts)}</div>')


# --------------------------------------------------------------------------- #
# Landing page (no data yet)
# --------------------------------------------------------------------------- #
if load_error:
    st.error(f"Could not read **{st.session_state.file_name}**: {load_error}", icon=":material/error:")
    st.stop()

if df is None:
    cards = [
        ("📂", "Bring your data", "CSV, Excel, JSON or Parquet up to 200 MB — or click <b>Try the sample dataset</b>."),
        ("🔎", "Instant profiling", "Types, missing values, statistics, distributions, correlations and quality alerts."),
        ("🧠", "AI agent analyses", "An OpenAI model, orchestrated by LangGraph, writes and runs pandas code on your data."),
        ("💡", "Answers & charts", "Clear answers with real numbers, interactive charts, and the code behind them."),
    ]
    for col, (icon, title, text) in zip(st.columns(4), cards):
        col.html(f'<div class="dsb-card"><div class="i">{icon}</div><h4>{title}</h4><p>{text}</p></div>')
    st.info("**Get started:** add your OpenAI API key and upload a dataset from the sidebar.", icon=":material/arrow_back:")
    st.stop()


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def render_profile() -> None:
    p = profile
    with st.container(horizontal=True):
        st.metric("Rows", f"{p.n_rows:,}", border=True)
        st.metric("Columns", p.n_cols, border=True)
        st.metric("Missing cells", f"{p.missing_pct}%", help=f"{p.n_missing:,} empty cells", border=True)
        st.metric("Duplicate rows", f"{p.n_duplicates:,}", border=True)
        st.metric("Memory", f"{p.memory_mb:.2f} MB", border=True)
        st.metric("Quality score", f"{p.quality_score}/100", border=True,
                  help="100 − missing % − duplicate % − a penalty per quality alert")

    left, right = st.columns([1, 1.6], gap="medium")
    with left.container(border=True, height="stretch"):
        st.markdown("#### :material/donut_large: Column types")
        kinds = p.columns["type"].value_counts().reset_index()
        fig = px.pie(kinds, names="type", values="count", hole=0.62, color_discrete_sequence=PALETTE)
        fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    with right.container(border=True, height="stretch"):
        st.markdown(f"#### :material/health_and_safety: Data quality alerts · {len(p.alerts)}")
        if not p.alerts:
            st.success("No issues detected — your data looks clean.", icon=":material/check_circle:")
        with st.container(height=250, border=False):
            for a in p.alerts:
                icon = "⚠️" if a["level"] == "warning" else "ℹ️"
                where = "" if a["column"] == "—" else f'<b>{a["column"]}</b> — '
                st.html(f'<div class="dsb-alert {a["level"]}">{icon} {where}{a["message"]}</div>')

    tabs = st.tabs([":material/view_column: Columns", ":material/functions: Statistics",
                    ":material/format_color_reset: Missing values", ":material/bar_chart: Distributions",
                    ":material/hub: Correlations"])
    with tabs[0]:
        st.dataframe(
            p.columns, hide_index=True, width="stretch",
            column_config={
                "missing_%": st.column_config.ProgressColumn("Missing %", min_value=0, max_value=100, format="%.1f%%"),
                "unique_%": st.column_config.ProgressColumn("Unique %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    with tabs[1]:
        if p.numeric_stats.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(p.numeric_stats, width="stretch")
        if p.categorical_cols:
            st.caption("Categorical columns")
            st.dataframe(df[p.categorical_cols].describe().T, width="stretch")
    with tabs[2]:
        miss = p.columns[p.columns["missing"] > 0].sort_values("missing_%")
        if miss.empty:
            st.success("No missing values in any column.", icon=":material/check_circle:")
        else:
            fig = px.bar(miss, x="missing_%", y="column", orientation="h", text="missing",
                         labels={"missing_%": "Missing (%)", "column": ""}, color_discrete_sequence=[PALETTE[3]])
            fig.update_layout(height=max(220, 36 * len(miss)), margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")
    with tabs[3]:
        col = st.selectbox("Column", df.columns, key="dist_col")
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            fig = px.histogram(df, x=col, nbins=40, marginal="box", color_discrete_sequence=PALETTE)
        elif pd.api.types.is_datetime64_any_dtype(s):
            counts = s.dt.to_period("M").value_counts().sort_index()
            fig = px.area(x=counts.index.astype(str), y=counts.values, labels={"x": "Month", "y": "Rows"}, color_discrete_sequence=[PALETTE[1]])
        else:
            counts = s.astype(str).value_counts().head(20).sort_values()
            fig = px.bar(x=counts.values, y=counts.index, orientation="h", labels={"x": "Count", "y": ""}, color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with tabs[4]:
        if p.correlations.empty:
            st.info("Need at least two numeric columns for correlations.")
        else:
            fig = px.imshow(p.correlations, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(height=max(360, 44 * len(p.correlations)), margin=dict(t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.download_button("Download profile report", profile_to_markdown(p), file_name="data_profile.md",
                       mime="text/markdown", icon=":material/download:")


def render_preview() -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"**{st.session_state.file_name}** · {len(df):,} rows × {df.shape[1]} columns")
        st.space("stretch")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="data.csv", mime="text/csv",
                           icon=":material/download:")
    date_only = {
        c: st.column_config.DatetimeColumn(format="YYYY-MM-DD")
        for c in profile.datetime_cols
        if (df[c].dropna().dt.normalize() == df[c].dropna()).all()
    }
    st.dataframe(df, width="stretch", height=560, column_config=date_only)


def render_assistant(msg: dict) -> None:
    """Replay a saved assistant message: collapsed agent steps, then the answer and charts."""
    steps = msg.get("steps", [])
    if steps:
        with st.expander(msg.get("thought_label", f"{len(steps)} analysis steps"), type="compact"):
            for i, step in enumerate(steps, 1):
                with st.expander(f"Analysis step {i}", type="step", icon=":material/error:" if step.get("error") else None):
                    st.code(step["code"], language="python")
                    if step.get("error"):
                        st.error(step["error"])
                    else:
                        st.code(step["output"][:2000], language="text")
    st.markdown(msg["content"])
    for i, fig in enumerate(msg.get("figures", [])):
        st.plotly_chart(fig, width="stretch", key=f"fig_{msg['id']}_{i}")


def answer_question(question: str) -> dict:
    """Workflow steps 3 & 4: run the LangGraph agent and render its work live."""
    from data_se_baten.agent import run_agent

    steps, figures, answer, pending = [], [], "", []
    started = time.time()
    try:
        agent = get_agent(api_key, model, df, profile_text)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.status(":shimmer[Thinking]", type="compact") as status:
            for event in run_agent(agent, history, question):
                if event["type"] == "tool_call":
                    box = st.status(f"Analysis step {len(steps) + len(pending) + 1}", type="step")
                    box.code(event["code"], language="python")
                    pending.append(box)
                elif event["type"] == "tool_result":
                    art = event["artifact"]
                    box = pending.pop(0) if pending else st.status("Analysis step", type="step")
                    if art.get("error"):
                        box.error(art["error"])
                        box.update(state="error")
                    else:
                        box.code(event["content"][:2000], language="text")
                        box.update(state="complete")
                    steps.append({"code": art.get("code", ""), "output": event["content"], "error": art.get("error")})
                    figures.extend(art.get("figures", []))
                elif event["type"] == "answer":
                    answer = event["content"]
            n = len(steps)
            label = f"Worked for {time.time() - started:.0f}s · {n} analysis step{'s' if n != 1 else ''}"
            status.update(label=label, state="complete")
    except Exception as exc:
        answer, label = friendly_error(exc, model), "Stopped"
    return {"id": len(st.session_state.messages), "role": "assistant", "content": answer,
            "steps": steps, "figures": figures, "thought_label": label}


def render_chat() -> None:
    if not api_key:
        st.warning("Add your **OpenAI API key** in the sidebar to start chatting with your data.", icon=":material/key:")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)

    suggestion = None
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(
                f"**Assalam-o-Alaikum!** I've profiled **{st.session_state.file_name}** — "
                f"{profile.n_rows:,} rows and {profile.n_cols} columns. Ask me anything, or start with one of these:"
            )
            suggestion = st.pills("Suggested questions", suggested_questions(profile), key="suggestion",
                                  label_visibility="collapsed", disabled=not api_key)

    question = st.chat_input("Ask a question about your data…", disabled=not api_key, submit_mode="disable")
    question = question or suggestion
    if not question or not api_key:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()  # show the question in the history immediately, then answer on the next run


def maybe_answer_pending() -> None:
    msgs = st.session_state.messages
    if api_key and msgs and msgs[-1]["role"] == "user":
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            reply = answer_question(msgs[-1]["content"])
            st.markdown(reply["content"])
            for i, fig in enumerate(reply["figures"]):
                st.plotly_chart(fig, width="stretch", key=f"live_fig_{i}")
        msgs.append(reply)
        st.rerun()


view = st.segmented_control("View", list(VIEWS), format_func=VIEWS.get, default="chat", required=True,
                            key="view", label_visibility="collapsed")

if view == "chat":
    render_chat()
    maybe_answer_pending()
elif view == "profile":
    render_profile()
else:
    render_preview()
