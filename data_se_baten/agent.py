"""Step 3 of the workflow: the OpenAI LLM that chats with the data.

A small LangGraph agent:

    START -> agent (OpenAI LLM) --tool call?--> tools (run pandas code) -> agent -> ... -> END

The LLM gets the data profile in its system prompt and can call one tool,
`python_analysis`, which executes pandas / Plotly code against the DataFrame `df`.
Tables and charts produced by the tool are passed back to the UI as artifacts
(they are never sent to the model, only the text output is).
"""

from __future__ import annotations

import ast
import io
from typing import Annotated, Iterator, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MAX_TOOL_OUTPUT_CHARS = 6000

SYSTEM_PROMPT = """You are **Data Se Baten**, an expert data analyst AI agent. \
The user has uploaded a dataset and you help them understand it by answering questions about it.

A pandas DataFrame named `df` holds the full dataset. Its automatic profile is:

<data_profile>
{profile}
</data_profile>

How to work:
- For anything that needs numbers, filtering, grouping, ranking or statistics, call the \
`python_analysis` tool. Never guess or invent numbers — compute them.
- For simple questions that the profile already answers (e.g. column names, row count), you may answer directly.
- When a chart would help, or the user asks for one, build it with Plotly (`px` or `go`) in the tool \
and assign it to a variable named `fig`. The chart is shown to the user automatically; do not describe \
the code, describe the insight.
- If code fails, read the error, fix the code and try again.
- Final answers: clear, concise Markdown. Lead with the direct answer, then key numbers \
(format large numbers with thousands separators), then a short insight or next step when useful. \
Mention assumptions you made (e.g. how missing values were treated).
- Reply in the same language the user writes in (English, Urdu or Roman Urdu).
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------------------------------- #
# Tool: sandbox-ish pandas execution
# --------------------------------------------------------------------------- #
def _format_value(value) -> str:
    if isinstance(value, pd.DataFrame):
        extra = f"\n... ({len(value):,} rows total)" if len(value) > 50 else ""
        return value.head(50).to_markdown() + extra
    if isinstance(value, pd.Series):
        extra = f"\n... ({len(value):,} rows total)" if len(value) > 50 else ""
        return value.head(50).to_frame().to_markdown() + extra
    if isinstance(value, go.Figure):
        return "[Plotly chart created and shown to the user]"
    return repr(value)


def execute_python(code: str, df: pd.DataFrame) -> tuple[str, dict]:
    """Run `code` with `df` in scope. Returns (text for the LLM, artifact for the UI)."""
    output = io.StringIO()

    def _print(*args, **kwargs):  # per-call print, so concurrent sessions don't share stdout
        kwargs.pop("file", None)
        print(*args, file=output, **kwargs)

    namespace = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go, "print": _print}
    artifact: dict = {"code": code, "figures": [], "tables": [], "error": None}

    try:
        tree = ast.parse(code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<analysis>", "exec"), namespace)
        result = eval(compile(last_expr, "<analysis>", "eval"), namespace) if last_expr else namespace.get("result")
    except Exception as exc:  # the error goes back to the LLM so it can fix its code
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        return f"ERROR: {artifact['error']}\nOutput before error:\n{output.getvalue()[:2000]}", artifact

    figures = [v for k, v in namespace.items() if isinstance(v, go.Figure) and not k.startswith("_")]
    if isinstance(result, go.Figure) and not any(result is f for f in figures):
        figures.append(result)
    artifact["figures"] = figures
    if isinstance(result, (pd.DataFrame, pd.Series)):
        artifact["tables"].append(result if isinstance(result, pd.DataFrame) else result.to_frame())

    parts = []
    if output.getvalue():
        parts.append("stdout:\n" + output.getvalue())
    if result is not None:
        parts.append("result:\n" + _format_value(result))
    if figures:
        parts.append(f"{len(figures)} chart(s) shown to the user.")
    text = "\n\n".join(parts) or "Code ran successfully but produced no output. Print or return a value."
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n... [output truncated]"
    return text, artifact


def make_tools(df: pd.DataFrame) -> list:
    @tool(response_format="content_and_artifact")
    def python_analysis(code: str) -> tuple[str, dict]:
        """Execute Python code to analyse the pandas DataFrame `df`.

        Available names: df (the dataset), pd (pandas), np (numpy), px (plotly.express), go (plotly.graph_objects).
        The value of the last expression is returned (or assign it to `result`); use print() for extra output.
        To show a chart, create a Plotly figure and assign it to `fig`.
        """
        return execute_python(code, df)

    return [python_analysis]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_agent(df: pd.DataFrame, profile_text: str, api_key: str, model: str):
    llm = ChatOpenAI(model=model, api_key=api_key, max_retries=2, timeout=120)
    tools = make_tools(df)
    llm_with_tools = llm.bind_tools(tools)
    system = SystemMessage(SYSTEM_PROMPT.format(profile=profile_text))

    def call_model(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke([system, *state["messages"]])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(agent, history: list[dict], question: str, max_steps: int = 24) -> Iterator[dict]:
    """Stream the agent's work as simple events for the UI.

    `history` is the chat so far as [{"role": "user"|"assistant", "content": str}].
    Only questions and final answers are replayed to the model (compact memory).

    Yields dicts with type: "tool_call" | "tool_result" | "answer".
    """
    messages: list[AnyMessage] = []
    for turn in history[-12:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    messages.append(HumanMessage(content=question))

    final_text = ""
    for update in agent.stream({"messages": messages}, {"recursion_limit": max_steps}, stream_mode="updates"):
        for node, payload in update.items():
            for msg in payload.get("messages", []):
                if node == "agent" and isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        yield {"type": "tool_call", "code": call["args"].get("code", "")}
                    if not msg.tool_calls:
                        final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                elif isinstance(msg, ToolMessage):
                    yield {"type": "tool_result", "content": msg.content, "artifact": msg.artifact or {}}
    yield {"type": "answer", "content": final_text or "_I could not produce an answer. Please rephrase the question._"}
