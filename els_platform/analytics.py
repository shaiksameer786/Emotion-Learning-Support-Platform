"""Demonstration-only Plotly analytics fixtures."""

from __future__ import annotations

import plotly.graph_objects as go

from .prediction.schema import EMOTION_LABELS


DEMONSTRATION_COUNTS = {
    "Bored": 4,
    "Confident": 6,
    "Confused": 8,
    "Curious": 5,
    "Frustrated": 3,
}


def make_demonstration_figure() -> go.Figure:
    labels = [label.value for label in EMOTION_LABELS]
    values = [DEMONSTRATION_COUNTS[label] for label in labels]
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#64748b", "#16a34a", "#f59e0b", "#0ea5e9", "#ef4444"],
                hovertemplate="%{x}<br>Illustrative count: %{y}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Demonstration counts only - not learner analytics",
        xaxis_title="Target category",
        yaxis_title="Illustrative count",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 65, "b": 20},
    )
    return figure
