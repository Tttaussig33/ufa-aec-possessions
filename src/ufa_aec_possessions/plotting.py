from __future__ import annotations

from html import escape

import pandas as pd

from ufa_aec_possessions.possessions import (
    ENDZONE_HIGH_Y,
    ENDZONE_LOW_Y,
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    _path_points,
)


def _format_number(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def _path_hover_text(path: pd.DataFrame) -> list[str]:
    thrower = path.get("Thrower", path.get("thrower", pd.Series("", index=path.index)))
    receiver = path.get("Receiver", path.get("receiver", pd.Series("", index=path.index)))
    aec = pd.to_numeric(path.get("aec", pd.Series(index=path.index, dtype=float)), errors="coerce")
    cp = pd.to_numeric(path.get("cp", pd.Series(index=path.index, dtype=float)), errors="coerce")
    yards = pd.to_numeric(
        path.get("throw_distance", pd.Series(index=path.index, dtype=float)),
        errors="coerce",
    )

    text = []
    for throw_number, (_, row) in enumerate(path.iterrows(), start=1):
        parts = [f"Throw {throw_number}"]
        if str(thrower.loc[row.name]).strip() or str(receiver.loc[row.name]).strip():
            parts.append(f"{thrower.loc[row.name]} -> {receiver.loc[row.name]}")
        if pd.notna(aec.loc[row.name]):
            parts.append(f"aEC: {aec.loc[row.name]:.3f}")
        if pd.notna(cp.loc[row.name]):
            parts.append(f"CP: {cp.loc[row.name]:.1%}")
        if pd.notna(yards.loc[row.name]):
            parts.append(f"Distance: {yards.loc[row.name]:.1f}")
        text.append("<br>".join(parts))
    return text


def _add_field_shapes(fig, row: int | None = None, col: int | None = None):
    shape_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    fig.add_shape(
        type="rect",
        x0=FIELD_X_MIN,
        x1=FIELD_X_MAX,
        y0=FIELD_Y_MIN,
        y1=FIELD_Y_MAX,
        line={"color": "#1B1E26", "width": 2},
        fillcolor="#86d973",
        layer="below",
        **shape_kwargs,
    )
    for y_value in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        fig.add_shape(
            type="line",
            x0=FIELD_X_MIN,
            x1=FIELD_X_MAX,
            y0=y_value,
            y1=y_value,
            line={"color": "#1B1E26", "width": 1},
            layer="below",
            **shape_kwargs,
        )
    for x_value in [-8.88, 8.88]:
        fig.add_shape(
            type="line",
            x0=x_value,
            x1=x_value,
            y0=ENDZONE_LOW_Y,
            y1=ENDZONE_HIGH_Y,
            line={"color": "#b9c8bb", "width": 1, "dash": "dot"},
            layer="below",
            **shape_kwargs,
        )


def _add_path_arrows(
    fig,
    points: pd.DataFrame,
    color: str,
    every: int = 1,
    opacity: float = 0.85,
    row: int | None = None,
    col: int | None = None,
):
    for index, (start, end) in enumerate(zip(points.iloc[:-1].itertuples(), points.iloc[1:].itertuples())):
        if index % every != 0:
            continue
        annotation = {
            "x": end.x,
            "y": end.y,
            "ax": start.x,
            "ay": start.y,
            "showarrow": True,
            "arrowhead": 3,
            "arrowsize": 1,
            "arrowwidth": 1.8,
            "arrowcolor": color,
            "opacity": opacity,
        }
        if row is not None and col is not None:
            fig.add_annotation(**annotation, row=row, col=col)
        else:
            fig.add_annotation(
                **annotation,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
            )


def _apply_field_layout(fig, title: str, width: int = 540, height: int = 760):
    fig.update_xaxes(
        range=[FIELD_X_MIN - 5, FIELD_X_MAX + 5],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        range=[FIELD_Y_MIN - 3, FIELD_Y_MAX + 3],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor="#f6faf5",
        paper_bgcolor="#ffffff",
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        legend={"orientation": "h", "y": -0.04},
    )
    return fig


def plot_possession_path(path: pd.DataFrame, title: str = "Scoring possession path", color: str = "#b74126"):
    """Plot one real possession, preserving its actual zig-zag shape."""
    import plotly.graph_objects as go

    points = _path_points(path)
    fig = go.Figure()
    _add_field_shapes(fig)
    fig.add_trace(
        go.Scatter(
            x=points["x"],
            y=points["y"],
            mode="lines+markers",
            line={"color": color, "width": 4},
            marker={"size": 8, "color": color},
            text=["Start"] + _path_hover_text(path),
            hovertemplate="%{text}<extra></extra>",
            name="Real possession",
        )
    )
    _add_path_arrows(fig, points, color)
    return _apply_field_layout(fig, title)


def plot_representative_paths(
    representative_paths: dict[str, pd.DataFrame],
    title: str = "Representative scoring path styles",
    show_arrows: bool = False,
):
    """Overlay multiple real possession paths on one field."""
    import plotly.graph_objects as go

    colors = ["#b74126", "#164e87", "#7a3db8", "#2f7d32", "#d97706", "#0f766e"]
    fig = go.Figure()
    _add_field_shapes(fig)
    for index, (label, path) in enumerate(representative_paths.items()):
        color = colors[index % len(colors)]
        points = _path_points(path)
        fig.add_trace(
            go.Scatter(
                x=points["x"],
                y=points["y"],
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 7, "color": color},
                text=["Start"] + _path_hover_text(path),
                hovertemplate=f"{escape(str(label))}<br>%{{text}}<extra></extra>",
                name=str(label),
            )
        )
        if show_arrows:
            _add_path_arrows(fig, points, color, every=2, opacity=0.65)
    return _apply_field_layout(fig, title, width=680)


def plot_side_by_side_paths(
    left_paths: dict[str, pd.DataFrame],
    right_paths: dict[str, pd.DataFrame],
    *,
    left_title: str = "aEC per throw",
    right_title: str = "Total aEC",
    title: str = "Top scoring possessions",
    show_arrows: bool = False,
):
    """Plot two possession-path overlays side by side for metric comparison."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = ["#b74126", "#164e87", "#7a3db8", "#2f7d32", "#d97706", "#0f766e"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[left_title, right_title],
        horizontal_spacing=0.08,
    )

    for col, path_group in enumerate([left_paths, right_paths], start=1):
        _add_field_shapes(fig, row=1, col=col)
        for index, (label, path) in enumerate(path_group.items()):
            color = colors[index % len(colors)]
            points = _path_points(path)
            fig.add_trace(
                go.Scatter(
                    x=points["x"],
                    y=points["y"],
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                    marker={"size": 7, "color": color},
                    text=["Start"] + _path_hover_text(path),
                    hovertemplate=f"{escape(str(label))}<br>%{{text}}<extra></extra>",
                    name=str(label),
                    legendgroup=f"side-{col}-{index}",
                    showlegend=True,
                ),
                row=1,
                col=col,
            )
            if show_arrows:
                _add_path_arrows(
                    fig,
                    points,
                    color,
                    every=2,
                    opacity=0.65,
                    row=1,
                    col=col,
                )

        fig.update_xaxes(
            range=[FIELD_X_MIN - 5, FIELD_X_MAX + 5],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor=f"y{col if col > 1 else ''}",
            scaleratio=1,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            range=[FIELD_Y_MIN - 3, FIELD_Y_MAX + 3],
            showgrid=False,
            zeroline=False,
            visible=False,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=title,
        width=1180,
        height=760,
        plot_bgcolor="#f6faf5",
        paper_bgcolor="#ffffff",
        margin={"l": 20, "r": 20, "t": 70, "b": 10},
        legend={"orientation": "h", "y": -0.04},
    )
    return fig


def render_shownspace_possession_svg(path: pd.DataFrame, width: int = 260, height: int = 560) -> str:
    """Return a compact Shown Space-style SVG field for one possession."""
    path = path.sort_values("possession_throw").copy()
    field_width = width - 34
    field_height = height - 34
    left = (width - field_width) / 2
    top = 17

    def sx(value):
        scale = (float(value) - FIELD_X_MIN) / (FIELD_X_MAX - FIELD_X_MIN)
        return left + scale * field_width

    def sy(value):
        scale = (FIELD_Y_MAX - float(value)) / (FIELD_Y_MAX - FIELD_Y_MIN)
        return top + scale * field_height

    shapes = [
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{field_width:.2f}" height="{field_height:.2f}" rx="4" fill="#86d973" stroke="#1B1E26" stroke-width="2" />'
    ]
    for y_value in [ENDZONE_LOW_Y, ENDZONE_HIGH_Y]:
        shapes.append(
            f'<line x1="{left:.2f}" y1="{sy(y_value):.2f}" x2="{left + field_width:.2f}" y2="{sy(y_value):.2f}" stroke="#1B1E26" stroke-width="1" />'
        )
    for x_value in [-8.88, 8.88]:
        shapes.append(
            f'<line x1="{sx(x_value):.2f}" y1="{sy(ENDZONE_LOW_Y):.2f}" x2="{sx(x_value):.2f}" y2="{sy(ENDZONE_HIGH_Y):.2f}" stroke="#b9c8bb" stroke-width="1" stroke-dasharray="4 4" />'
        )
    for y_value in [40, 80]:
        shapes.append(f'<circle cx="{sx(0):.2f}" cy="{sy(y_value):.2f}" r="2.5" fill="#1B1E26" />')

    throw_shapes = []
    for index, (_, throw) in enumerate(path.iterrows(), start=1):
        x1 = sx(throw["ThrowerX"])
        y1 = sy(throw["ThrowerY"])
        x2 = sx(throw["ReceiverX"])
        y2 = sy(throw["ReceiverY"])
        title = escape(
            f"Throw {index}: {throw.get('Thrower', '')} -> {throw.get('Receiver', '')} | "
            f"aEC {_format_number(throw.get('aec'), 3)} | "
            f"CP {_format_percent(throw.get('cp'))} | "
            f"Distance {_format_number(throw.get('throw_distance'), 1)}"
        )
        throw_shapes.append(
            f'<g><title>{title}</title><line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#b74126" stroke-width="3" stroke-linecap="round" />'
            f'<circle cx="{x2:.2f}" cy="{y2:.2f}" r="4" fill="#b74126" stroke="#ffffff" stroke-width="1" /></g>'
        )
    if not path.empty:
        throw_shapes.insert(
            0,
            f'<circle cx="{sx(path["ThrowerX"].iloc[0]):.2f}" cy="{sy(path["ThrowerY"].iloc[0]):.2f}" r="4" fill="#1B1E26"><title>Start</title></circle>',
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f'<title>{escape(str(path["possession_id"].iloc[0])) if "possession_id" in path and not path.empty else "Possession path"}</title>'
        f'{"".join(shapes)}{"".join(throw_shapes)}</svg>'
    )
