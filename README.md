# UFA AEC Possessions

A focused analysis repo for finding high-aEC UFA possessions and asking what those possessions look like on the field.

The first version uses Shown Space throw data as the source of truth for `aec`, builds possession-level rows, and renders possession-path diagrams for the highest and representative middle `aec_per_throw` possessions.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m pytest
```

Analyze a team from the command line:

```powershell
python scripts\analyze_team.py --team glory --season 2026 --top 5 --middle 5
```

Analyze every team and print each team's top five qualifying possessions:

```powershell
python scripts\analyze_team.py --all-teams --season 2026 --top 5
```

Optionally write one side-by-side comparison HTML file per team:

```powershell
python scripts\analyze_team.py --all-teams --season 2026 --top 5 --output-dir outputs\league_top5
```

Or open `notebooks/01_team_aec_possession_shapes.ipynb` and run the cells interactively.

For team-level aEC/T rankings, open `notebooks/02_team_aec_t_leaderboard.ipynb`.
League notebooks cache Shown Space fetches under `data/cache/` so rerunning analysis cells is faster after the first download.

## Default Analysis

The default filter is designed to match the earlier middle-aEC exploration:

- scoring possessions only
- O-line possessions only
- long-field possessions with `start_y <= 45` and `field_progress >= 50`
- huck-free possessions
- ranked by `aec_per_throw`

This gives two useful views:

- highest-aEC possessions, which may include extreme examples
- middle ranked possessions from the filtered set, which are often closer to normal efficient offense

## Package Surface

```python
from ufa_aec_possessions import (
    build_scoring_possessions,
    fetch_shownspace_season_throws,
    filter_analysis_possessions,
    plot_possession_path,
    render_shownspace_possession_svg,
    select_middle_aec_possessions,
    select_top_aec_possessions,
)
```
