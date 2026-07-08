from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ufa_aec_possessions import (  # noqa: E402
    build_aec_possession_sets,
    build_scoring_possessions,
    fetch_shownspace_season_throws,
    plot_representative_paths,
    select_top_aec_possessions_by_team,
)


def _label_paths(possessions, paths, prefix, metric):
    lookup = {path["possession_id"].iloc[0]: path for path in paths}
    labeled = {}
    for index, row in possessions.reset_index(drop=True).iterrows():
        possession_id = row["possession_id"]
        if possession_id in lookup:
            labeled[f"{prefix} {index + 1}: {row[metric]:.3f}"] = lookup[possession_id]
    return labeled


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze high-aEC UFA possessions.")
    parser.add_argument("--team", help="Shown Space team id, e.g. glory, empire, breeze")
    parser.add_argument("--all-teams", action="store_true", help="Analyze every team in the season")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--middle", type=int, default=5)
    parser.add_argument("--metric", default="aec_per_throw")
    parser.add_argument("--include-hucks", action="store_true", help="Keep huck possessions in the analysis set")
    parser.add_argument("--all-lines", action="store_true", help="Analyze O-line and D-line possessions together")
    parser.add_argument("--output-html", type=Path, default=None, help="Optional Plotly HTML path for an overlay figure")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for per-team HTML overlays with --all-teams")
    args = parser.parse_args()

    if not args.all_teams and not args.team:
        parser.error("provide --team or use --all-teams")

    games, throws = fetch_shownspace_season_throws(
        season=args.season,
        team_id=None if args.all_teams else args.team,
        max_games=args.max_games,
    )
    possessions, paths = build_scoring_possessions(
        throws,
        team_id=None if args.all_teams else args.team,
    )

    columns = [
        "team_id",
        "team_rank",
        "possession_id",
        "GameID",
        "line_type",
        "start_y",
        "end_y",
        "field_progress",
        "throw_count",
        "huck_count",
        "total_aec",
        args.metric,
    ]

    if args.all_teams:
        top_by_team, paths_by_team = select_top_aec_possessions_by_team(
            possessions,
            paths,
            metric=args.metric,
            n=args.top,
            exclude_hucks=not args.include_hucks,
            line_type=None if args.all_lines else "o_line",
        )
        print(f"Season: {args.season}")
        print(f"Games loaded: {len(games):,}")
        print(f"Throws loaded: {len(throws):,}")
        print(f"Scoring possessions: {len(possessions):,}")
        print(f"Teams with qualifying possessions: {top_by_team['team_id'].nunique() if not top_by_team.empty else 0:,}")
        print("\nTop AEC possessions by team")
        print(top_by_team.reindex(columns=columns).to_string(index=False))

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for team_id, team_possessions in top_by_team.groupby("team_id"):
                team_paths = paths_by_team.get(str(team_id), [])
                labeled = _label_paths(team_possessions, team_paths, "top", args.metric)
                if not labeled:
                    continue
                fig = plot_representative_paths(
                    labeled,
                    title=f"{str(team_id).title()} top {len(labeled)} non-huck long-field scoring possessions",
                )
                output_path = args.output_dir / f"{team_id}_top_aec_possessions.html"
                fig.write_html(output_path)
                print(f"Wrote {output_path}")
        return 0

    sets = build_aec_possession_sets(
        possessions,
        paths,
        top_n=args.top,
        middle_count=args.middle,
        metric=args.metric,
        exclude_hucks=not args.include_hucks,
        line_type=None if args.all_lines else "o_line",
    )

    filtered, filtered_paths = sets["filtered"]
    highest, highest_paths = sets["highest"]
    middle, middle_paths = sets["middle"]

    print(f"Team: {args.team}")
    print(f"Season: {args.season}")
    print(f"Games loaded: {len(games):,}")
    print(f"Throws loaded: {len(throws):,}")
    print(f"Scoring possessions: {len(possessions):,}")
    print(f"Filtered analysis possessions: {len(filtered):,}")

    print("\nHighest AEC possessions")
    print(highest.reindex(columns=columns).to_string(index=False))
    print("\nMiddle AEC possessions")
    print(middle.reindex(columns=columns).to_string(index=False))

    if args.output_html is not None:
        labeled = {}
        labeled.update(_label_paths(highest, highest_paths, "top", args.metric))
        labeled.update(_label_paths(middle, middle_paths, "middle", args.metric))
        fig = plot_representative_paths(labeled, title=f"{args.team.title()} high and middle AEC possessions")
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(args.output_html)
        print(f"\nWrote {args.output_html}")

    # Keep variables referenced so linters do not complain when the script is adapted.
    _ = filtered_paths
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
