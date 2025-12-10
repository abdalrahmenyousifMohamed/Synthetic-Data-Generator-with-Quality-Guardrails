import csv
from pathlib import Path
from typing import List, Dict

import click

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "data" / "real_reviews"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


COMMON_FIELDS = [
    "id",
    "platform",
    "product",
    "title",
    "body",
    "rating",
    "author",
    "url",
]


PLATFORM_SPECIFIC: Dict[str, List[str]] = {
    "g2": ["company_size", "industry", "role"],
    "capterra": ["company_size", "industry", "use_case"],
    "reddit": ["subreddit", "thread_title"],
}


def _write_template(name: str, extra_fields: List[str]) -> Path:
    path = TEMPLATE_DIR / f"{name}_reviews_template.csv"
    fields = COMMON_FIELDS + extra_fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    return path


@click.command()
@click.option(
    "--platform",
    "-p",
    type=click.Choice(["g2", "capterra", "reddit", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
def main(platform: str) -> None:
    """Generate CSV templates for collecting real reviews (G2, Capterra, Reddit)."""
    platforms = ["g2", "capterra", "reddit"] if platform == "all" else [platform]
    paths = []
    for p in platforms:
        extra = PLATFORM_SPECIFIC.get(p, [])
        paths.append(_write_template(p, extra))

    for p in paths:
        click.echo(f"✓ Wrote template: {p}")


if __name__ == "__main__":
    main()
