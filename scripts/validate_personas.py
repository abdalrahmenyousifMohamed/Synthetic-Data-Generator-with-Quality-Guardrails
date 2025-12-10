from pathlib import Path

import click

from src.utils.config_loader import load_config


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config.yaml",
    show_default=True,
)
def main(config: Path) -> None:
    """Validate persona weights and rating distribution from the config."""
    cfg = load_config(str(config))

    # Personas
    total_weight = sum(p.weight for p in cfg.personas)
    click.echo("Personas:")
    for p in cfg.personas:
        pct = 100 * p.weight / total_weight if total_weight else 0
        click.echo(f"  - {p.name:20s} weight={p.weight:.3f} ({pct:5.1f}%)")
    click.echo(f"Total persona weight: {total_weight:.3f}")

    # Ratings
    click.echo("\nRating distribution:")
    r_sum = sum(cfg.rating_distribution.values())
    for r in [1, 2, 3, 4, 5]:
        key = f"{r}_star"
        val = cfg.rating_distribution.get(key, 0.0)
        pct = 100 * val / r_sum if r_sum else 0
        click.echo(f"  - {r}★: {val:.3f} ({pct:5.1f}%)")
    click.echo(f"Total rating mass: {r_sum:.3f}")

    if abs(total_weight - 1.0) < 1e-6 and abs(r_sum - 1.0) < 1e-6:
        click.echo("\n✓ Personas and rating distribution are normalized.")
    else:
        click.echo("\n⚠ Distributions are not perfectly normalized – please review.")


if __name__ == "__main__":
    main()
