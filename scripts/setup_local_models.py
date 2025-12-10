from pathlib import Path

import click

from src.utils.config_loader import ModelConfig


@click.command()
@click.option(
    "--config-out",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default="config_local_only.yaml",
    show_default=True,
)
def main(config_out: Path) -> None:
    """
    Interactive wizard to create a local-model-only config.

    This does NOT download models; use download_local_model.py for that.
    """
    click.echo("Local model setup wizard")
    click.echo("-" * 40)

    model_name = click.prompt(
        "Hugging Face model id", default="microsoft/Phi-3.5-mini-instruct"
    )
    temperature = click.prompt("Default temperature", type=float, default=1.0)
    batch_size = click.prompt("Generation batch size", type=int, default=10)
    target_samples = click.prompt("Target samples", type=int, default=100)

    cfg = {
        "generation": {
            "target_samples": target_samples,
            "batch_size": batch_size,
            "temperature": temperature,
            "seed_word_count": 10,
            "domain": "developer-tools",
        },
        "models": [
            {
                "provider": "local",
                "model": model_name,
                "weight": 1.0,
                "temperature": temperature,
                "max_tokens": 1024,
                "local_config": {
                    "device": "auto",
                    "load_in_4bit": True,
                    "load_in_8bit": False,
                },
            }
        ],
        "llm_judge": {
            "primary_model": "gemini-2.0-flash-exp",
            "secondary_model": "gpt-4",
            "temperature": 0.2,
            "enable_multi_judge": False,
            "parallel_execution": False,
            "max_workers": 4,
            "evaluation_criteria": [
                "authenticity",
                "alignment",
                "expertise",
                "uniqueness",
            ],
        },
        "personas": [
            {
                "name": "default_dev",
                "characteristics": "General developer using the tool",
                "experience": "intermediate",
                "focus": ["usability", "reliability"],
                "weight": 1.0,
            }
        ],
        "rating_distribution": {
            "1_star": 0.05,
            "2_star": 0.10,
            "3_star": 0.20,
            "4_star": 0.35,
            "5_star": 0.30,
        },
        "review_characteristics": {
            "min_length": 50,
            "max_length": 500,
            "include_pros_cons": 0.7,
            "include_technical_details": 0.5,
        },
        "quality_thresholds": {
            "min_length": 50,
            "max_length": 500,
            "min_unique_words": 20,
            "max_self_bleu": 0.5,
            "min_sentiment_alignment": 0.6,
            "min_diversity_score": 0.5,
            "min_realism_score": 0.6,
            "min_llm_judge_score": 0.0,
            "min_authenticity_score": 0.0,
            "min_alignment_score": 0.0,
            "min_expertise_score": 0.0,
            "min_uniqueness_score": 0.0,
            "max_regeneration_attempts": 2,
        },
        "logging": {
            "level": "INFO",
            "log_generation": True,
            "log_quality_checks": True,
            "log_llm_judge": True,
            "output_dir": "logs",
        },
    }

    import yaml

    config_out = Path(config_out)
    config_out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    click.echo(f"✓ Wrote local-model config to {config_out}")


if __name__ == "__main__":
    main()
