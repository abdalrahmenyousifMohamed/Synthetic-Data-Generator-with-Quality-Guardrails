from pathlib import Path

import click
from transformers import AutoModelForCausalLM, AutoTokenizer


@click.command()
@click.argument("model_name")
@click.option(
    "--cache-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional local cache directory (defaults to Hugging Face cache).",
)
def main(model_name: str, cache_dir: Path | None) -> None:
    """
    Pre-download a local model and tokenizer so later runs don't hit the network.

    Example:
      python scripts/download_local_model.py microsoft/Phi-3.5-mini-instruct
    """
    kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)

    click.echo(f"↓ Downloading tokenizer for {model_name}...")
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    click.echo(f"↓ Downloading model weights for {model_name}...")
    AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    click.echo("✓ Download complete.")


if __name__ == "__main__":
    main()
