from pathlib import Path

import click

from src.models.local_client import LocalModelClient


@click.command()
@click.option(
    "--model-name",
    default="microsoft/Phi-3.5-mini-instruct",
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--load-in-4bit", is_flag=True, help="Use 4-bit quantization.")
@click.option("--load-in-8bit", is_flag=True, help="Use 8-bit quantization.")
def main(model_name: str, device: str, load_in_4bit: bool, load_in_8bit: bool) -> None:
    """Quick smoke test for local generation."""
    client = LocalModelClient(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    system_prompt = "You are a helpful assistant that writes short, realistic product reviews."
    user_prompt = "Write a 3-sentence review for a developer tool that helps with CI pipelines."

    result = client.generate(system_prompt, user_prompt, max_tokens=120)
    if not result["success"]:
        click.echo(f"✗ Generation failed: {result['error']}")
    else:
        click.echo("✓ Generation succeeded:")
        click.echo("-" * 80)
        click.echo(result["text"])
        click.echo("-" * 80)
        click.echo(
            f"time={result['time']:.2f}s, "
            f"tokens={result['tokens']['total']}, "
            f"model={result['model']}"
        )

    client.unload()


if __name__ == "__main__":
    main()
