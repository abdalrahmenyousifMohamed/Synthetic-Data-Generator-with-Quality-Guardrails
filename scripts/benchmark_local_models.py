import time
from statistics import mean
from typing import List

import click

from src.models.local_client import LocalModelClient


def _run_single(
    model_name: str,
    prompt: str,
    runs: int,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> None:
    client = LocalModelClient(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    times: List[float] = []
    for i in range(runs):
        start = time.time()
        res = client.generate("You are a helpful assistant.", prompt, max_tokens=128)
        end = time.time()
        times.append(end - start)
        if not res["success"]:
            print(f"[{model_name}] run {i+1}/{runs} FAILED: {res['error']}")
        else:
            print(f"[{model_name}] run {i+1}/{runs}: {end-start:.2f}s")

    client.unload()
    print(f"[{model_name}] avg={mean(times):.2f}s over {runs} runs")


@click.command()
@click.argument("model_names", nargs=-1)
@click.option("--runs", "-n", default=3, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--load-in-4bit", is_flag=True)
@click.option("--load-in-8bit", is_flag=True)
def main(
    model_names: tuple[str, ...],
    runs: int,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> None:
    """Benchmark one or more local models on the same prompt."""
    if not model_names:
        raise click.UsageError("Please provide at least one MODEL_NAME.")

    prompt = "Write a short, 3-sentence product review for a SaaS developer tool."

    for name in model_names:
        print("=" * 80)
        print(f"Benchmarking {name}...")
        _run_single(name, prompt, runs, device, load_in_4bit, load_in_8bit)


if __name__ == "__main__":
    main()
