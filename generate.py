"""
Synthetic Review Data Generator - Main CLI Entry Point
"""

import click
import os
import json
import jsonlines
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from datetime import datetime

# Import all components
from src.utils.config_loader import load_config, get_api_key
from src.utils.logger import PipelineLogger
from src.utils.faker_utils import DiversityInjector
from src.utils.metrics_tracker import MetricsTracker
from src.models.model_router import ModelRouter
from src.models.openai_client import OpenAIClient
from src.models.gemini_client import GeminiClient
from src.models.local_client import LocalModelClient
from src.agents.generator import ReviewGeneratorAgent
from src.agents.orchestrator import ReviewGenerationOrchestrator
from src.quality.diversity_metrics import DiversityMetrics
from src.quality.bias_detection import BiasDetector
from src.quality.realism_validator import RealismValidator
from src.quality.llm_judge_evaluator import LLMJudgeEvaluator
from src.report.report_generator import QualityReportGenerator

console = Console()

"""
Fixed generate.py - Allow running without LLM Judge
"""

@click.command()
@click.option('--config', default='config.yaml', help='Path to configuration file', type=click.Path(exists=True))
@click.option('--output', default='data/generated/reviews.jsonl', help='Output path for generated dataset', type=click.Path())
@click.option('--domain', default=None, help='Override domain from config', type=str)
@click.option('--samples', default=None, help='Override target samples from config', type=int)
@click.option('--real-reviews', default='data/real_reviews/reviews.jsonl', help='Path to real reviews for comparison', type=click.Path())
@click.option('--skip-llm-judge', is_flag=True, help='Skip LLM judge evaluation (statistical only)')
@click.option('--report-output', default='data/generated/quality_report.md', help='Output path for quality report', type=click.Path())
@click.option('--benchmark', is_flag=True, help='Run in benchmark mode (compare models)')

def main(config, output, domain, samples, real_reviews, skip_llm_judge, report_output, benchmark):
    """
    Generate synthetic review dataset with comprehensive quality guardrails
    """
    
    console.print("[bold cyan]Synthetic Review Data Generator v1.0[/bold cyan]")
    console.print("=" * 70)
    
    # Load configuration
    console.print("\n[yellow]Loading configuration...[/yellow]")
    try:
        cfg = load_config(config)
        
        if domain:
            cfg.generation.domain = domain
        if samples:
            cfg.generation.target_samples = samples
        
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"  - Domain: {cfg.generation.domain}")
        console.print(f"  - Target samples: {cfg.generation.target_samples}")
        console.print(f"  - Models: {', '.join([m.model for m in cfg.models])}")
        console.print(f"  - LLM Judge: {'Disabled (statistical only)' if skip_llm_judge else f'Enabled (Primary: {cfg.llm_judge.primary_model})'}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {str(e)}")
        return
    
    # Initialize components
    console.print("\n[yellow]Initializing components...[/yellow]")
    
    # Logger
    logger = PipelineLogger(
        log_dir=cfg.logging.get('output_dir', 'logs'),
        level=cfg.logging.get('level', 'INFO')
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(target_samples=cfg.generation.target_samples)
    
    # Diversity injector
    diversity_injector = DiversityInjector()
    
    # Model router
    model_router = ModelRouter(cfg.models)
    console.print(f"[green]✓[/green] Model router initialized with {len(cfg.models)} models")
    
    # Quality metrics
    diversity_metrics = DiversityMetrics()
    bias_detector = BiasDetector()  # Pass config dict
    
    # Load real reviews if available
    real_reviews_list = []
    if Path(real_reviews).exists():
        try:
            with jsonlines.open(real_reviews) as reader:
                # real_reviews_list = [r['text'] for r in reader if 'text' in r]
                real_reviews_list = [r['body'] for r in reader if 'body' in r]
            console.print(f"[green]✓[/green] Loaded {len(real_reviews_list)} real reviews for comparison")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not load real reviews: {str(e)}")
    else:
        console.print(f"[yellow]⚠[/yellow] No real reviews found at {real_reviews}")
    
    realism_validator = RealismValidator(real_reviews=real_reviews_list)
    
    # LLM Judge - OPTIONAL
    llm_judge = None
    if not skip_llm_judge:
        try:
            primary_judge_model = cfg.llm_judge.primary_model
            is_primary_local = "/" in primary_judge_model
            
            if is_primary_local:
                console.print(f"[yellow]Loading local model as primary judge: {primary_judge_model}[/yellow]")
                primary_judge = LocalModelClient(
                    model_name=primary_judge_model,
                    temperature=cfg.llm_judge.temperature,
                    device="auto",
                    load_in_4bit=True
                )
                console.print(f"[green]✓[/green] Local primary judge loaded")
            elif primary_judge_model.startswith("gemini"):
                gemini_key = get_api_key('GEMINI_API_KEY')
                primary_judge = GeminiClient(
                    api_key=gemini_key,
                    model=primary_judge_model,
                    temperature=cfg.llm_judge.temperature
                )
            else:
                openai_key = get_api_key('OPENAI_API_KEY')
                primary_judge = OpenAIClient(
                    api_key=openai_key,
                    model=primary_judge_model,
                    temperature=cfg.llm_judge.temperature
                )
            
            # Secondary judge (optional)
            secondary_judge = None
            if cfg.llm_judge.enable_multi_judge:
                try:
                    secondary_judge_model = cfg.llm_judge.secondary_model
                    is_secondary_local = "/" in secondary_judge_model
                    
                    if is_secondary_local:
                        console.print(f"[yellow]Loading local model as secondary judge: {secondary_judge_model}[/yellow]")
                        secondary_judge = LocalModelClient(
                            model_name=secondary_judge_model,
                            temperature=cfg.llm_judge.temperature,
                            device="auto",
                            load_in_4bit=True
                        )
                        console.print(f"[green]✓[/green] Local secondary judge loaded")
                    elif secondary_judge_model.startswith("gemini"):
                        gemini_key = get_api_key('GEMINI_API_KEY')
                        secondary_judge = GeminiClient(
                            api_key=gemini_key,
                            model=secondary_judge_model,
                            temperature=cfg.llm_judge.temperature
                        )
                    else:
                        openai_key = get_api_key('OPENAI_API_KEY')
                        secondary_judge = OpenAIClient(
                            api_key=openai_key,
                            model=secondary_judge_model,
                            temperature=cfg.llm_judge.temperature
                        )
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Secondary judge not available: {str(e)}")
            
            llm_judge = LLMJudgeEvaluator(
                primary_model_client=primary_judge,
                secondary_model_client=secondary_judge,
                enable_multi_judge=cfg.llm_judge.enable_multi_judge,
                temperature=cfg.llm_judge.temperature
            )
            console.print(f"[green]✓[/green] LLM Judge initialized (Primary: {cfg.llm_judge.primary_model})")
        
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize LLM Judge: {str(e)}")
            console.print("[yellow]Falling back to statistical checks only[/yellow]")
            llm_judge = None
    else:
        console.print("[yellow]⚠[/yellow] LLM Judge disabled - statistical checks only")
    
    # Generator agent
    generator_agent = ReviewGeneratorAgent(
        model_router=model_router,
        diversity_injector=diversity_injector
    )
    
    # Orchestrator - NOW WORKS WITH OR WITHOUT LLM JUDGE
    orchestrator = ReviewGenerationOrchestrator(
        config=cfg,
        generator_agent=generator_agent,
        diversity_metrics=diversity_metrics,
        bias_detector=bias_detector,
        realism_validator=realism_validator,
        llm_judge=llm_judge,  # Can be None!
        logger=logger,
        metrics_tracker=metrics_tracker
    )
    
    console.print("[green]✓[/green] All components initialized")
    
    # Generate dataset
    console.print(f"\n[bold yellow]Starting dataset generation...[/bold yellow]")
    console.print(f"Target: {cfg.generation.target_samples} reviews\n")
    
    start_time = datetime.now()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Generating reviews...",
            total=cfg.generation.target_samples
        )
        
        # Run generation
        generated_reviews = orchestrator.generate_dataset()
        
        progress.update(task, completed=len(generated_reviews))
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Finalize metrics
    metrics_tracker.finalize()
    
    # Save dataset
    console.print(f"\n[yellow]Saving dataset...[/yellow]")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, mode='w') as writer:
        for review in generated_reviews:
            writer.write({
                "id": review['review_id'],
                "text": review['review_text'],
                "rating": review['rating'],
                "persona": review['persona']['name'],
                "model": review['model_used'],
                "seed_words": review['seed_words'],
                "generation_time": review['generation_time'],
                "cost": review['cost'],
                "statistical_scores": review['statistical_metrics'],
                "llm_judge_scores": review.get('llm_judge_results', {}),
                "final_decision": review['final_decision'],
                "regeneration_count": review['regeneration_count'],
                "metadata": review['persona']
            })
    
    console.print(f"[green]✓[/green] Dataset saved to {output_path}")
    
    # Generate quality report
    console.print(f"\n[yellow]Generating quality report...[/yellow]")
    
    report_generator = QualityReportGenerator(
        metrics=metrics_tracker.get_metrics(),
        diversity_metrics=diversity_metrics,
        realism_validator=realism_validator,
        generated_reviews=generated_reviews
    )
    
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_generator.generate_markdown_report(str(report_path))
    
    console.print(f"[green]✓[/green] Quality report saved to {report_path}")
    
    # Save metrics as JSON
    metrics_path = output_path.parent / f"metrics_{metrics_tracker.metrics.start_time.strftime('%Y%m%d_%H%M%S')}.json"
    metrics_tracker.metrics.save(str(metrics_path))
    
    # Print summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]Generation Complete![/bold green]")
    console.print(f"""
[cyan]Summary:[/cyan]
  - Samples Generated: {metrics_tracker.metrics.total_generated}
  - Samples Accepted: {metrics_tracker.metrics.total_accepted}
  - Success Rate: {metrics_tracker.metrics.success_rate*100:.1f}%
  - Duration: {duration/60:.1f} minutes
  - Total Cost: ${metrics_tracker.metrics.total_cost:.2f}
  - Throughput: {metrics_tracker.metrics.total_accepted/(duration/60):.1f} samples/min

[cyan]Output Files:[/cyan]
  - Dataset: {output_path}
  - Report: {report_path}
  - Metrics: {metrics_path}
  - Logs: logs/session_{logger.session_id}.log
""")
    
    console.print("=" * 70)


if __name__ == "__main__":
    main()