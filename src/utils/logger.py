import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import os


class PipelineLogger:
    """Comprehensive logging system for all pipeline stages"""
    
    def __init__(self, log_dir: str = "logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.log_dir / "generation").mkdir(exist_ok=True)
        (self.log_dir / "quality").mkdir(exist_ok=True)
        (self.log_dir / "llm_judge").mkdir(exist_ok=True)
        (self.log_dir / "pipeline").mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup main logger
        self.logger = logging.getLogger("SyntheticReviewGen")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / "pipeline" / f"session_{self.session_id}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def log_generation(self, review_id: str, data: Dict[str, Any]):
        """Log review generation details"""
        log_file = self.log_dir / "generation" / f"{self.session_id}.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "review_id": review_id,
            "session_id": self.session_id,
            **data
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_quality_check(self, review_id: str, stage: str, data: Dict[str, Any]):
        """Log quality check results"""
        log_file = self.log_dir / "quality" / f"{self.session_id}.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "review_id": review_id,
            "stage": stage,
            "session_id": self.session_id,
            **data
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_llm_judge(self, review_id: str, evaluation_data: Dict[str, Any]):
        """Log LLM judge evaluation details"""
        log_file = self.log_dir / "llm_judge" / f"{self.session_id}.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "review_id": review_id,
            "session_id": self.session_id,
            **evaluation_data
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_rejection(self, review_id: str, reasons: list[str], attempt: int):
        """Log review rejection"""
        self.logger.warning(
            f"Review {review_id} rejected (attempt {attempt}): {', '.join(reasons)}"
        )
        
        log_file = self.log_dir / "pipeline" / f"rejections_{self.session_id}.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "review_id": review_id,
            "attempt": attempt,
            "reasons": reasons
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    # ============================================
    # LOGGING LEVEL METHODS (all required)
    # ============================================
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def log_pipeline_start(self, config: Dict[str, Any]):
        """Log pipeline start with configuration"""
        self.info("=" * 80)
        self.info("SYNTHETIC REVIEW GENERATION PIPELINE STARTED")
        self.info(f"Session ID: {self.session_id}")
        self.info(f"Target Samples: {config.get('target_samples', 'N/A')}")
        self.info(f"Domain: {config.get('domain', 'N/A')}")
        self.info("=" * 80)
        
        log_file = self.log_dir / "pipeline" / f"config_{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_pipeline_end(self, summary: Dict[str, Any]):
        """Log pipeline completion with summary"""
        self.info("=" * 80)
        self.info("PIPELINE COMPLETED")
        self.info(f"Total Generated: {summary.get('total_generated', 0)}")
        self.info(f"Accepted: {summary.get('accepted', 0)}")
        self.info(f"Rejected: {summary.get('rejected', 0)}")
        self.info(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        self.info(f"Total Cost: ${summary.get('total_cost', 0):.2f}")
        self.info(f"Total Time: {summary.get('total_time', 0):.1f}s")
        self.info("=" * 80)
        
        log_file = self.log_dir / "pipeline" / f"summary_{self.session_id}.json"
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_session_id(self) -> str:
        """Get current session ID"""
        return self.session_id
    
    def get_log_dir(self) -> Path:
        """Get log directory path"""
        return self.log_dir


# Convenience function to get a logger instance
def get_logger(name: str = "SyntheticReviewGen", log_dir: str = "logs", level: str = "INFO") -> PipelineLogger:
    """
    Get or create a PipelineLogger instance.
    
    Args:
        name: Logger name (for namespacing)
        log_dir: Directory to store logs
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(log_dir=log_dir, level=level)