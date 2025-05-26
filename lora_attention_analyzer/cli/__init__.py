"""
Command-line interface for LoRA attention analysis.

This module provides the CLI entry points for running LoRA attention analysis
from the command line using Fire.
"""

from .main import main, run_lora_comparison, analyze_lora_file

__all__ = [
    "main",
    "run_lora_comparison", 
    "analyze_lora_file",
]

