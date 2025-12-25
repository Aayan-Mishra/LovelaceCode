"""Agentic tools for Lovelace Code."""

from __future__ import annotations

from .base import Tool, ToolResult
from .registry import get_tool, list_tools, execute_tool

# File tools
from .file_tools import ReadFileTool, WriteFileTool, ListDirectoryTool, SearchFilesTool

# Shell tools
from .shell_tools import RunCommandTool

# Git tools
from .git_tools import GitStatusTool, GitDiffTool, GitCommitTool

# Repo analysis tools
from .repo_tools import RepoIndexerTool, SemanticSearchTool, DependencyTracerTool

# Task planning and memory tools
from .task_tools import TaskPlannerTool, ProgressTrackerTool, ProjectMemoryTool

# Code tools (patching, review, security)
from .code_tools import ApplyPatchTool, DiffReviewerTool, SecurityScanTool, RunTestsTool

# Model delegation tools
from .model_tools import DelegateToModelTool, ModelCapabilitiesTool

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "get_tool",
    "list_tools",
    "execute_tool",
    # File tools
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "SearchFilesTool",
    # Shell tools
    "RunCommandTool",
    # Git tools
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    # Repo analysis tools
    "RepoIndexerTool",
    "SemanticSearchTool",
    "DependencyTracerTool",
    # Task planning and memory tools
    "TaskPlannerTool",
    "ProgressTrackerTool",
    "ProjectMemoryTool",
    # Code tools
    "ApplyPatchTool",
    "DiffReviewerTool",
    "SecurityScanTool",
    "RunTestsTool",
    # Model delegation tools
    "DelegateToModelTool",
    "ModelCapabilitiesTool",
]
