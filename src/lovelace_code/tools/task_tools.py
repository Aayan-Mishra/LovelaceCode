"""Task planning, memory, and progress tracking tools."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .registry import register_tool


# ─────────────────────────────────────────────────────────────────────────────
# Task Planner Data Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Subtask:
    """A single subtask in a plan."""
    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    dependencies: list[int] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


@dataclass
class TaskPlan:
    """A complete task plan with subtasks."""
    goal: str
    subtasks: list[Subtask] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subtasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "dependencies": t.dependencies,
                    "notes": t.notes,
                    "created_at": t.created_at,
                    "completed_at": t.completed_at,
                }
                for t in self.subtasks
            ],
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Progress Tracker with Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Checkpoint:
    """A checkpoint for rollback."""
    id: str
    description: str
    timestamp: str
    files_snapshot: dict[str, str]  # path -> content hash
    

@dataclass
class ProgressState:
    """Current progress state."""
    current_plan: TaskPlan | None = None
    checkpoints: list[Checkpoint] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    

# Global state
_progress_state: ProgressState | None = None


def get_progress_state() -> ProgressState:
    global _progress_state
    if _progress_state is None:
        _progress_state = ProgressState()
    return _progress_state


# ─────────────────────────────────────────────────────────────────────────────
# Task Planner Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class TaskPlannerTool(Tool):
    name = "task_planner"
    description = "Break a high-level goal into ordered, executable subtasks"
    parameters = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "High-level goal to break down"},
            "context": {"type": "string", "description": "Additional context about the project"},
        },
        "required": ["goal"],
    }

    def execute(self, goal: str, context: str = "") -> ToolResult:
        try:
            state = get_progress_state()
            
            # Generate subtasks based on common patterns
            subtasks = self._generate_subtasks(goal, context)
            
            plan = TaskPlan(goal=goal, subtasks=subtasks)
            state.current_plan = plan
            
            # Format output
            output_parts = [
                f"📋 Task Plan: {goal}",
                "─" * 50,
                "",
            ]
            
            for task in subtasks:
                deps_str = f" (after: {task.dependencies})" if task.dependencies else ""
                output_parts.append(f"  {task.id}. [ ] {task.description}{deps_str}")
            
            output_parts.extend([
                "",
                f"Total subtasks: {len(subtasks)}",
                "Use progress_tracker to update status.",
            ])
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data=plan.to_dict(),
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _generate_subtasks(self, goal: str, context: str) -> list[Subtask]:
        """Generate subtasks based on the goal."""
        goal_lower = goal.lower()
        subtasks = []
        task_id = 1
        
        # Common patterns
        if any(word in goal_lower for word in ["add", "implement", "create", "build"]):
            subtasks.extend([
                Subtask(id=task_id, description="Analyze existing codebase structure"),
                Subtask(id=task_id + 1, description="Identify affected files and modules", dependencies=[task_id]),
                Subtask(id=task_id + 2, description="Design the implementation approach", dependencies=[task_id + 1]),
                Subtask(id=task_id + 3, description="Implement core functionality", dependencies=[task_id + 2]),
                Subtask(id=task_id + 4, description="Add error handling and edge cases", dependencies=[task_id + 3]),
                Subtask(id=task_id + 5, description="Write/update tests", dependencies=[task_id + 3]),
                Subtask(id=task_id + 6, description="Update documentation", dependencies=[task_id + 4, task_id + 5]),
                Subtask(id=task_id + 7, description="Review and validate changes", dependencies=[task_id + 6]),
            ])
        elif any(word in goal_lower for word in ["fix", "bug", "debug", "repair"]):
            subtasks.extend([
                Subtask(id=task_id, description="Reproduce the issue"),
                Subtask(id=task_id + 1, description="Identify root cause", dependencies=[task_id]),
                Subtask(id=task_id + 2, description="Implement fix", dependencies=[task_id + 1]),
                Subtask(id=task_id + 3, description="Add regression test", dependencies=[task_id + 2]),
                Subtask(id=task_id + 4, description="Verify fix works", dependencies=[task_id + 2, task_id + 3]),
            ])
        elif any(word in goal_lower for word in ["refactor", "improve", "optimize", "clean"]):
            subtasks.extend([
                Subtask(id=task_id, description="Analyze current implementation"),
                Subtask(id=task_id + 1, description="Identify improvement opportunities", dependencies=[task_id]),
                Subtask(id=task_id + 2, description="Create checkpoint before changes"),
                Subtask(id=task_id + 3, description="Apply refactoring incrementally", dependencies=[task_id + 1, task_id + 2]),
                Subtask(id=task_id + 4, description="Run tests after each change", dependencies=[task_id + 3]),
                Subtask(id=task_id + 5, description="Verify no regressions", dependencies=[task_id + 4]),
            ])
        elif any(word in goal_lower for word in ["test", "coverage"]):
            subtasks.extend([
                Subtask(id=task_id, description="Identify untested code paths"),
                Subtask(id=task_id + 1, description="Write unit tests", dependencies=[task_id]),
                Subtask(id=task_id + 2, description="Write integration tests", dependencies=[task_id]),
                Subtask(id=task_id + 3, description="Run test suite", dependencies=[task_id + 1, task_id + 2]),
                Subtask(id=task_id + 4, description="Review coverage report", dependencies=[task_id + 3]),
            ])
        else:
            # Generic task breakdown
            subtasks.extend([
                Subtask(id=task_id, description="Understand requirements"),
                Subtask(id=task_id + 1, description="Research and plan approach", dependencies=[task_id]),
                Subtask(id=task_id + 2, description="Implement changes", dependencies=[task_id + 1]),
                Subtask(id=task_id + 3, description="Test and validate", dependencies=[task_id + 2]),
                Subtask(id=task_id + 4, description="Document changes", dependencies=[task_id + 3]),
            ])
        
        return subtasks


# ─────────────────────────────────────────────────────────────────────────────
# Progress Tracker Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class ProgressTrackerTool(Tool):
    name = "progress_tracker"
    description = "Track task completion, update status, and manage checkpoints"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "complete", "start", "block", "checkpoint", "rollback"],
                "description": "Action to perform",
            },
            "task_id": {"type": "integer", "description": "Task ID (for complete/start/block)"},
            "checkpoint_id": {"type": "string", "description": "Checkpoint ID (for rollback)"},
            "description": {"type": "string", "description": "Description (for checkpoint)"},
        },
        "required": ["action"],
    }

    def execute(
        self,
        action: str,
        task_id: int | None = None,
        checkpoint_id: str | None = None,
        description: str = "",
    ) -> ToolResult:
        try:
            state = get_progress_state()
            
            if action == "status":
                return self._show_status(state)
            elif action == "complete":
                return self._complete_task(state, task_id)
            elif action == "start":
                return self._start_task(state, task_id)
            elif action == "block":
                return self._block_task(state, task_id)
            elif action == "checkpoint":
                return self._create_checkpoint(state, description)
            elif action == "rollback":
                return self._rollback(state, checkpoint_id)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
                
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _show_status(self, state: ProgressState) -> ToolResult:
        output_parts = ["📊 Progress Status", "─" * 40, ""]
        
        if state.current_plan:
            plan = state.current_plan
            completed = sum(1 for t in plan.subtasks if t.status == "completed")
            total = len(plan.subtasks)
            pct = (completed / total * 100) if total > 0 else 0
            
            output_parts.append(f"Current Goal: {plan.goal}")
            output_parts.append(f"Progress: {completed}/{total} ({pct:.0f}%)")
            output_parts.append("")
            
            status_icons = {
                "pending": "○",
                "in_progress": "◐",
                "completed": "●",
                "blocked": "✗",
            }
            
            for task in plan.subtasks:
                icon = status_icons.get(task.status, "?")
                output_parts.append(f"  {icon} {task.id}. {task.description}")
        else:
            output_parts.append("No active plan. Use task_planner to create one.")
        
        output_parts.extend(["", f"Checkpoints: {len(state.checkpoints)}"])
        for cp in state.checkpoints[-3:]:
            output_parts.append(f"  • {cp.id}: {cp.description}")
        
        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            data={
                "plan": state.current_plan.to_dict() if state.current_plan else None,
                "checkpoints": len(state.checkpoints),
            },
        )
    
    def _complete_task(self, state: ProgressState, task_id: int | None) -> ToolResult:
        if not state.current_plan:
            return ToolResult(success=False, output="", error="No active plan")
        if task_id is None:
            return ToolResult(success=False, output="", error="task_id required")
        
        for task in state.current_plan.subtasks:
            if task.id == task_id:
                task.status = "completed"
                task.completed_at = datetime.now().isoformat()
                return ToolResult(
                    success=True,
                    output=f"✓ Completed: {task.description}",
                    data={"task_id": task_id, "status": "completed"},
                )
        
        return ToolResult(success=False, output="", error=f"Task {task_id} not found")
    
    def _start_task(self, state: ProgressState, task_id: int | None) -> ToolResult:
        if not state.current_plan:
            return ToolResult(success=False, output="", error="No active plan")
        if task_id is None:
            return ToolResult(success=False, output="", error="task_id required")
        
        for task in state.current_plan.subtasks:
            if task.id == task_id:
                task.status = "in_progress"
                return ToolResult(
                    success=True,
                    output=f"▶ Started: {task.description}",
                    data={"task_id": task_id, "status": "in_progress"},
                )
        
        return ToolResult(success=False, output="", error=f"Task {task_id} not found")
    
    def _block_task(self, state: ProgressState, task_id: int | None) -> ToolResult:
        if not state.current_plan:
            return ToolResult(success=False, output="", error="No active plan")
        if task_id is None:
            return ToolResult(success=False, output="", error="task_id required")
        
        for task in state.current_plan.subtasks:
            if task.id == task_id:
                task.status = "blocked"
                return ToolResult(
                    success=True,
                    output=f"✗ Blocked: {task.description}",
                    data={"task_id": task_id, "status": "blocked"},
                )
        
        return ToolResult(success=False, output="", error=f"Task {task_id} not found")
    
    def _create_checkpoint(self, state: ProgressState, description: str) -> ToolResult:
        checkpoint = Checkpoint(
            id=f"cp_{int(time.time())}",
            description=description or f"Checkpoint at {datetime.now().strftime('%H:%M:%S')}",
            timestamp=datetime.now().isoformat(),
            files_snapshot={},  # Would store file hashes in real implementation
        )
        state.checkpoints.append(checkpoint)
        
        return ToolResult(
            success=True,
            output=f"📌 Checkpoint created: {checkpoint.id}",
            data={"checkpoint_id": checkpoint.id, "description": checkpoint.description},
        )
    
    def _rollback(self, state: ProgressState, checkpoint_id: str | None) -> ToolResult:
        if not checkpoint_id:
            return ToolResult(success=False, output="", error="checkpoint_id required")
        
        for i, cp in enumerate(state.checkpoints):
            if cp.id == checkpoint_id:
                # Remove checkpoints after this one
                state.checkpoints = state.checkpoints[:i + 1]
                return ToolResult(
                    success=True,
                    output=f"↩ Rolled back to: {cp.description}",
                    data={"checkpoint_id": checkpoint_id},
                )
        
        return ToolResult(success=False, output="", error=f"Checkpoint {checkpoint_id} not found")


# ─────────────────────────────────────────────────────────────────────────────
# Project Memory Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class ProjectMemoryTool(Tool):
    name = "project_memory"
    description = "Store and retrieve important design decisions and project context"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "clear"],
                "description": "Action to perform",
            },
            "content": {"type": "string", "description": "Memory content (for add)"},
            "query": {"type": "string", "description": "Search query (for search)"},
            "category": {
                "type": "string",
                "enum": ["decision", "architecture", "convention", "todo", "note"],
                "description": "Memory category",
            },
        },
        "required": ["action"],
    }

    def execute(
        self,
        action: str,
        content: str = "",
        query: str = "",
        category: str = "note",
    ) -> ToolResult:
        try:
            state = get_progress_state()
            
            if action == "add":
                if not content:
                    return ToolResult(success=False, output="", error="content required for add")
                
                memory = {
                    "id": len(state.memories) + 1,
                    "content": content,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                }
                state.memories.append(memory)
                
                return ToolResult(
                    success=True,
                    output=f"💾 Memory saved: [{category}] {content[:50]}...",
                    data=memory,
                )
            
            elif action == "list":
                if not state.memories:
                    return ToolResult(success=True, output="No memories stored.", data={"memories": []})
                
                output_parts = ["📚 Project Memory", "─" * 40, ""]
                
                # Group by category
                by_category: dict[str, list] = {}
                for mem in state.memories:
                    cat = mem.get("category", "note")
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(mem)
                
                for cat, mems in by_category.items():
                    output_parts.append(f"[{cat.upper()}]")
                    for mem in mems:
                        output_parts.append(f"  • {mem['content']}")
                    output_parts.append("")
                
                return ToolResult(
                    success=True,
                    output="\n".join(output_parts),
                    data={"memories": state.memories},
                )
            
            elif action == "search":
                if not query:
                    return ToolResult(success=False, output="", error="query required for search")
                
                query_lower = query.lower()
                matches = [
                    mem for mem in state.memories
                    if query_lower in mem.get("content", "").lower()
                ]
                
                if not matches:
                    return ToolResult(success=True, output="No matching memories found.", data={"matches": []})
                
                output_parts = [f"🔍 Search results for '{query}':", ""]
                for mem in matches:
                    output_parts.append(f"  [{mem['category']}] {mem['content']}")
                
                return ToolResult(
                    success=True,
                    output="\n".join(output_parts),
                    data={"matches": matches},
                )
            
            elif action == "clear":
                count = len(state.memories)
                state.memories.clear()
                return ToolResult(
                    success=True,
                    output=f"🗑 Cleared {count} memories.",
                    data={"cleared": count},
                )
            
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
                
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
