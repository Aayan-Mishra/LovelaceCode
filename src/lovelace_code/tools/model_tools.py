"""Model delegation tool for escalating to larger/specialized models."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class DelegateToModelTool(Tool):
    name = "delegate_to_model"
    description = "Escalate complex reasoning to a larger or more specialized model"
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The task to delegate"},
            "context": {"type": "string", "description": "Relevant context for the task"},
            "model_preference": {
                "type": "string",
                "enum": ["larger", "specialized", "code", "reasoning", "fast"],
                "description": "Type of model to prefer for this task",
            },
            "max_tokens": {"type": "integer", "description": "Maximum tokens for response"},
            "temperature": {"type": "number", "description": "Temperature for generation"},
        },
        "required": ["task"],
    }

    # Model recommendations by preference
    MODEL_RECOMMENDATIONS = {
        "larger": {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4.5-preview",
            "gemini": "gemini-2.5-pro-preview-05-06",
            "openrouter": "anthropic/claude-sonnet-4-20250514",
        },
        "specialized": {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "o3",
            "gemini": "gemini-2.5-pro-preview-05-06",
            "openrouter": "openai/o3",
        },
        "code": {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4.5-preview",
            "gemini": "gemini-2.5-pro-preview-05-06",
            "openrouter": "anthropic/claude-sonnet-4-20250514",
        },
        "reasoning": {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "o3",
            "gemini": "gemini-2.5-flash-preview-04-17",
            "openrouter": "openai/o3",
        },
        "fast": {
            "anthropic": "claude-3-5-haiku-20241022",
            "openai": "gpt-4.1-mini",
            "gemini": "gemini-2.0-flash",
            "openrouter": "anthropic/claude-3-5-haiku-20241022",
        },
    }

    def execute(
        self,
        task: str,
        context: str = "",
        model_preference: str = "larger",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ToolResult:
        try:
            # This tool returns delegation instructions for the agent
            # The actual model call would be handled by the main agent loop
            
            recommendations = self.MODEL_RECOMMENDATIONS.get(model_preference, {})
            
            # Build the delegation request
            prompt_parts = [
                f"Task: {task}",
            ]
            
            if context:
                prompt_parts.extend([
                    "",
                    "Context:",
                    context,
                ])
            
            prompt = "\n".join(prompt_parts)
            
            output_parts = [
                "🔄 Delegation Request",
                "─" * 40,
                "",
                f"Preference: {model_preference}",
                f"Max tokens: {max_tokens}",
                f"Temperature: {temperature}",
                "",
                "Recommended models by provider:",
            ]
            
            for provider, model in recommendations.items():
                output_parts.append(f"  • {provider}: {model}")
            
            output_parts.extend([
                "",
                "Task:",
                task[:200] + ("..." if len(task) > 200 else ""),
            ])
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={
                    "delegation_request": {
                        "prompt": prompt,
                        "model_preference": model_preference,
                        "recommended_models": recommendations,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    "requires_escalation": True,
                },
            )
            
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


@register_tool
class ModelCapabilitiesTool(Tool):
    name = "model_capabilities"
    description = "Get information about available models and their capabilities"
    parameters = {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "enum": ["all", "anthropic", "openai", "gemini", "xai", "openrouter", "local"],
                "description": "Provider to get model info for",
            },
        },
        "required": [],
    }

    CAPABILITIES = {
        "anthropic": {
            "models": [
                {"name": "claude-sonnet-4-20250514", "context": 200000, "strengths": ["code", "reasoning", "analysis"]},
                {"name": "claude-3-5-haiku-20241022", "context": 200000, "strengths": ["speed", "efficiency"]},
            ],
            "features": ["function_calling", "vision", "streaming"],
        },
        "openai": {
            "models": [
                {"name": "gpt-4.5-preview", "context": 128000, "strengths": ["code", "general"]},
                {"name": "gpt-4.1", "context": 128000, "strengths": ["balanced", "reliable"]},
                {"name": "gpt-4.1-mini", "context": 128000, "strengths": ["speed", "cost"]},
                {"name": "o3", "context": 200000, "strengths": ["reasoning", "math", "code"]},
                {"name": "o4-mini", "context": 200000, "strengths": ["reasoning", "efficiency"]},
            ],
            "features": ["function_calling", "vision", "streaming", "json_mode"],
        },
        "gemini": {
            "models": [
                {"name": "gemini-2.5-pro-preview-05-06", "context": 1000000, "strengths": ["long_context", "multimodal"]},
                {"name": "gemini-2.5-flash-preview-04-17", "context": 1000000, "strengths": ["speed", "reasoning"]},
                {"name": "gemini-2.0-flash", "context": 1000000, "strengths": ["speed", "efficiency"]},
            ],
            "features": ["function_calling", "vision", "streaming", "grounding"],
        },
        "xai": {
            "models": [
                {"name": "grok-3", "context": 131072, "strengths": ["reasoning", "realtime"]},
                {"name": "grok-3-fast", "context": 131072, "strengths": ["speed"]},
            ],
            "features": ["function_calling", "streaming"],
        },
        "local": {
            "models": [
                {"name": "Spestly/Lovelace-1-3B", "context": 8192, "strengths": ["privacy", "offline"]},
                {"name": "Spestly/Lovelace-1-7B", "context": 8192, "strengths": ["privacy", "quality"]},
            ],
            "features": ["offline", "privacy", "customizable"],
        },
    }

    def execute(self, provider: str = "all") -> ToolResult:
        try:
            output_parts = ["🤖 Model Capabilities", "─" * 40, ""]
            data: dict[str, Any] = {}
            
            providers = [provider] if provider != "all" else list(self.CAPABILITIES.keys())
            
            for prov in providers:
                if prov not in self.CAPABILITIES:
                    continue
                    
                info = self.CAPABILITIES[prov]
                data[prov] = info
                
                output_parts.append(f"📦 {prov.upper()}")
                output_parts.append(f"   Features: {', '.join(info['features'])}")
                output_parts.append("   Models:")
                
                for model in info["models"]:
                    ctx = f"{model['context']:,}" if model['context'] else "?"
                    strengths = ", ".join(model["strengths"])
                    output_parts.append(f"     • {model['name']}")
                    output_parts.append(f"       Context: {ctx} | Strengths: {strengths}")
                
                output_parts.append("")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data=data,
            )
            
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
