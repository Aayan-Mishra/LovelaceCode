"""Code patching, diff review, and security scanning tools."""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .registry import register_tool


# ─────────────────────────────────────────────────────────────────────────────
# Apply Patch Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = "Apply minimal diff-based changes to files"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to patch"},
            "patch": {
                "type": "string",
                "description": "Unified diff patch to apply, or search/replace format",
            },
            "mode": {
                "type": "string",
                "enum": ["diff", "search_replace"],
                "description": "Patch mode: 'diff' for unified diff, 'search_replace' for find/replace",
            },
            "search": {"type": "string", "description": "Text to search for (for search_replace mode)"},
            "replace": {"type": "string", "description": "Replacement text (for search_replace mode)"},
            "dry_run": {"type": "boolean", "description": "Preview changes without applying"},
        },
        "required": ["file_path"],
    }

    def execute(
        self,
        file_path: str,
        patch: str = "",
        mode: str = "search_replace",
        search: str = "",
        replace: str = "",
        dry_run: bool = False,
    ) -> ToolResult:
        try:
            path = Path(file_path)
            
            if not path.exists():
                return ToolResult(success=False, output="", error=f"File not found: {file_path}")
            
            original = path.read_text(encoding="utf-8")
            
            if mode == "search_replace":
                if not search:
                    return ToolResult(success=False, output="", error="'search' required for search_replace mode")
                
                if search not in original:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Search string not found in {file_path}",
                    )
                
                modified = original.replace(search, replace, 1)
            
            elif mode == "diff":
                if not patch:
                    return ToolResult(success=False, output="", error="'patch' required for diff mode")
                
                modified = self._apply_unified_diff(original, patch)
                if modified is None:
                    return ToolResult(success=False, output="", error="Failed to apply patch")
            
            else:
                return ToolResult(success=False, output="", error=f"Unknown mode: {mode}")
            
            # Generate preview diff
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            ))
            diff_preview = "".join(diff_lines)
            
            if not diff_lines:
                return ToolResult(
                    success=True,
                    output="No changes detected.",
                    data={"changed": False},
                )
            
            if dry_run:
                return ToolResult(
                    success=True,
                    output=f"[DRY RUN] Would apply:\n{diff_preview}",
                    data={"changed": True, "dry_run": True, "diff": diff_preview},
                )
            
            # Apply the patch
            path.write_text(modified, encoding="utf-8")
            
            return ToolResult(
                success=True,
                output=f"✓ Patched {path.name}\n{diff_preview}",
                data={"changed": True, "diff": diff_preview},
            )
            
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _apply_unified_diff(self, original: str, patch: str) -> str | None:
        """Apply a unified diff to original content."""
        try:
            original_lines = original.splitlines(keepends=True)
            result_lines = original_lines.copy()
            
            # Parse patch hunks
            hunk_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
            
            lines = patch.splitlines(keepends=True)
            offset = 0
            i = 0
            
            while i < len(lines):
                line = lines[i]
                match = hunk_pattern.match(line)
                
                if match:
                    old_start = int(match.group(1)) - 1
                    hunk_lines = []
                    i += 1
                    
                    while i < len(lines) and not lines[i].startswith('@@'):
                        hunk_lines.append(lines[i])
                        i += 1
                    
                    # Apply hunk
                    new_lines = []
                    j = old_start + offset
                    
                    for hunk_line in hunk_lines:
                        if hunk_line.startswith('-'):
                            # Remove line
                            if j < len(result_lines):
                                result_lines.pop(j)
                                offset -= 1
                        elif hunk_line.startswith('+'):
                            # Add line
                            result_lines.insert(j, hunk_line[1:])
                            j += 1
                            offset += 1
                        elif hunk_line.startswith(' '):
                            # Context line
                            j += 1
                else:
                    i += 1
            
            return "".join(result_lines)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Diff Reviewer Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class DiffReviewerTool(Tool):
    name = "diff_reviewer"
    description = "Review code changes for correctness, style, and potential issues"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "File to review changes for"},
            "original": {"type": "string", "description": "Original content"},
            "modified": {"type": "string", "description": "Modified content"},
            "language": {"type": "string", "description": "Programming language"},
        },
        "required": ["original", "modified"],
    }

    def execute(
        self,
        original: str,
        modified: str,
        file_path: str = "",
        language: str = "",
    ) -> ToolResult:
        try:
            issues: list[dict[str, Any]] = []
            suggestions: list[str] = []
            
            # Detect language
            if not language and file_path:
                ext = Path(file_path).suffix.lower()
                lang_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".java": "java", ".rs": "rust", ".go": "go", ".rb": "ruby",
                }
                language = lang_map.get(ext, "")
            
            # Generate diff
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                lineterm="",
            ))
            
            added_lines = [l[1:] for l in diff_lines if l.startswith('+') and not l.startswith('+++')]
            removed_lines = [l[1:] for l in diff_lines if l.startswith('-') and not l.startswith('---')]
            
            # Check for common issues
            for line_num, line in enumerate(added_lines, 1):
                # Check for debug statements
                if re.search(r'\b(print|console\.log|debugger|breakpoint)\s*\(', line):
                    issues.append({
                        "type": "warning",
                        "message": f"Debug statement found: {line.strip()[:50]}",
                        "line": line_num,
                    })
                
                # Check for TODO/FIXME
                if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', line, re.IGNORECASE):
                    issues.append({
                        "type": "info",
                        "message": f"TODO marker found: {line.strip()[:50]}",
                        "line": line_num,
                    })
                
                # Check for long lines
                if len(line) > 120:
                    issues.append({
                        "type": "style",
                        "message": f"Line exceeds 120 characters ({len(line)} chars)",
                        "line": line_num,
                    })
                
                # Check for trailing whitespace
                if line.rstrip() != line.rstrip('\n\r'):
                    issues.append({
                        "type": "style",
                        "message": "Trailing whitespace",
                        "line": line_num,
                    })
            
            # Python-specific checks
            if language == "python":
                issues.extend(self._check_python(modified))
            
            # Analyze change statistics
            stats = {
                "lines_added": len(added_lines),
                "lines_removed": len(removed_lines),
                "net_change": len(added_lines) - len(removed_lines),
            }
            
            # Build output
            output_parts = ["📝 Diff Review", "─" * 40, ""]
            
            output_parts.append(f"Lines added: {stats['lines_added']}")
            output_parts.append(f"Lines removed: {stats['lines_removed']}")
            output_parts.append(f"Net change: {stats['net_change']:+d}")
            output_parts.append("")
            
            if issues:
                output_parts.append("Issues Found:")
                for issue in issues:
                    icon = {"warning": "⚠", "info": "ℹ", "style": "🎨", "error": "✗"}.get(issue["type"], "•")
                    output_parts.append(f"  {icon} {issue['message']}")
                output_parts.append("")
            else:
                output_parts.append("✓ No issues found")
            
            severity = "clean"
            if any(i["type"] == "error" for i in issues):
                severity = "error"
            elif any(i["type"] == "warning" for i in issues):
                severity = "warning"
            elif issues:
                severity = "info"
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={
                    "issues": issues,
                    "stats": stats,
                    "severity": severity,
                },
            )
            
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _check_python(self, code: str) -> list[dict[str, Any]]:
        """Python-specific checks."""
        issues = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "error",
                "message": f"Syntax error: {e.msg}",
                "line": e.lineno or 0,
            })
        
        # Check for bare except
        if re.search(r'\bexcept\s*:', code):
            issues.append({
                "type": "warning",
                "message": "Bare 'except:' clause - consider catching specific exceptions",
            })
        
        # Check for mutable default arguments
        if re.search(r'def\s+\w+\([^)]*=\s*(\[\]|\{\}|\set\(\))', code):
            issues.append({
                "type": "warning",
                "message": "Mutable default argument detected",
            })
        
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# Security Scan Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class SecurityScanTool(Tool):
    name = "security_scan"
    description = "Detect security vulnerabilities and unsafe patterns in code"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "File to scan"},
            "content": {"type": "string", "description": "Code content to scan (if no file_path)"},
            "language": {"type": "string", "description": "Programming language"},
        },
        "required": [],
    }

    # Security patterns by category
    PATTERNS = {
        "secrets": [
            (r'(?i)(api[_-]?key|secret[_-]?key|password|passwd|token)\s*[=:]\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
            (r'(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "Private key in code"),
            (r'(?i)(aws|amazon).*["\'][A-Z0-9]{20}["\']', "Possible AWS access key"),
        ],
        "injection": [
            (r'exec\s*\(', "Use of exec() - potential code injection"),
            (r'eval\s*\(', "Use of eval() - potential code injection"),
            (r'os\.system\s*\(', "Use of os.system() - prefer subprocess"),
            (r'shell\s*=\s*True', "shell=True in subprocess - potential command injection"),
            (r'__import__\s*\(', "Dynamic import - potential code injection"),
        ],
        "sql": [
            (r'["\'].*\%s.*["\'].*%', "Possible SQL injection via string formatting"),
            (r'f["\'].*SELECT.*\{', "Possible SQL injection via f-string"),
            (r'\.format\([^)]*\).*(?:SELECT|INSERT|UPDATE|DELETE)', "Possible SQL injection via .format()"),
        ],
        "path_traversal": [
            (r'\.\./', "Path traversal pattern"),
            (r'open\s*\([^)]*\+', "Dynamic file path in open()"),
        ],
        "crypto": [
            (r'(?i)\b(md5|sha1)\s*\(', "Weak hash function (MD5/SHA1)"),
            (r'(?i)random\s*\(', "Non-cryptographic random - use secrets module"),
        ],
        "deserialization": [
            (r'pickle\.loads?\s*\(', "Unsafe pickle deserialization"),
            (r'yaml\.load\s*\([^)]*\)', "Unsafe YAML load - use safe_load"),
            (r'marshal\.loads?\s*\(', "Unsafe marshal deserialization"),
        ],
        "xss": [
            (r'innerHTML\s*=', "innerHTML assignment - potential XSS"),
            (r'document\.write\s*\(', "document.write() - potential XSS"),
            (r'\|safe\b', "Django safe filter - ensure input is sanitized"),
        ],
    }

    def execute(
        self,
        file_path: str = "",
        content: str = "",
        language: str = "",
    ) -> ToolResult:
        try:
            # Get content
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    return ToolResult(success=False, output="", error=f"File not found: {file_path}")
                content = path.read_text(encoding="utf-8")
                
                # Detect language
                if not language:
                    ext = path.suffix.lower()
                    lang_map = {
                        ".py": "python", ".js": "javascript", ".ts": "typescript",
                        ".java": "java", ".rb": "ruby", ".php": "php",
                    }
                    language = lang_map.get(ext, "")
            
            if not content:
                return ToolResult(success=False, output="", error="No content to scan")
            
            findings: list[dict[str, Any]] = []
            lines = content.split('\n')
            
            for category, patterns in self.PATTERNS.items():
                for pattern, description in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            findings.append({
                                "category": category,
                                "severity": self._get_severity(category),
                                "line": line_num,
                                "description": description,
                                "code": line.strip()[:80],
                            })
            
            # Build output
            output_parts = ["🔒 Security Scan Results", "─" * 40, ""]
            
            if file_path:
                output_parts.append(f"File: {file_path}")
                output_parts.append("")
            
            if findings:
                # Group by severity
                by_severity: dict[str, list] = {"critical": [], "high": [], "medium": [], "low": []}
                for f in findings:
                    by_severity[f["severity"]].append(f)
                
                severity_icons = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🔵",
                }
                
                for severity in ["critical", "high", "medium", "low"]:
                    items = by_severity[severity]
                    if items:
                        output_parts.append(f"{severity_icons[severity]} {severity.upper()} ({len(items)})")
                        for item in items:
                            output_parts.append(f"  Line {item['line']}: {item['description']}")
                            output_parts.append(f"    └─ {item['code']}")
                        output_parts.append("")
                
                total = len(findings)
                output_parts.append(f"Total findings: {total}")
            else:
                output_parts.append("✓ No security issues detected")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={
                    "findings": findings,
                    "total": len(findings),
                    "by_category": {
                        cat: len([f for f in findings if f["category"] == cat])
                        for cat in self.PATTERNS.keys()
                    },
                },
            )
            
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _get_severity(self, category: str) -> str:
        """Map category to severity."""
        severity_map = {
            "secrets": "critical",
            "injection": "critical",
            "sql": "critical",
            "deserialization": "high",
            "xss": "high",
            "path_traversal": "medium",
            "crypto": "medium",
        }
        return severity_map.get(category, "low")


# ─────────────────────────────────────────────────────────────────────────────
# Run Tests Tool
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class RunTestsTool(Tool):
    name = "run_tests"
    description = "Execute test suite and return results"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to test file or directory"},
            "framework": {
                "type": "string",
                "enum": ["auto", "pytest", "unittest", "jest", "mocha", "go"],
                "description": "Test framework to use",
            },
            "pattern": {"type": "string", "description": "Test name pattern to match"},
            "verbose": {"type": "boolean", "description": "Verbose output"},
        },
        "required": ["path"],
    }

    def execute(
        self,
        path: str,
        framework: str = "auto",
        pattern: str = "",
        verbose: bool = False,
    ) -> ToolResult:
        try:
            import subprocess
            
            test_path = Path(path)
            
            if not test_path.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}")
            
            # Auto-detect framework
            if framework == "auto":
                framework = self._detect_framework(test_path)
            
            # Build command
            cmd = self._build_command(framework, path, pattern, verbose)
            
            if not cmd:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown test framework: {framework}",
                )
            
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=test_path.parent if test_path.is_file() else test_path,
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            # Parse results
            stats = self._parse_results(output, framework)
            
            # Format output
            status = "✓ PASSED" if success else "✗ FAILED"
            output_parts = [
                f"🧪 Test Results: {status}",
                "─" * 40,
                "",
                f"Framework: {framework}",
                f"Path: {path}",
                "",
            ]
            
            if stats:
                output_parts.append(f"Passed: {stats.get('passed', 0)}")
                output_parts.append(f"Failed: {stats.get('failed', 0)}")
                output_parts.append(f"Skipped: {stats.get('skipped', 0)}")
                output_parts.append("")
            
            output_parts.append("Output:")
            output_parts.append(output[-2000:] if len(output) > 2000 else output)
            
            return ToolResult(
                success=success,
                output="\n".join(output_parts),
                data={
                    "passed": success,
                    "stats": stats,
                    "exit_code": result.returncode,
                },
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Test execution timed out (5 minutes)")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
    
    def _detect_framework(self, path: Path) -> str:
        """Auto-detect test framework."""
        if path.is_file():
            check_path = path.parent
        else:
            check_path = path
        
        # Check for pytest
        if (check_path / "pytest.ini").exists() or (check_path / "pyproject.toml").exists():
            return "pytest"
        
        # Check for jest
        if (check_path / "jest.config.js").exists() or (check_path / "jest.config.ts").exists():
            return "jest"
        
        # Check for Go
        if any(check_path.glob("*_test.go")):
            return "go"
        
        # Default to pytest for Python
        if path.suffix == ".py" or any(check_path.glob("*.py")):
            return "pytest"
        
        return "pytest"
    
    def _build_command(
        self,
        framework: str,
        path: str,
        pattern: str,
        verbose: bool,
    ) -> list[str] | None:
        """Build test command."""
        if framework == "pytest":
            cmd = ["python", "-m", "pytest", path]
            if pattern:
                cmd.extend(["-k", pattern])
            if verbose:
                cmd.append("-v")
            return cmd
        
        elif framework == "unittest":
            cmd = ["python", "-m", "unittest"]
            if Path(path).is_file():
                cmd.append(path)
            else:
                cmd.extend(["discover", "-s", path])
            if verbose:
                cmd.append("-v")
            return cmd
        
        elif framework == "jest":
            cmd = ["npx", "jest", path]
            if pattern:
                cmd.extend(["--testNamePattern", pattern])
            if verbose:
                cmd.append("--verbose")
            return cmd
        
        elif framework == "mocha":
            cmd = ["npx", "mocha", path]
            if pattern:
                cmd.extend(["--grep", pattern])
            return cmd
        
        elif framework == "go":
            cmd = ["go", "test"]
            if verbose:
                cmd.append("-v")
            cmd.append(path)
            return cmd
        
        return None
    
    def _parse_results(self, output: str, framework: str) -> dict[str, int]:
        """Parse test results from output."""
        stats = {"passed": 0, "failed": 0, "skipped": 0}
        
        if framework == "pytest":
            match = re.search(r'(\d+) passed', output)
            if match:
                stats["passed"] = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                stats["failed"] = int(match.group(1))
            match = re.search(r'(\d+) skipped', output)
            if match:
                stats["skipped"] = int(match.group(1))
        
        elif framework == "jest":
            match = re.search(r'Tests:\s+(\d+) passed', output)
            if match:
                stats["passed"] = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                stats["failed"] = int(match.group(1))
        
        return stats
