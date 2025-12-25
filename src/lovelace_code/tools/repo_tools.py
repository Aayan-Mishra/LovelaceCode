"""Repository analysis and semantic search tools."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .registry import register_tool


# ─────────────────────────────────────────────────────────────────────────────
# Repository Index Storage
# ─────────────────────────────────────────────────────────────────────────────
class RepoIndex:
    """In-memory repository index with structural and semantic information."""
    
    def __init__(self):
        self.files: dict[str, dict[str, Any]] = {}
        self.symbols: dict[str, list[dict[str, Any]]] = {}  # symbol -> locations
        self.embeddings: dict[str, list[float]] = {}  # chunk_id -> embedding
        self.chunks: dict[str, str] = {}  # chunk_id -> content
        self.dependencies: dict[str, set[str]] = {}  # file -> imported files
        self.root_path: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "symbols": self.symbols,
            "chunks": self.chunks,
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "root_path": self.root_path,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoIndex":
        index = cls()
        index.files = data.get("files", {})
        index.symbols = data.get("symbols", {})
        index.chunks = data.get("chunks", {})
        index.dependencies = {k: set(v) for k, v in data.get("dependencies", {}).items()}
        index.root_path = data.get("root_path", "")
        return index


# Global index storage
_repo_index: RepoIndex | None = None


def get_repo_index() -> RepoIndex:
    global _repo_index
    if _repo_index is None:
        _repo_index = RepoIndex()
    return _repo_index


# ─────────────────────────────────────────────────────────────────────────────
# File Type Detection
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
}

IGNORE_PATTERNS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", "*.egg-info",
    ".idea", ".vscode", ".DS_Store", "*.pyc", "*.pyo", "*.so", "*.dylib",
}


def should_ignore(path: Path) -> bool:
    """Check if a path should be ignored."""
    for part in path.parts:
        for pattern in IGNORE_PATTERNS:
            if "*" in pattern:
                if path.match(pattern):
                    return True
            elif part == pattern:
                return True
    return False


def get_language(path: Path) -> str | None:
    """Get the language for a file based on extension."""
    return LANGUAGE_EXTENSIONS.get(path.suffix.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Python AST Analysis
# ─────────────────────────────────────────────────────────────────────────────
def extract_python_symbols(content: str, file_path: str) -> list[dict[str, Any]]:
    """Extract symbols (classes, functions, etc.) from Python code."""
    symbols = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append({
                    "name": node.name,
                    "type": "class",
                    "file": file_path,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "",
                })
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.append({
                    "name": node.name,
                    "type": "function",
                    "file": file_path,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "",
                })
    except SyntaxError:
        pass
    return symbols


def extract_python_imports(content: str) -> list[str]:
    """Extract imports from Python code."""
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])
    except SyntaxError:
        pass
    return imports


# ─────────────────────────────────────────────────────────────────────────────
# Generic Symbol Extraction (regex-based for other languages)
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL_PATTERNS = {
    "javascript": [
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"(?:export\s+)?class\s+(\w+)", "class"),
        (r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
    ],
    "typescript": [
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"(?:export\s+)?class\s+(\w+)", "class"),
        (r"(?:export\s+)?interface\s+(\w+)", "interface"),
        (r"(?:export\s+)?type\s+(\w+)", "type"),
        (r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
    ],
    "java": [
        (r"(?:public|private|protected)?\s*class\s+(\w+)", "class"),
        (r"(?:public|private|protected)?\s*interface\s+(\w+)", "interface"),
        (r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(", "method"),
    ],
    "go": [
        (r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "function"),
        (r"type\s+(\w+)\s+struct", "struct"),
        (r"type\s+(\w+)\s+interface", "interface"),
    ],
    "rust": [
        (r"(?:pub\s+)?fn\s+(\w+)", "function"),
        (r"(?:pub\s+)?struct\s+(\w+)", "struct"),
        (r"(?:pub\s+)?trait\s+(\w+)", "trait"),
        (r"(?:pub\s+)?enum\s+(\w+)", "enum"),
        (r"impl(?:<[^>]+>)?\s+(\w+)", "impl"),
    ],
}


def extract_generic_symbols(content: str, language: str, file_path: str) -> list[dict[str, Any]]:
    """Extract symbols using regex patterns."""
    symbols = []
    patterns = SYMBOL_PATTERNS.get(language, [])
    lines = content.splitlines()
    
    for pattern, symbol_type in patterns:
        for match in re.finditer(pattern, content):
            name = match.group(1)
            # Find line number
            pos = match.start()
            line_num = content[:pos].count("\n") + 1
            symbols.append({
                "name": name,
                "type": symbol_type,
                "file": file_path,
                "line": line_num,
            })
    
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# Chunking for Semantic Search
# ─────────────────────────────────────────────────────────────────────────────
def chunk_content_generator(content: str, file_path: str, chunk_size: int = 1000) -> list[tuple[str, str]]:
    """Split content into chunks for semantic search."""
    chunks = []
    lines = content.splitlines(keepends=True)
    
    current_chunk = []
    current_size = 0
    chunk_start = 1
    
    for i, line in enumerate(lines, 1):
        current_chunk.append(line)
        current_size += len(line)
        
        if current_size >= chunk_size:
            current_chunk_content = "".join(current_chunk)
            chunk_id = f"{file_path}:{chunk_start}-{i}"
            chunks.append((chunk_id, current_chunk_content))
            current_chunk = []
            current_size = 0
            chunk_start = i + 1
    
    # Remaining content
    if current_chunk:
        current_chunk_content = "".join(current_chunk)
        chunk_id = f"{file_path}:{chunk_start}-{len(lines)}"
        chunks.append((chunk_id, current_chunk_content))
    
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Simple TF-IDF based similarity (no external deps)
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Simple tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_tfidf_similarity(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    """Compute TF-IDF similarity between query and documents."""
    query_tokens = set(tokenize(query))
    
    # Document frequencies
    doc_freq: dict[str, int] = {}
    doc_tokens: dict[str, set[str]] = {}
    
    for doc_id, content in documents.items():
        tokens = set(tokenize(content))
        doc_tokens[doc_id] = tokens
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    
    # Score documents
    scores = []
    num_docs = len(documents)
    
    for doc_id, tokens in doc_tokens.items():
        score = 0.0
        for token in query_tokens:
            if token in tokens:
                # TF-IDF score
                tf = 1  # Binary TF
                idf = num_docs / (doc_freq.get(token, 1) + 1)
                score += tf * idf
        if score > 0:
            scores.append((doc_id, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────
@register_tool
class RepoIndexerTool(Tool):
    name = "repo_indexer"
    description = "Scan a repository and build a structural + semantic map of the codebase"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the repository root"},
            "max_files": {"type": "integer", "description": "Maximum files to index (default: 1000)"},
        },
        "required": ["path"],
    }

    def execute(self, path: str, max_files: int = 1000) -> ToolResult:
        try:
            root = Path(path).resolve()
            if not root.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}")
            
            index = get_repo_index()
            index.root_path = str(root)
            index.files.clear()
            index.symbols.clear()
            index.chunks.clear()
            index.dependencies.clear()
            
            files_indexed = 0
            symbols_found = 0
            
            for file_path in root.rglob("*"):
                if files_indexed >= max_files:
                    break
                    
                if not file_path.is_file():
                    continue
                if should_ignore(file_path):
                    continue
                
                rel_path = str(file_path.relative_to(root))
                language = get_language(file_path)
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                
                # Store file info
                index.files[rel_path] = {
                    "path": rel_path,
                    "language": language,
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                }
                
                # Extract symbols
                if language == "python":
                    symbols = extract_python_symbols(content, rel_path)
                    imports = extract_python_imports(content)
                    index.dependencies[rel_path] = set(imports)
                elif language:
                    symbols = extract_generic_symbols(content, language, rel_path)
                else:
                    symbols = []
                
                for sym in symbols:
                    sym_name = sym["name"]
                    if sym_name not in index.symbols:
                        index.symbols[sym_name] = []
                    index.symbols[sym_name].append(sym)
                    symbols_found += 1
                
                # Create chunks for semantic search
                for chunk_id, chunk in chunk_content_generator(content, rel_path):
                    index.chunks[chunk_id] = chunk
                
                files_indexed += 1
            
            summary = f"Indexed {files_indexed} files, found {symbols_found} symbols, created {len(index.chunks)} searchable chunks"
            
            return ToolResult(
                success=True,
                output=summary,
                data={
                    "files_indexed": files_indexed,
                    "symbols_found": symbols_found,
                    "chunks_created": len(index.chunks),
                    "languages": list(set(f.get("language") for f in index.files.values() if f.get("language"))),
                },
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


@register_tool
class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = "Retrieve relevant code snippets using semantic matching"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"},
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            index = get_repo_index()
            
            if not index.chunks:
                return ToolResult(
                    success=False,
                    output="",
                    error="No repository indexed. Run repo_indexer first.",
                )
            
            # Compute similarities
            results = compute_tfidf_similarity(query, index.chunks)[:max_results]
            
            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant results found.",
                    data={"results": []},
                )
            
            # Format results
            output_parts = []
            result_data = []
            
            for chunk_id, score in results:
                content = index.chunks[chunk_id]
                # Parse chunk_id for file:lines format
                file_path, line_range = chunk_id.rsplit(":", 1)
                
                output_parts.append(f"── {file_path} (lines {line_range}) ──")
                output_parts.append(content[:500] + "..." if len(content) > 500 else content)
                output_parts.append("")
                
                result_data.append({
                    "file": file_path,
                    "lines": line_range,
                    "score": round(score, 3),
                    "preview": content[:200],
                })
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={"results": result_data},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


@register_tool
class DependencyTracerTool(Tool):
    name = "dependency_tracer"
    description = "Identify what files and modules are affected by a change"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to trace dependencies for"},
        },
        "required": ["path"],
    }

    def execute(self, path: str) -> ToolResult:
        try:
            index = get_repo_index()
            
            if not index.files:
                return ToolResult(
                    success=False,
                    output="",
                    error="No repository indexed. Run repo_indexer first.",
                )
            
            # Normalize path
            target_path = Path(path)
            if target_path.is_absolute() and index.root_path:
                try:
                    target_path = target_path.relative_to(index.root_path)
                except ValueError:
                    pass
            target = str(target_path)
            
            # Find files that depend on this file
            dependents = []
            for file_path, deps in index.dependencies.items():
                # Check if target module is imported
                target_module = Path(target).stem
                if target_module in deps or target in deps:
                    dependents.append(file_path)
            
            # Find files this file depends on
            dependencies = list(index.dependencies.get(target, set()))
            
            # Find symbols defined in this file
            symbols_in_file = []
            for sym_name, locations in index.symbols.items():
                for loc in locations:
                    if loc.get("file") == target:
                        symbols_in_file.append({
                            "name": sym_name,
                            "type": loc.get("type"),
                            "line": loc.get("line"),
                        })
            
            output_parts = [f"Dependency analysis for: {target}", ""]
            
            if symbols_in_file:
                output_parts.append(f"Symbols defined ({len(symbols_in_file)}):")
                for sym in symbols_in_file[:10]:
                    output_parts.append(f"  • {sym['type']} {sym['name']} (line {sym['line']})")
                output_parts.append("")
            
            if dependencies:
                output_parts.append(f"Imports/Dependencies ({len(dependencies)}):")
                for dep in dependencies[:10]:
                    output_parts.append(f"  → {dep}")
                output_parts.append("")
            
            if dependents:
                output_parts.append(f"Files that import this ({len(dependents)}):")
                for dep in dependents[:10]:
                    output_parts.append(f"  ← {dep}")
            else:
                output_parts.append("No files depend on this file (or not detected).")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                data={
                    "file": target,
                    "symbols": symbols_in_file,
                    "dependencies": dependencies,
                    "dependents": dependents,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
