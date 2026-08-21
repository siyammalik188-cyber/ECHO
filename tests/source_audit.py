"""Read a module's *code* without its prose, for tests that audit source.

Several ECHO tests assert that something does not appear in a module — a
hard-coded answer, a simulated feeling, an import across a wall. Grepping the
raw file cannot do that job: a docstring saying "there is no method that returns
*I feel uncertain*" contains the very string the test is looking for, and the
test fails on the sentence that promises the opposite of the violation.

So docstrings and comments are stripped and everything else is kept. Ordinary
string literals stay, deliberately: `if answer == "I feel uncertain"` is exactly
the kind of cheat these audits exist to catch, and dropping all strings would
hide it.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path


def code_without_docs(path: Path | str) -> str:
    """Source with docstrings and comments removed, all other code intact."""
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)

    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            doc_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string

    kept: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if number in doc_lines:
            continue
        if number in comments:
            line = line.replace(comments[number], "")
        kept.append(line)
    return "\n".join(kept)


def imports_of(path: Path | str) -> list[str]:
    """Every import statement in a module, as stripped source lines."""
    return [
        line.strip()
        for line in code_without_docs(path).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
