"""Code Generator agent – Qwen3-Coder 4B.

Consumes the Tech Manifest and writes clean, type-hinted,
production-quality code files.
"""

from __future__ import annotations

import json
import re
from typing import Any

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.models import AgentRole, CodeFile, GeneratedCode
from slm_council.utils.logging import get_logger
from slm_council.utils.prompts import GENERATOR_SYSTEM, GENERATOR_TASK

logger = get_logger(__name__)


@registry.register(AgentRole.GENERATOR)
class CodeGeneratorAgent(BaseAgent):
    role = AgentRole.GENERATOR
    system_prompt = GENERATOR_SYSTEM

    def build_prompt(self, task_instruction: str, **ctx: Any) -> str:
        tech_manifest = ctx.get("tech_manifest", "{}")
        if not isinstance(tech_manifest, str):
            tech_manifest = json.dumps(tech_manifest, indent=2)

        refinement_context = ""
        if feedback := ctx.get("refinement_feedback"):
            refinement_context = (
                f"**Refinement feedback from Orchestrator (pass {ctx.get('pass_number', '?')}):**\n"
                f"{feedback}\n\n"
                "Fix the issues listed above and regenerate the code."
            )

        architecture_plan = ctx.get("architecture_plan", "N/A")
        if not isinstance(architecture_plan, str):
            architecture_plan = json.dumps(architecture_plan, indent=2)

        return GENERATOR_TASK.format(
            instruction=task_instruction,
            tech_manifest=tech_manifest,
            architecture_plan=architecture_plan,
            refinement_context=refinement_context,
        )

    def parse_output(self, raw: str) -> GeneratedCode:
        try:
            data = self._extract_json(raw)
            files = [CodeFile(**f) for f in data.get("files", [])]
            if files:
                return GeneratedCode(
                    files=files,
                    explanation=data.get("explanation", ""),
                    assumptions=data.get("assumptions", []),
                )
        except Exception:
            pass

        try:
            fixed = self._fix_triple_quotes(raw)
            data = self._extract_json(fixed)
            files = [CodeFile(**f) for f in data.get("files", [])]
            if files:
                logger.info("generator.parse_recovered", method="triple_quote_fix")
                return GeneratedCode(
                    files=files,
                    explanation=data.get("explanation", ""),
                    assumptions=data.get("assumptions", []),
                )
        except Exception:
            pass

        files = self._regex_extract_files(raw)
        if files:
            logger.info("generator.parse_recovered", method="regex_extraction", file_count=len(files))
            return GeneratedCode(
                files=files,
                explanation="Recovered from malformed JSON via regex extraction.",
                assumptions=["Output was not valid JSON; code extracted heuristically."],
            )

        logger.warning("generator.parse_failed", raw_preview=raw[:300])
        return GeneratedCode(
            files=[],
            explanation="Generator returned non-JSON or malformed JSON output.",
            assumptions=[f"Raw output preview: {raw[:500]}..." if len(raw) > 500 else raw],
        )


    @staticmethod
    def _fix_triple_quotes(text: str) -> str:
        """Replace unescaped triple-quotes with properly escaped JSON quotes.

        Small code models often emit Python triple-quotes inside JSON string
        values.  Two variants observed:
          - Opening:  \"\"\"  (three unescaped double-quotes)
          - Closing:  ""\\\"  (two unescaped + one already-escaped quote)

        Strategy: textually replace both patterns with \\\"\\\"\\\" so the
        surrounding JSON becomes valid, then the caller re-parses.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        cleaned = cleaned.replace('"""', '\\"\\"\\"')
        cleaned = cleaned.replace('""\\"', '\\"\\"\\"')

        return cleaned

    @staticmethod
    def _regex_extract_files(raw: str) -> list[CodeFile]:
        """Extract code files from raw output using heuristic patterns.

        Works even when JSON is completely broken by:
        1. Finding "filename" values, then extracting the "content" field text
           between known JSON field boundaries.
        2. Falling back to markdown code blocks if no filenames are found.
        """
        files: list[CodeFile] = []

        filename_pattern = re.compile(
            r'"filename"\s*:\s*"([^"]+)"', re.IGNORECASE
        )
        filenames = filename_pattern.findall(raw)

        if filenames:
            for fname in filenames:
                fname_pos = raw.find(f'"{fname}"')
                if fname_pos == -1:
                    continue

                content_key = raw.find('"content"', fname_pos)
                if content_key == -1:
                    continue

                colon_pos = raw.find(":", content_key + 9)
                if colon_pos == -1:
                    continue
                val_start = raw.find('"', colon_pos)
                if val_start == -1:
                    continue

                best_end = -1
                for end_marker in (
                    '",\n      "description"',
                    '",\n      "language"',
                    '"\n    }',
                    '",\n    }',
                    '"\n  }',
                ):
                    pos = raw.find(end_marker, val_start + 1)
                    if pos != -1 and (best_end == -1 or pos < best_end):
                        best_end = pos

                if best_end == -1:
                    continue

                code_raw = raw[val_start + 1 : best_end]

                code = (
                    code_raw.replace("\\\\", "\x00")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\'", "'")
                    .replace("\x00", "\\")
                )
                code = code.strip()

                if code:
                    lang = _guess_language(fname)
                    files.append(
                        CodeFile(
                            filename=fname,
                            language=lang,
                            content=code,
                            description="Extracted from generator output (regex)",
                        )
                    )

        if not files:
            md_blocks = re.findall(
                r"```(\w+)\n(.*?)```", raw, re.DOTALL
            )
            for lang, code in md_blocks:
                code = code.strip()
                if code and lang.lower() not in ("json",):
                    ext = {
                        "python": ".py", "py": ".py",
                        "javascript": ".js", "js": ".js",
                        "typescript": ".ts", "ts": ".ts",
                        "java": ".java", "go": ".go", "rust": ".rs",
                        "c": ".c", "cpp": ".cpp",
                    }.get(lang.lower(), f".{lang}")
                    files.append(
                        CodeFile(
                            filename=f"main{ext}",
                            language=lang,
                            content=code,
                            description="Extracted from markdown code block",
                        )
                    )

        return files


def _guess_language(filename: str) -> str:
    """Guess the programming language from a filename."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".c": "c",
        ".cpp": "cpp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    }
    for ext, lang in ext_map.items():
        if filename.endswith(ext):
            return lang
    return "text"
