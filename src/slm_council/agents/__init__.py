"""Agents sub-package – specialised SLM workers.

Importing this module triggers agent registration via @registry.register
decorators.  The registry can then be used to discover and instantiate
agents dynamically by role.
"""

from slm_council.agents.base import BaseAgent
from slm_council.agents.registry import registry
from slm_council.agents.researcher import TechResearcherAgent
from slm_council.agents.planner import PlannerAgent
from slm_council.agents.generator import CodeGeneratorAgent
from slm_council.agents.reviewer import ReviewerAgent
from slm_council.agents.debugger import DebuggerAgent
from slm_council.agents.tester import TesterAgent
from slm_council.agents.optimizer import OptimizerAgent
from slm_council.agents.refactorer import RefactorerAgent

__all__ = [
    "BaseAgent",
    "registry",
    "TechResearcherAgent",
    "PlannerAgent",
    "CodeGeneratorAgent",
    "ReviewerAgent",
    "DebuggerAgent",
    "TesterAgent",
    "OptimizerAgent",
    "RefactorerAgent",
]
