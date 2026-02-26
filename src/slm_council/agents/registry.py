"""Agent Registry – plugin-style discovery and instantiation.

Agents self-register via the ``@registry.register`` decorator so that
the orchestrator and loop can discover available agents at runtime
without hard-coded imports.  This makes adding new specialist agents
a one-file operation.

Usage in an agent module::

    from slm_council.agents.registry import registry
    from slm_council.models import AgentRole

    @registry.register(AgentRole.MY_NEW_AGENT)
    class MyNewAgent(BaseAgent):
        ...

Usage from the loop / orchestrator::

    agent_cls = registry.get(AgentRole.GENERATOR)
    agent = registry.create(
        AgentRole.GENERATOR,
        endpoint="https://...",
        model="qwen3-coder",
        api_key="sk-...",
    )
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from slm_council.models import AgentRole
from slm_council.utils.logging import get_logger

if TYPE_CHECKING:
    from slm_council.agents.base import BaseAgent

logger = get_logger(__name__)


class AgentRegistry:
    """Singleton registry mapping ``AgentRole`` → agent class."""

    def __init__(self) -> None:
        self._agents: dict[AgentRole, type[BaseAgent]] = {}


    def register(self, role: AgentRole):  # type: ignore[no-untyped-def]
        """Class decorator that registers an agent under *role*."""

        def wrapper(cls: type[BaseAgent]) -> type[BaseAgent]:
            if role in self._agents:
                logger.warning(
                    "registry.overwrite",
                    role=role.value,
                    old=self._agents[role].__name__,
                    new=cls.__name__,
                )
            self._agents[role] = cls
            return cls

        return wrapper

    def register_class(self, role: AgentRole, cls: type[BaseAgent]) -> None:
        """Imperative registration (alternative to decorator)."""
        self._agents[role] = cls


    def get(self, role: AgentRole) -> type[BaseAgent] | None:
        """Return the agent class for *role*, or ``None``."""
        return self._agents.get(role)

    def available_roles(self) -> list[AgentRole]:
        """Return the roles that have registered agents."""
        return list(self._agents.keys())

    def has(self, role: AgentRole) -> bool:
        return role in self._agents


    def create(
        self,
        role: AgentRole,
        *,
        endpoint: str,
        model: str,
        api_key: str = "",
    ) -> BaseAgent:
        """Instantiate the agent for *role* with the given transport settings."""
        cls = self._agents.get(role)
        if cls is None:
            raise ValueError(
                f"No agent registered for role {role.value!r}. "
                f"Available: {[r.value for r in self._agents]}"
            )
        return cls(endpoint=endpoint, model=model, api_key=api_key)

    def create_all(
        self,
        configs: dict[AgentRole, dict[str, str]],
    ) -> dict[AgentRole, BaseAgent]:
        """Batch-create agents from a config mapping.

        *configs* maps ``AgentRole`` → ``{"endpoint": ..., "model": ..., "api_key": ...}``.
        Only roles that are both registered AND present in configs are created.
        """
        agents: dict[AgentRole, BaseAgent] = {}
        for role, cfg in configs.items():
            if self.has(role):
                agents[role] = self.create(
                    role,
                    endpoint=cfg.get("endpoint", ""),
                    model=cfg.get("model", ""),
                    api_key=cfg.get("api_key", ""),
                )
        return agents

    def __repr__(self) -> str:
        roles = ", ".join(r.value for r in self._agents)
        return f"AgentRegistry([{roles}])"


registry = AgentRegistry()
