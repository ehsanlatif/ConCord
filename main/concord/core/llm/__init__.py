from .client import LLMClient
from .factory import RoleClients, make_client
from .mock import MockLLM

# AnthropicAdapter is imported lazily — it imports `anthropic` which is fine
# but we want the package to be usable in CI without the dep loaded eagerly.
def _lazy_anthropic():
    from .anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter

__all__ = ["LLMClient", "MockLLM", "RoleClients", "make_client",
           "_lazy_anthropic"]
