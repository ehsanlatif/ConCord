"""MCTS engine with PUCT selection, progressive widening, and guarded-max backup.

Decisions §2 of the implementation plan that this module implements:
- (3) PUCT + progressive widening (C is the old beam M)
- (4) guarded-max backup with n_min visit guard
- (8) high c_puct travels with these

The CORE of the engine is provider-agnostic via two injectable interfaces:

  - `ExpansionPolicy` produces the K-samples / class / U_s output for a state.
    M3 unit tests use a synthetic policy; M5 wires it to the real confidence
    stack on top of the LLM.
  - `TerminalChecker` decides when a node is terminal (atomic, depth cap, or
    the M4 incremental gate fires).

These two seams are also where the spec's "neural prior + value" injection
would land later (§5.8).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ..config import Config
from ..types import SolutionState
from .transposition import TranspositionTable, TTKey
from .widening import may_widen


# ---------------------------------------------------------------------------
# Class summary the expansion policy returns per meaning-class
# ---------------------------------------------------------------------------

@dataclass
class ExpandedClass:
    """One PUCT child candidate produced by the expansion policy."""

    class_key: str             # stable identifier (e.g. medoid text or a hash)
    answer_text: str           # the text used to extend SolutionState
    mass: float                # weight oracle output -> PUCT prior numerator
    u_s: float                 # unified score over the class
    sigma: float = 0.0         # coherence severity (0 if not yet checked)
    terminal_hint: bool = False    # set True for assembled / atomic terminals


@dataclass
class ExpandedNodeInfo:
    """What the expansion policy returns for ONE call on a node's state."""

    classes: list[ExpandedClass]
    subproblem_text: str
    bindings: dict[str, str]


# ---------------------------------------------------------------------------
# Protocols (M3 swaps these for synthetic fixtures; M5 wires real ones)
# ---------------------------------------------------------------------------

class ExpansionPolicy(Protocol):
    def expand(self, state: SolutionState, depth: int, *,
               parent_u_s: float, K: int) -> ExpandedNodeInfo:
        """Sample, cluster, score K candidate continuations. Pure of MCTS state."""
        ...


class TerminalChecker(Protocol):
    def is_terminal(self, state: SolutionState, depth: int) -> bool:
        ...


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

_NODE_ID_COUNTER = [0]


def _next_node_id() -> str:
    """Cheap monotonic id generator. One process = one search loop normally,
    so global state is fine; reset_node_ids() lets tests start fresh.
    """
    nid = f"n{_NODE_ID_COUNTER[0]}"
    _NODE_ID_COUNTER[0] += 1
    return nid


def reset_node_ids() -> None:
    _NODE_ID_COUNTER[0] = 0


@dataclass
class Node:
    state: SolutionState
    depth: int
    parent: "Node | None" = None
    children: dict[str, "Node"] = field(default_factory=dict)
    untried: list[ExpandedClass] = field(default_factory=list)
    expanded: bool = False

    # MCTS bookkeeping
    N: int = 0
    Q: float = 0.0
    prior: float = 0.0
    U_s: float = 0.0
    sigma: float = 0.0
    terminal: bool = False
    gated_fail: bool = False

    # cached at first expansion for diagnostics
    subproblem_text: str = ""

    # stable id + provenance for tracing
    node_id: str = field(default_factory=_next_node_id)
    class_key: str | None = None       # which meaning-class this child was made from
    answer_text: str = ""              # the chosen answer that led to this child

    def is_fully_widened(self, C: float, beta: float) -> bool:
        return not may_widen(self.N, len(self.children), C, beta)

    def to_dict(self) -> dict:
        """Serialize a single node for the tree.json output (no children).

        Children are written separately via the parent-id link so the
        full structure can be reconstructed without graph traversal.
        """
        return {
            "id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "depth": self.depth,
            "N": self.N,
            "Q": round(self.Q, 6),
            "prior": round(self.prior, 6),
            "U_s": round(self.U_s, 6),
            "sigma": round(self.sigma, 6),
            "terminal": self.terminal,
            "gated_fail": self.gated_fail,
            "expanded": self.expanded,
            "n_children": len(self.children),
            "n_untried": len(self.untried),
            "subproblem_text": self.subproblem_text,
            "class_key": self.class_key,
            "answer_text": self.answer_text[:400],
            "state": {
                "resolved": list(self.state.resolved),
                "bindings": dict(self.state.bindings),
            },
        }


# ---------------------------------------------------------------------------
# Selection: PUCT descent
# ---------------------------------------------------------------------------

def select(root: Node, cfg: Config) -> Node:
    """Descend until we land on a node that wants to grow another child.

    "Wants to grow" = either the widening rule still permits a new child AND
    classes are left untried, OR the node hasn't been expanded yet. When
    neither holds, descend via PUCT.
    """
    s = root
    c_puct = cfg.mcts.c_puct
    C = cfg.mcts.C
    beta = cfg.mcts.beta
    while True:
        if s.terminal or not s.expanded:
            return s
        widen_room = (
            may_widen(s.N, len(s.children), C, beta) and bool(s.untried)
        )
        if widen_room:
            return s
        if not s.children:
            return s
        sib_total = sum(c.N for c in s.children.values())
        sqrt_sib = math.sqrt(max(1, sib_total))

        def puct_score(c: Node) -> float:
            return c.Q + c_puct * c.prior * sqrt_sib / (1 + c.N)

        s = max(s.children.values(), key=puct_score)


# ---------------------------------------------------------------------------
# Expansion: lazy K-sample + class population
# ---------------------------------------------------------------------------

def expand(node: Node, policy: ExpansionPolicy, K: int, cfg: Config,
           tt: TranspositionTable | None = None) -> Node | None:
    """Add the next child according to progressive widening.

    Two phases:
      - First call on this node: ask the policy to produce K samples,
        cluster them, store sorted-by-prior as `untried`. Mark expanded.
        (NB: this also triggers the transposition cache.)
      - Subsequent calls: pop one untried class, instantiate it as a child.

    Returns the new child node (the one to evaluate next), or None when the
    widening rule forbids adding another child.
    """
    if node.terminal:
        return None
    if not node.expanded:
        # Key on the FULL state (resolved chain + bindings). The next
        # subproblem is a deterministic function of state via the
        # decomposer — but we don't know it yet at this point in the code,
        # so we hash the state instead. State equality is the correct
        # transposition condition.
        state_sig = {
            "resolved": list(node.state.resolved),
            "bindings": dict(node.state.bindings),
            "depth": node.depth,
        }
        ttkey = TTKey.make("expand", state_sig)
        cached = tt.get(ttkey) if tt is not None else None
        if cached is not None:
            info = cached
        else:
            info = policy.expand(node.state, node.depth,
                                 parent_u_s=node.U_s, K=K)
            if tt is not None:
                tt.put(ttkey, info)
        # sort untried classes by prior = mass * U_s, descending
        node.untried = sorted(
            info.classes, key=lambda c: -(c.mass * c.u_s),
        )
        node.subproblem_text = info.subproblem_text
        node.expanded = True

    if not may_widen(node.N, len(node.children), cfg.mcts.C, cfg.mcts.beta):
        return None
    if not node.untried:
        # Spec §3 line 17 + plan §5.5: "If untried empties and more width is
        # permitted, draw K more samples (re-cluster, append new classes)."
        # Without this, K=16 with a real LLM still hard-caps each node at the
        # number of classes the first K-sample produced. We re-call the policy
        # and merge any NEW class keys we haven't seen yet.
        info = policy.expand(node.state, node.depth,
                              parent_u_s=node.U_s, K=K)
        existing_keys = set(node.children.keys()) | {
            c.class_key for c in node.untried
        }
        new_classes = [c for c in info.classes
                       if c.class_key not in existing_keys]
        node.untried = sorted(
            new_classes, key=lambda c: -(c.mass * c.u_s),
        )
        if not node.untried:
            return None

    chosen = node.untried.pop(0)
    # Carry the newly-resolved subproblem into bindings under both the G'
    # node id (subproblem_text) and any source-language alias the policy
    # expansion put in `bindings` (e.g. raw `node_K` keys for math
    # prompts). Without this the TT key never changes between rollouts
    # and the cache wrongly dedupes distinct subtree paths.
    extra_bindings = {node.subproblem_text: chosen.answer_text}
    new_state = node.state.extend(node.subproblem_text, chosen.answer_text,
                                  **extra_bindings)
    child = Node(
        state=new_state,
        depth=node.depth + 1,
        parent=node,
        prior=chosen.mass * chosen.u_s,
        U_s=chosen.u_s,
        sigma=chosen.sigma,
        terminal=chosen.terminal_hint,
        class_key=chosen.class_key,
        answer_text=chosen.answer_text,
    )
    node.children[chosen.class_key] = child
    return child


# ---------------------------------------------------------------------------
# Evaluation: U_s + (M4) incremental coherence gate
# ---------------------------------------------------------------------------

# A gate function maps (state, depth) -> sigma in [0, 1]. 0 == coherent.
# The orchestrator passes one in; M3 unit tests omit it (defaults to no-gate).
GateFn = Callable[[SolutionState, int], float]


def evaluate(node: Node, terminal_check: TerminalChecker | None = None,
             gate_fn: GateFn | None = None) -> float:
    """Return the value used to backup from this node.

    Incremental hard gate (decision 5 + spec lines 26-28): if any newly-
    checkable constraint is violated at this depth, set sigma > 0, value 0,
    and mark the node terminal-fail. ALWAYS record sigma even when gating
    hard (decision 6) — soft-relax ranks on it later.
    """
    if terminal_check is not None and terminal_check.is_terminal(node.state, node.depth):
        node.terminal = True
    # Gate runs unconditionally (the spec: "evaluated at the depth each
    # constraint first becomes checkable"). Recording the sigma is the
    # invariant that soft-relax depends on.
    if gate_fn is not None:
        sigma = float(gate_fn(node.state, node.depth))
        node.sigma = sigma
        if sigma > 0.0:
            node.gated_fail = True
            node.terminal = True
            return 0.0
    if node.gated_fail:
        return 0.0
    return node.U_s


# ---------------------------------------------------------------------------
# Backup: guarded-max to root
# ---------------------------------------------------------------------------

def backup(leaf: Node, value: float, cfg: Config) -> None:
    """Walk leaf -> root. Each ancestor's Q is the max over CHILDREN with
    N >= n_min. If no child qualifies, hold the ancestor's own U_s.

    This is the spec's "guarded max" — single-rollout overestimation guard.
    """
    n_min = cfg.mcts.n_min
    use_avg = cfg.ablation.backup == "average"

    # First, the leaf itself
    leaf.N += 1
    # Leaf's Q for its own row = its evaluated value (no children to max over).
    leaf.Q = value

    cur = leaf.parent
    while cur is not None:
        cur.N += 1
        if use_avg:
            # ablation arm: incremental running mean of all child Qs weighted
            # by visit count, plus the rollout's value via the current child.
            total_q = 0.0
            total_n = 0
            for c in cur.children.values():
                total_q += c.Q * c.N
                total_n += c.N
            cur.Q = total_q / max(1, total_n)
        else:
            qualified = [c.Q for c in cur.children.values() if c.N >= n_min]
            if qualified:
                cur.Q = max(qualified)
            else:
                # nothing visited enough yet — fall back to this node's
                # intrinsic U_s (which expand populated from its prior).
                cur.Q = cur.U_s
        cur = cur.parent


# ---------------------------------------------------------------------------
# search() — main rollout loop (without M4 gate / depth governor)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    root: Node
    rollouts: int
    best_terminal: Node | None
    best_value: float


@dataclass
class RolloutRecord:
    """Full provenance for one rollout — what was selected, what was created,
    what value was backed up. Consumed by the Tracer to write rollouts.jsonl.
    """

    i: int
    selected_path: list[str]            # node_ids from root to the select() result
    expanded_parent_id: str | None      # the node we tried to expand
    new_child_id: str | None            # the newly created child, if any
    new_child_state_extend: tuple[str, str] | None    # (subproblem, answer) added
    leaf_id: str                        # the node actually evaluated
    leaf_depth: int
    leaf_terminal: bool
    leaf_gated_fail: bool
    leaf_sigma: float
    leaf_U_s: float
    value: float                        # value backed up
    backup_path: list[tuple[str, int, float]]   # (id, new_N, new_Q) along ancestors
    expansion_info: dict | None = None   # set by orchestrator/policy if available
    stop_now: bool = False


def search(
    root_state: SolutionState,
    *,
    cfg: Config,
    policy: ExpansionPolicy,
    terminal_check: TerminalChecker,
    gate_fn: GateFn | None = None,
    stop_fn: Callable[[int, Node], bool] | None = None,
    K: int | None = None,
    rng: random.Random | None = None,
    on_rollout: Callable[[int, Node], None] | None = None,
    reached_terminals: list[Node] | None = None,
) -> SearchResult:
    """Run the MCTS rollout loop until budget N is exhausted.

    `on_rollout(i, leaf)` is called after each backup; the orchestrator uses
    it for telemetry, budget checks, and (M4) the marginal-gain stop.
    """
    K = K or cfg.sampling.K_blackbox
    rng = rng or random.Random(cfg.seed)

    root = Node(state=root_state, depth=0)
    tt = TranspositionTable()

    best_terminal: Node | None = None
    best_value: float = -math.inf

    rollouts_done = 0
    import inspect
    rollout_arity = (len(inspect.signature(on_rollout).parameters)
                     if on_rollout is not None else 0)

    for i in range(cfg.mcts.N):
        # ---- select ----
        s = select(root, cfg)
        # build the path from root to s for tracing (cheap; depth is small)
        path: list[str] = []
        cur: Node | None = s
        while cur is not None:
            path.append(cur.node_id)
            cur = cur.parent
        path.reverse()

        # ---- expand ----
        expanded_parent = s
        new_child_id: str | None = None
        new_child_extend = None
        if not s.terminal:
            child = expand(s, policy, K=K, cfg=cfg, tt=tt)
            if child is not None:
                new_child_id = child.node_id
                if child.state.resolved:
                    new_child_extend = child.state.resolved[-1]
                s = child

        # ---- evaluate (incl. gate) ----
        val = evaluate(s, terminal_check, gate_fn=gate_fn)

        # ---- backup ----
        backup(s, val, cfg)
        # Record post-backup state along ancestors
        bk_path: list[tuple[str, int, float]] = []
        cur = s
        while cur is not None:
            bk_path.append((cur.node_id, cur.N, round(cur.Q, 6)))
            cur = cur.parent
        rollouts_done = i + 1

        if s.terminal:
            if reached_terminals is not None:
                reached_terminals.append(s)
            if not s.gated_fail and val > best_value:
                best_value = val
                best_terminal = s

        record = RolloutRecord(
            i=i,
            selected_path=path,
            expanded_parent_id=expanded_parent.node_id,
            new_child_id=new_child_id,
            new_child_state_extend=new_child_extend,
            leaf_id=s.node_id,
            leaf_depth=s.depth,
            leaf_terminal=s.terminal,
            leaf_gated_fail=s.gated_fail,
            leaf_sigma=s.sigma,
            leaf_U_s=s.U_s,
            value=val,
            backup_path=bk_path,
            stop_now=False,
        )

        if on_rollout is not None:
            # Backwards compatible: legacy callers expect (i, leaf).
            if rollout_arity >= 3:
                on_rollout(i, s, record)
            else:
                on_rollout(i, s)
        if stop_fn is not None and stop_fn(i, s):
            record.stop_now = True
            break

    return SearchResult(
        root=root, rollouts=rollouts_done,
        best_terminal=best_terminal, best_value=best_value,
    )
