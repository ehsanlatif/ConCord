"""Dependency-graph extraction + condensation (§5.3).

`extract_graph` walks the problem text to build an explicit DAG of subproblems
(when the problem text exposes such structure — the eval set's math templates
do). `condense` applies Tarjan's SCC to handle cycles (lifting any SCC to a
single super-node), and `critical_path_len` is the longest path on the
condensation — the spec's `d_init` = the structural depth estimate.

For problems with no exploitable explicit structure, callers should use the
implicit decomposer in `decompose.py` to discover the graph online; the
graph here can stay a trivial 1-node DAG until then.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

NODE_HEADER_RE = re.compile(
    r"^Problem\s+node_(\d+)\s*:\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)
# Broad cross-block dependency match. The the eval set dataset uses MANY
# variant phrasings — `answer from`, `fraction from`, `solution set from`,
# `total area from`, `binomial term from`, etc. — all ending in
# `from problem node_K`. We capture every such occurrence as a dependency
# so the topological ordering doesn't miss the link. (Earlier the regex
# required the literal word `answer`, which silently dropped many real
# dependencies and made downstream blocks look independent.)
NODE_REF_RE = re.compile(
    r"from\s+problem\s+node_(\d+)",
    flags=re.IGNORECASE,
)


@dataclass
class SubproblemNode:
    node_id: str          # e.g. "node_0"
    text: str             # the subproblem text (header stripped)
    refs: list[str]       # node_ids this subproblem depends on


def parse_explicit_nodes(problem_text: str) -> list[SubproblemNode]:
    """Split a problem text on `Problem node_N:` headers.

    Returns one SubproblemNode per header in source order, with refs populated
    from any `answer from problem node_M` patterns inside the body.
    Returns [] when the text has no headers (caller should fall back to
    implicit decomposition).
    """
    headers = list(NODE_HEADER_RE.finditer(problem_text))
    if not headers:
        return []

    nodes: list[SubproblemNode] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(problem_text)
        body = problem_text[start:end].strip()
        node_id = f"node_{m.group(1)}"
        refs = [f"node_{x}" for x in NODE_REF_RE.findall(body)]
        # Deduplicate while preserving order
        refs = list(dict.fromkeys(refs))
        nodes.append(SubproblemNode(node_id=node_id, text=body, refs=refs))
    return nodes


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------

def extract_graph(problem_text: str) -> nx.DiGraph:
    """Build G (may contain cycles in general; eval-set math is acyclic).

    Edge convention: dep -> dependent.  So if node_1 references node_0,
    we add `node_0 -> node_1`. That makes "ready to solve" = predecessors
    all resolved, and topological order = solve order.
    """
    G: nx.DiGraph = nx.DiGraph()
    parsed = parse_explicit_nodes(problem_text)

    if not parsed:
        # fallback: single-node graph wrapping the whole problem
        G.add_node("root", text=problem_text, source="implicit")
        return G

    for n in parsed:
        G.add_node(n.node_id, text=n.text, source="explicit")
    for n in parsed:
        for r in n.refs:
            if r not in G.nodes:
                # forward reference to an undeclared node — record it so the
                # SCC pass treats it as a stub.
                G.add_node(r, text="", source="forward_ref")
            G.add_edge(r, n.node_id)
    return G


def condense(G: nx.DiGraph) -> nx.DiGraph:
    """Tarjan SCC condensation → DAG (G'). Each G' node carries `members`."""
    cond = nx.condensation(G)
    # Stable string-keyed nodes for downstream use. `mapping` is index->scc_id.
    # nx.condensation already gives integer node ids; rewrite to scc_N strings.
    out: nx.DiGraph = nx.DiGraph()
    for n in cond.nodes:
        members = sorted(cond.nodes[n]["members"])
        out.add_node(f"scc_{n}", members=members,
                     text=" || ".join(G.nodes[m].get("text", "") for m in members))
    for u, v in cond.edges:
        out.add_edge(f"scc_{u}", f"scc_{v}")
    return out


def critical_path_len(Gp: nx.DiGraph) -> int:
    """Longest path in G' (in edges). 0 for a single-node graph.

    Spec calls this `d_init` — the structural depth estimate. For
    math templates that chain 4 nodes linearly, this is 3.
    """
    if Gp.number_of_nodes() == 0:
        return 0
    # `dag_longest_path` returns the node list; length in edges = len - 1.
    try:
        path = nx.dag_longest_path(Gp)
    except nx.NetworkXUnfeasible as e:
        raise ValueError("Graph passed to critical_path_len is not a DAG") from e
    return max(0, len(path) - 1)


def topo_order(Gp: nx.DiGraph) -> list[str]:
    """Stable topological order of G' (alphabetic tiebreak)."""
    # `lexicographical_topological_sort` is deterministic — useful for
    # reproducible runs.
    return list(nx.lexicographical_topological_sort(Gp))
