"""
iot_botnet_wl.py
----------------
Python port of the browser simulation: detects a botnet C2 cluster inside
an IoT device network using 1-Weisfeiler-Lehman colour refinement, the
same mathematical operation a Graph Isomorphism Network (GIN) layer
performs. Also demonstrates evasion (routing the attack through a relay
device) and a relay-tolerant 2-hop detection method.

Requires: networkx, matplotlib
    pip install networkx matplotlib --break-system-packages
"""

import random
import hashlib
from itertools import count

import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------

def build_network(n=16, edge_prob=0.05, embed=True, evade=False, seed=None):
    """
    Build a random IoT device network, optionally embedding a botnet
    cluster as either:
      - a direct star (hub -> 4 leaves), or
      - an evasive tree (hub -> 2 leaves + relay -> 2 leaves)

    Background noise edges avoid the embedded cluster's own nodes, so
    the demo cleanly shows: exact match when un-evaded, and a genuine
    (not noise-caused) mismatch when evaded.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    special = set()
    if embed:
        special = {0, 1, 2, 3, 4, 5} if evade else {0, 1, 2, 3, 4}

    for i in range(n):
        for j in range(i + 1, n):
            if i in special or j in special:
                continue
            if rng.random() < edge_prob:
                G.add_edge(i, j)

    if embed:
        hub, leaves = 0, [1, 2, 3, 4]
        if not evade:
            for l in leaves:
                G.add_edge(hub, l)
        else:
            relay = 5
            G.add_edge(hub, leaves[0])
            G.add_edge(hub, leaves[1])
            G.add_edge(hub, relay)
            G.add_edge(relay, leaves[2])
            G.add_edge(relay, leaves[3])
    return G


def signature_graph():
    """The known botnet signature: a hub controlling 4 leaves directly."""
    S = nx.Graph()
    S.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return S


# ----------------------------------------------------------------------
# 1-Weisfeiler-Lehman colour refinement (what a GIN layer computes)
# ----------------------------------------------------------------------

def wl_refinement(G, rounds=2):
    """
    Return a list `colors[r][v]` = the WL colour (a string certificate)
    of node v after r rounds of refinement.
      round 0: colour = degree
      round r: colour = (own colour, sorted multiset of neighbours' colours)
    """
    colors = [{v: f"d{G.degree(v)}" for v in G.nodes()}]
    cur = colors[0]
    for _ in range(rounds):
        nxt = {}
        for v in G.nodes():
            nbr_cols = sorted(cur[u] for u in G.neighbors(v))
            nxt[v] = cur[v] + "|" + ",".join(nbr_cols)
        colors.append(nxt)
        cur = nxt
    return colors


def color_hue(label: str) -> float:
    """Deterministically map a colour-string to a hue in [0, 1) for plotting."""
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return (h % 360) / 360.0


# ----------------------------------------------------------------------
# Detection methods
# ----------------------------------------------------------------------

def detect_wl_certificate(G, colors, sig_colors, final_round=2):
    """
    Exact isomorphism-certificate match: flag the first node whose
    round-`final_round` WL colour exactly equals the signature hub's
    colour at the same round.
    """
    sig_hub_cert = sig_colors[final_round][0]
    print(f"Signature hub certificate (round {final_round}): {sig_hub_cert[:60]}...")
    for v in G.nodes():
        cert = colors[final_round][v]
        print(f"  node {v}: {cert[:50]}{'...' if len(cert) > 50 else ''}")
        if cert == sig_hub_cert:
            leaves = list(G.neighbors(v))
            print(f"EXACT MATCH: node {v} + leaves {leaves}")
            return v, leaves
    print("No certificate match. (Possibly evaded.)")
    return None, []


def detect_depth2(G, min_reach=4):
    """
    Relay-tolerant detection: flag the first node whose 2-hop
    neighbourhood contains at least `min_reach` other devices.
    """
    for v in sorted(G.nodes(), key=lambda x: -G.degree(x)):
        if G.degree(v) < 2:
            continue
        reach = nx.single_source_shortest_path_length(G, v, cutoff=2)
        others = [k for k in reach if k != v]
        print(f"  node {v}: reaches {len(others)} devices within 2 hops")
        if len(others) >= min_reach:
            print(f"MATCH: node {v} 2-hop neighbourhood {others}")
            return v, others
    print("No 2-hop match found.")
    return None, []


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def visualize(G, colors, result=None, title="IoT Network", save_path="network.png"):
    pos = nx.circular_layout(G)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for r, ax in enumerate(axes):
        node_colors = [color_hue(colors[r][v]) for v in G.nodes()]
        if result and r == 2:
            hub, cluster = result
            node_colors = []
            for v in G.nodes():
                if v == hub:
                    node_colors.append("crimson")
                elif v in cluster:
                    node_colors.append("orange")
                else:
                    node_colors.append(plt.cm.hsv(color_hue(colors[r][v])))
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                    node_size=500, font_size=8, font_color="white")
        else:
            cmap_colors = [plt.cm.hsv(h) for h in node_colors]
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=cmap_colors,
                     node_size=500, font_size=8)
        ax.set_title(f"Round {r}")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

def run_demo(evade=False, method="wl", seed=1):
    print("=" * 70)
    print(f"Building network (evade={evade})")
    print("=" * 70)
    G = build_network(embed=True, evade=evade, seed=seed)
    S = signature_graph()

    colors = wl_refinement(G, rounds=2)
    sig_colors = wl_refinement(S, rounds=2)

    print("\nRunning detection method:", method)
    if method == "wl":
        result = detect_wl_certificate(G, colors, sig_colors)
    else:
        result = detect_depth2(G)

    visualize(G, colors, result=result if result[0] is not None else None,
              title=f"WL refinement (evade={evade}, method={method})",
              save_path=f"network_{method}_evade{int(evade)}.png")
    return G, result


if __name__ == "__main__":
    print("\n### Case 1: direct star, exact WL certificate match ###")
    run_demo(evade=False, method="wl", seed=2)

    print("\n### Case 2: relay evasion, exact WL certificate match (should FAIL) ###")
    run_demo(evade=True, method="wl", seed=2)

    print("\n### Case 3: relay evasion, 2-hop relay-tolerant match (should SUCCEED) ###")
    run_demo(evade=True, method="depth2", seed=2)
