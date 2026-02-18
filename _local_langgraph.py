"""
Lightweight fallback implementation for StateGraph used by the
workflow modules (e.g., workflow/graph_chat.py).
This implements the minimal API the workflow expects: node registration,
edges, a simple conditional edge, compile (no-op) and invoke to run the graph.
"""

END = "END"

class StateGraph:
    def __init__(self, state_type=None):
        self.nodes = {}
        self.edges = {}
        self.entry = None
        self.conditional = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, src, dst):
        self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, node, func, path_map=None):
        # func: state -> next_key_or_node_or_END
        # path_map: optional mapping from key -> node name
        self.conditional = (node, func, path_map)

    def compile(self):
        return self

    def invoke(self, state):
        current = self.entry
        if current is None:
            return
        st = dict(state or {})
        while True:
            fn = self.nodes.get(current)
            if fn is None:
                break
            out = fn(st) or {}
            if isinstance(out, dict):
                st.update(out)

            if self.conditional and current == self.conditional[0]:
                _, cond_fn, path_map = self.conditional
                result = cond_fn(st)
                if path_map and isinstance(path_map, dict):
                    nxt = path_map.get(result, result)
                else:
                    nxt = result
            else:
                nxts = self.edges.get(current, [])
                nxt = nxts[0] if nxts else None

            if nxt in (None, END):
                break
            current = nxt

        return st
