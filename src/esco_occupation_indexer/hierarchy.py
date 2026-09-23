from __future__ import annotations

from dataclasses import dataclass

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.models import OccupationHierarchyData, OccupationHierarchyNode


@dataclass(frozen=True)
class GraphNode:
    uri: str
    label: str
    node_type: str
    code: str | None = None


class HierarchyGraph:
    """ISCO + ESCO occupation pillar graph.

    Nodes are ISCO groups (``…/esco/isco/<code>``) and ESCO occupations.
    The ten single-digit ISCO major groups (``0``–``9``) are roots; every other
    node must have at least one broader parent.  ESCO is poly-hierarchical, so
    one canonical path per occupation is exposed: the longest root path, with
    URI tuples breaking ties deterministically.
    """

    def __init__(self, nodes: dict[str, GraphNode], parents: dict[str, set[str]]) -> None:
        self.nodes = dict(nodes)
        self.parents = {
            child_uri: set(parent_uris)
            for child_uri, parent_uris in parents.items()
            if parent_uris
        }
        self._canonical_paths: dict[str, tuple[GraphNode, ...]] = {}
        self._validate_relationships()

    def _is_root(self, node: GraphNode) -> bool:
        return (
            node.node_type == "isco_group"
            and node.code is not None
            and len(node.code.strip()) == 1
        )

    def _validate_relationships(self) -> None:
        for uri, node in self.nodes.items():
            if not uri or node.uri != uri:
                raise ValidationError(f"Hierarchy node URI is inconsistent: {uri!r}")
            if node.node_type not in {"isco_group", "occupation"}:
                raise ValidationError(f"Invalid hierarchy node type for {uri}: {node.node_type}")
            if not node.label.strip():
                raise ValidationError(f"Hierarchy node has an empty label: {uri}")
            if node.node_type == "isco_group" and not (node.code or "").strip():
                raise ValidationError(f"ISCO group has an empty code: {uri}")
            if node.node_type == "occupation" and node.code is not None:
                raise ValidationError(f"Occupation must not carry a group code: {uri}")

        for child_uri, parent_uris in self.parents.items():
            if child_uri not in self.nodes:
                raise ValidationError(f"Hierarchy child is missing from source nodes: {child_uri}")
            for parent_uri in parent_uris:
                if parent_uri not in self.nodes:
                    raise ValidationError(
                        f"Hierarchy parent {parent_uri} for {child_uri} "
                        "is missing from source nodes"
                    )

        state: dict[str, int] = {}

        def visit(uri: str, trail: list[str]) -> None:
            if state.get(uri) == 1:
                cycle = " -> ".join([*trail, uri])
                raise ValidationError(f"Hierarchy cycle detected: {cycle}")
            if state.get(uri) == 2:
                return
            state[uri] = 1
            for parent_uri in self.parents.get(uri, set()):
                visit(parent_uri, [*trail, uri])
            state[uri] = 2

        for uri in self.nodes:
            visit(uri, [])

        for uri, node in self.nodes.items():
            parent_uris = self.parents.get(uri, set())
            if self._is_root(node):
                if parent_uris:
                    raise ValidationError(f"Hierarchy root must not have a parent: {uri}")
            elif not parent_uris:
                raise ValidationError(f"Hierarchy node is orphaned: {uri}")

    def _canonical_path(self, uri: str) -> tuple[GraphNode, ...]:
        cached = self._canonical_paths.get(uri)
        if cached is not None:
            return cached
        parent_uris = self.parents.get(uri, set())
        if not parent_uris:
            path = (self.nodes[uri],)
        else:
            parent_paths = [self._canonical_path(parent_uri) for parent_uri in parent_uris]
            parent_path = min(
                parent_paths,
                key=lambda candidate: (
                    -len(candidate),
                    tuple(node.uri for node in candidate),
                ),
            )
            path = (*parent_path, self.nodes[uri])
        self._canonical_paths[uri] = path
        return path

    def hierarchy_for(self, occupation_uri: str) -> OccupationHierarchyData:
        if occupation_uri not in self.nodes:
            raise ValidationError(f"Unknown hierarchy node: {occupation_uri}")
        if self.nodes[occupation_uri].node_type != "occupation":
            raise ValidationError(
                f"Hierarchy metadata is only defined for occupations: {occupation_uri}"
            )
        if occupation_uri not in self.parents:
            raise ValidationError(f"Released occupation has no broader relation: {occupation_uri}")

        path_nodes = list(self._canonical_path(occupation_uri)[:-1])
        direct_parent = path_nodes[-1].uri
        isco_nodes = [node for node in reversed(path_nodes) if node.node_type == "isco_group"]
        if not isco_nodes:
            raise ValidationError(
                f"Released occupation has no ISCO group ancestor: {occupation_uri}"
            )
        terminal_isco = isco_nodes[0]

        root = path_nodes[0]
        if root.node_type != "isco_group" or not self._is_root(root):
            raise ValidationError(f"Could not derive ISCO major root for {occupation_uri}")
        isco_major = (root.code or "").strip()

        parent_node = self.nodes[direct_parent]
        return OccupationHierarchyData(
            level=len(path_nodes) + 1,
            isco_major=isco_major,
            parent_uri=direct_parent,
            parent_type=parent_node.node_type,  # type: ignore[arg-type]
            terminal_isco_uri=terminal_isco.uri,
            ancestor_uris=[node.uri for node in path_nodes],
            path=[
                OccupationHierarchyNode(
                    uri=node.uri,
                    code=node.code,
                    label=node.label,
                    node_type=node.node_type,  # type: ignore[arg-type]
                )
                for node in path_nodes
            ],
        )
