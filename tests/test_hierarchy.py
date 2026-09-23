import pytest

from esco_occupation_indexer.errors import ValidationError
from esco_occupation_indexer.hierarchy import GraphNode, HierarchyGraph
from tests.helpers import CHILD_OCC_URI, PARENT_OCC_URI, ROOT_URI, UNIT_URI


def _nodes() -> dict[str, GraphNode]:
    return {
        ROOT_URI: GraphNode(ROOT_URI, "Professionals", "isco_group", "2"),
        UNIT_URI: GraphNode(
            UNIT_URI,
            "Software and applications developers and analysts",
            "isco_group",
            "251",
        ),
        PARENT_OCC_URI: GraphNode(PARENT_OCC_URI, "software developer", "occupation"),
        CHILD_OCC_URI: GraphNode(CHILD_OCC_URI, "backend developer", "occupation"),
    }


def test_hierarchy_supports_occupation_parent_with_full_isco_path() -> None:
    graph = HierarchyGraph(
        _nodes(),
        {
            UNIT_URI: {ROOT_URI},
            PARENT_OCC_URI: {UNIT_URI},
            CHILD_OCC_URI: {PARENT_OCC_URI},
        },
    )
    hierarchy = graph.hierarchy_for(CHILD_OCC_URI)
    assert hierarchy.parent_type == "occupation"
    assert hierarchy.parent_uri == PARENT_OCC_URI
    assert hierarchy.terminal_isco_uri == UNIT_URI
    assert hierarchy.isco_major == "2"
    assert hierarchy.ancestor_uris == [ROOT_URI, UNIT_URI, PARENT_OCC_URI]
    assert hierarchy.level == 4


def test_hierarchy_chooses_longest_path_for_multiple_parents() -> None:
    graph = HierarchyGraph(
        _nodes(),
        {
            UNIT_URI: {ROOT_URI},
            PARENT_OCC_URI: {UNIT_URI},
            CHILD_OCC_URI: {PARENT_OCC_URI, UNIT_URI},
        },
    )

    hierarchy = graph.hierarchy_for(CHILD_OCC_URI)

    assert hierarchy.parent_uri == PARENT_OCC_URI
    assert hierarchy.ancestor_uris == [ROOT_URI, UNIT_URI, PARENT_OCC_URI]


def test_hierarchy_rejects_cycles() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        HierarchyGraph(_nodes(), {UNIT_URI: {ROOT_URI}, ROOT_URI: {UNIT_URI}})


def test_hierarchy_rejects_orphans() -> None:
    missing = "http://data.europa.eu/esco/isco/C9"
    with pytest.raises(ValidationError, match="missing"):
        HierarchyGraph(_nodes(), {CHILD_OCC_URI: {missing}})


def test_hierarchy_rejects_root_with_parent() -> None:
    nodes = {
        **_nodes(),
        "http://data.europa.eu/esco/isco/C1": GraphNode(
            "http://data.europa.eu/esco/isco/C1", "Managers", "isco_group", "1"
        ),
    }
    with pytest.raises(ValidationError, match="must not have a parent"):
        HierarchyGraph(
            nodes,
            {
                UNIT_URI: {ROOT_URI},
                ROOT_URI: {"http://data.europa.eu/esco/isco/C1"},
                PARENT_OCC_URI: {UNIT_URI},
                CHILD_OCC_URI: {PARENT_OCC_URI},
            },
        )
