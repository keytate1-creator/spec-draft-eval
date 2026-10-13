"""
Tree-structured speculative decoding verification engine (Medusa / Eagle style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from spec_engine.sampler import RejectionSampler


class DraftTreeNode:
    """A speculative candidate token node in a hypothesis tree."""

    def __init__(
        self,
        node_id: int,
        token_id: int,
        parent: Optional[DraftTreeNode] = None,
    ):
        self.node_id = node_id
        self.token_id = token_id
        self.parent = parent
        self.children: List[DraftTreeNode] = []

    def add_child(self, child: DraftTreeNode) -> None:
        child.parent = self
        self.children.append(child)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_path_from_root(self) -> List[DraftTreeNode]:
        """Returns sequence of nodes from root down to this node."""
        path: List[DraftTreeNode] = []
        curr: Optional[DraftTreeNode] = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path


@dataclass
class TreeVerificationResult:
    """Consolidated outcome of tree-structured speculative verification."""
    accepted_tokens: List[int]
    recovery_token: Optional[int]
    best_path_node_ids: List[int]

    @property
    def total_emitted_tokens(self) -> int:
        return len(self.accepted_tokens) + (1 if self.recovery_token is not None else 0)


class DraftTree:
    """Manages multi-candidate tree construction and parallel verification."""

    def __init__(self, root_token: int):
        self.root = DraftTreeNode(node_id=0, token_id=root_token, parent=None)
        self._next_node_id = 1
        self.nodes_by_id: Dict[int, DraftTreeNode] = {0: self.root}

    def add_node(self, parent_id: int, token_id: int) -> DraftTreeNode:
        """Adds a candidate branch to the tree."""
        if parent_id not in self.nodes_by_id:
            raise KeyError(f"Parent node {parent_id} does not exist in tree.")

        parent = self.nodes_by_id[parent_id]
        new_node = DraftTreeNode(
            node_id=self._next_node_id,
            token_id=token_id,
            parent=parent,
        )
        self.nodes_by_id[self._next_node_id] = new_node
        self._next_node_id += 1
        parent.add_child(new_node)
        return new_node

    def get_all_paths(self) -> List[List[DraftTreeNode]]:
        """Returns all distinct root-to-leaf paths in the tree."""
        paths: List[List[DraftTreeNode]] = []

        def dfs(node: DraftTreeNode, current_path: List[DraftTreeNode]) -> None:
            path_with_node = current_path + [node]
            if node.is_leaf:
                paths.append(path_with_node)
            else:
                for child in node.children:
                    dfs(child, path_with_node)

        dfs(self.root, [])
        return paths

    def verify(
        self,
        sampler: RejectionSampler,
        target_distributions: Dict[int, Dict[int, float]],
        draft_distributions: Dict[int, Dict[int, float]],
    ) -> TreeVerificationResult:
        """
        Verifies the tree hypothesis against target model distributions.
        Finds the longest path of consecutively accepted candidate tokens.
        target_distributions maps node_id -> p(x) distribution.
        draft_distributions maps node_id -> q(x) distribution.
        """
        all_paths = self.get_all_paths()

        best_accepted_tokens: List[int] = []
        best_recovery_token: Optional[int] = None
        best_node_ids: List[int] = []

        for path in all_paths:
            # Skip root token (root is already emitted)
            candidate_nodes = path[1:]

            curr_accepted: List[int] = []
            curr_recovery: Optional[int] = None
            curr_ids: List[int] = []

            for node in candidate_nodes:
                p_dist = target_distributions.get(node.node_id, {})
                q_dist = draft_distributions.get(node.node_id, {})

                decision = sampler.verify_token(node.token_id, p_dist, q_dist)
                if decision.accepted:
                    curr_accepted.append(node.token_id)
                    curr_ids.append(node.node_id)
                else:
                    curr_recovery = decision.output_token
                    break

            if len(curr_accepted) > len(best_accepted_tokens):
                best_accepted_tokens = curr_accepted
                best_recovery_token = curr_recovery
                best_node_ids = curr_ids

        # If no candidates accepted along any branch, fallback to recovery from first candidate
        if not best_accepted_tokens and not best_recovery_token and len(self.root.children) > 0:
            first_child = self.root.children[0]
            p_dist = target_distributions.get(first_child.node_id, {})
            q_dist = draft_distributions.get(first_child.node_id, {})
            decision = sampler.verify_token(first_child.token_id, p_dist, q_dist)
            best_recovery_token = decision.output_token

        return TreeVerificationResult(
            accepted_tokens=best_accepted_tokens,
            recovery_token=best_recovery_token,
            best_path_node_ids=best_node_ids,
        )
