"""
Unit tests for DraftTree structure and tree-based verification using unittest.
"""

import unittest
from spec_engine.sampler import RejectionSampler
from spec_engine.tree_draft import DraftTree


class TestTreeDraft(unittest.TestCase):

    def test_tree_construction_and_paths(self):
        # Root = 100
        tree = DraftTree(root_token=100)
        # Node 1: Child of root (token 101)
        # Node 2: Child of root (token 102)
        n1 = tree.add_node(parent_id=0, token_id=101)
        n2 = tree.add_node(parent_id=0, token_id=102)

        # Node 3: Child of node 1 (token 103)
        n3 = tree.add_node(parent_id=n1.node_id, token_id=103)

        paths = tree.get_all_paths()
        # Two distinct paths: [root, n1, n3] and [root, n2]
        self.assertEqual(len(paths), 2)

        path_lengths = sorted([len(p) for p in paths])
        self.assertEqual(path_lengths, [2, 3])

    def test_tree_verification_accepts_best_path(self):
        sampler = RejectionSampler(seed=42)
        tree = DraftTree(root_token=100)

        # Branch A: node 1 (token 201) -> node 2 (token 202)
        # Branch B: node 3 (token 301)
        n1 = tree.add_node(parent_id=0, token_id=201)
        n2 = tree.add_node(parent_id=n1.node_id, token_id=202)
        n3 = tree.add_node(parent_id=0, token_id=301)

        # Configure Branch A to have p >= q (100% accepted)
        # Configure Branch B to have p = 0 (100% rejected)
        target_dists = {
            n1.node_id: {201: 1.0},
            n2.node_id: {202: 1.0},
            n3.node_id: {301: 0.0, 999: 1.0},
        }
        draft_dists = {
            n1.node_id: {201: 1.0},
            n2.node_id: {202: 1.0},
            n3.node_id: {301: 1.0},
        }

        result = tree.verify(sampler, target_dists, draft_dists)

        # Branch A has length 2 accepted tokens [201, 202]
        self.assertEqual(result.accepted_tokens, [201, 202])
        self.assertEqual(result.best_path_node_ids, [n1.node_id, n2.node_id])


if __name__ == "__main__":
    unittest.main()
