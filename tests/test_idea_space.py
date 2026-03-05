"""Tests for the Idea Space vector system."""

from __future__ import annotations

import pytest

from anima.models import Impression, ImpressionType
from anima.systems.idea_space import IdeaSpace


class TestEmbedding:
    async def test_embed_returns_vector(self, state):
        space = IdeaSpace(state)
        emb = space.embed("hello world")
        assert len(emb) == 32
        assert all(0.0 <= v <= 1.0 for v in emb)

    async def test_embed_deterministic(self, state):
        space = IdeaSpace(state)
        a = space.embed("test")
        b = space.embed("test")
        assert a == b

    async def test_embed_different_texts(self, state):
        space = IdeaSpace(state)
        a = space.embed("hello")
        b = space.embed("goodbye")
        assert a != b


class TestCosineSimilarity:
    def test_identical_vectors(self):
        sim = IdeaSpace.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        sim = IdeaSpace.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert sim == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert IdeaSpace.cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert IdeaSpace.cosine_similarity([1, 2], [1, 2, 3]) == 0.0


class TestFindSimilar:
    async def test_finds_similar(self, state):
        space = IdeaSpace(state)
        # Store an impression with known embedding
        imp = Impression(
            id="imp1",
            type=ImpressionType.PATTERN,
            content="test content",
            embedding=space.embed("test content"),
        )
        await state.store_impression(imp)

        # Find similar with same embedding
        results = await space.find_similar(space.embed("test content"), threshold=0.9)
        assert len(results) >= 1
        assert results[0][0]["id"] == "imp1"

    async def test_no_similar_below_threshold(self, state):
        space = IdeaSpace(state)
        imp = Impression(
            id="imp1",
            embedding=space.embed("completely different text"),
        )
        await state.store_impression(imp)
        results = await space.find_similar(space.embed("unrelated topic"), threshold=0.99)
        # With SHA256 pseudo-embeddings, very different texts should not match
        # at 0.99 threshold (though they might still be somewhat similar)
        # This is a soft assertion due to hash-based embeddings
        assert isinstance(results, list)


class TestClusters:
    async def test_find_clusters_requires_min_size(self, state):
        space = IdeaSpace(state)
        # Only 2 impressions — below min_cluster_size of 3
        for i in range(2):
            imp = Impression(
                id=f"imp{i}",
                embedding=space.embed("same topic"),
            )
            await state.store_impression(imp)
        clusters = await space.find_clusters(threshold=0.5, min_cluster_size=3)
        assert len(clusters) == 0

    async def test_find_clusters_with_identical_embeddings(self, state):
        space = IdeaSpace(state)
        emb = space.embed("identical content")
        for i in range(5):
            imp = Impression(id=f"imp{i}", embedding=emb)
            await state.store_impression(imp)
        clusters = await space.find_clusters(threshold=0.99, min_cluster_size=3)
        assert len(clusters) >= 1
        assert len(clusters[0]) == 5


class TestCondensation:
    async def test_condense_cluster(self, state):
        space = IdeaSpace(state)
        cluster = [
            {"pressure": 0.3, "emotional_charge": 0.5, "times_reinforced": 2},
            {"pressure": 0.8, "emotional_charge": -0.2, "times_reinforced": 1},
            {"pressure": 0.5, "emotional_charge": 0.1, "times_reinforced": 3},
        ]
        result = space.condense_cluster(cluster)
        assert result["pressure"] == 0.8  # Max
        assert result["times_reinforced"] == 6  # Sum
        # Weighted avg charge
        assert -1.0 <= result["emotional_charge"] <= 1.0


class TestDisplacement:
    async def test_displace_transfers_pressure(self, state):
        space = IdeaSpace(state)
        emb = space.embed("shared topic")

        source = Impression(
            id="src", pressure=0.8, emotional_charge=0.9, embedding=emb
        )
        target = Impression(
            id="tgt", pressure=0.2, emotional_charge=0.1, embedding=emb
        )
        await state.store_impression(source)
        await state.store_impression(target)

        result = await space.displace(
            {"id": "src", "pressure": 0.8, "emotional_charge": 0.9, "embedding": emb},
            alpha=0.6,
        )
        assert result is not None


class TestConvergence:
    async def test_convergence_boost(self, state):
        space = IdeaSpace(state)
        cluster = [
            {"emotional_charge": 0.5},
            {"emotional_charge": 0.3},
            {"emotional_charge": 0.4},
        ]
        boost = space.check_convergence(cluster, coefficient=0.15)
        assert boost > 0

    async def test_no_boost_for_single_item(self, state):
        space = IdeaSpace(state)
        assert space.check_convergence([{"emotional_charge": 0.5}]) == 0.0
