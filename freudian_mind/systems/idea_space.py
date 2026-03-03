"""Idea Space — Vector-based semantic topology for impressions."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ..state import SharedState


class IdeaSpace:
    """
    A vector-based topology where impressions live as points in semantic space.

    - Condensation = clustering (nearby impressions merge)
    - Displacement = energy transfer along similarity gradients
    - Convergence = detectable clusters above density threshold
    """

    def __init__(self, state: SharedState):
        self.state = state

    def embed(self, text: str) -> list[float]:
        """
        SHA256-based pseudo-embedding for development.
        Replace with voyage-3 or sentence-transformers for production.
        """
        h = hashlib.sha256(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 64, 2)]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        va = np.array(a)
        vb = np.array(b)
        dot = float(np.dot(va, vb))
        mag_a = float(np.linalg.norm(va))
        mag_b = float(np.linalg.norm(vb))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    async def find_similar(
        self, embedding: list[float], threshold: float = 0.82
    ) -> list[tuple[dict, float]]:
        all_imps = await self.state.get_all_impressions_with_embeddings()
        results = []
        for imp in all_imps:
            imp_emb = (
                json.loads(imp["embedding"])
                if isinstance(imp["embedding"], str)
                else imp["embedding"]
            )
            if not imp_emb:
                continue
            sim = self.cosine_similarity(embedding, imp_emb)
            if sim >= threshold:
                results.append((imp, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    async def find_clusters(
        self, threshold: float = 0.82, min_cluster_size: int = 3
    ) -> list[list[dict]]:
        """Find clusters using union-find (single-linkage clustering)."""
        all_imps = await self.state.get_all_impressions_with_embeddings()
        if len(all_imps) < min_cluster_size:
            return []

        embeddings = []
        valid_imps = []
        for imp in all_imps:
            emb = (
                json.loads(imp["embedding"])
                if isinstance(imp["embedding"], str)
                else imp["embedding"]
            )
            if emb:
                embeddings.append(emb)
                valid_imps.append(imp)

        if len(valid_imps) < min_cluster_size:
            return []

        # Union-find
        n = len(valid_imps)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if self.cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                    union(i, j)

        # Collect clusters
        cluster_map: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            cluster_map.setdefault(root, []).append(i)

        return [
            [valid_imps[i] for i in members]
            for members in cluster_map.values()
            if len(members) >= min_cluster_size
        ]

    def condense_cluster(self, cluster: list[dict]) -> dict:
        """Merge a cluster into a single composite impression."""
        max_pressure = max(
            imp.get("pressure", 0) for imp in cluster
        )
        total_reinforcements = sum(
            imp.get("times_reinforced", 0) for imp in cluster
        )

        charges = [imp.get("emotional_charge", 0) for imp in cluster]
        pressures = [imp.get("pressure", 0) for imp in cluster]
        total_p = sum(pressures) or 1
        weighted_charge = sum(c * p for c, p in zip(charges, pressures)) / total_p

        return {
            "pressure": max_pressure,
            "emotional_charge": weighted_charge,
            "times_reinforced": total_reinforcements,
        }

    async def displace(
        self, imp: dict, alpha: float = 0.6
    ) -> dict | None:
        """Transfer pressure to a lower-charge neighbor."""
        emb = (
            json.loads(imp["embedding"])
            if isinstance(imp["embedding"], str)
            else imp["embedding"]
        )
        if not emb:
            return None

        similar = await self.find_similar(emb, threshold=0.5)
        imp_charge = abs(imp.get("emotional_charge", 0))

        for neighbor, sim in similar:
            if neighbor["id"] == imp.get("id"):
                continue
            n_charge = abs(neighbor.get("emotional_charge", 0))
            if n_charge < imp_charge:
                transfer = imp.get("pressure", 0) * alpha
                await self.state.reinforce_impression(neighbor["id"], transfer)
                new_pressure = imp.get("pressure", 0) * (1 - alpha)
                await self.state.update_impression_pressure(imp["id"], new_pressure)
                return neighbor
        return None

    def check_convergence(
        self,
        cluster: list[dict],
        coefficient: float = 0.15,
    ) -> float:
        """Calculate convergence boost for a cluster."""
        if len(cluster) < 2:
            return 0.0
        charges = [abs(imp.get("emotional_charge", 0)) for imp in cluster]
        mean_charge = sum(charges) / len(charges) if charges else 0
        return coefficient * math.log(len(cluster)) * mean_charge
