import unittest
from pathlib import Path

from app.embeddings import LocalHashEmbeddingProvider
from app.service import PolicyRAGService
from app.vector_store import InMemoryVectorStore


ROOT = Path(__file__).resolve().parents[1]


class PolicyRAGTests(unittest.TestCase):
    def setUp(self):
        store = InMemoryVectorStore(LocalHashEmbeddingProvider())
        self.service = PolicyRAGService(store, min_score=0.08)
        self.service.ingest_directory(ROOT / "data")

    def test_greeting_uses_controlled_route(self):
        result = self.service.answer("Hello")
        self.assertEqual(result.route, "greeting")
        self.assertEqual(result.citations, [])

    def test_geo_filter_returns_india_policy(self):
        result = self.service.answer("How much annual leave do employees receive?", geo="India")
        self.assertTrue(result.grounded)
        self.assertIn("18 days", result.answer)
        self.assertTrue(all(citation["geo"] == "India" for citation in result.citations))

    def test_missing_evidence_refuses_to_guess(self):
        result = self.service.answer("What is the company car allowance?", geo="India")
        self.assertFalse(result.grounded)
        self.assertIn("sufficient policy evidence", result.answer)


if __name__ == "__main__":
    unittest.main()

