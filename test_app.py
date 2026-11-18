import unittest
from unittest.mock import patch

import faiss
import numpy as np

import app


class ChapterRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            app.Chunk("Chapter 1 - Introduction.pdf", 1, "Introduction to the course."),
            app.Chunk("Chapter 1 - Introduction.pdf", 2, "The chapter's main ideas."),
            app.Chunk("Chapter 3 - Activity.pdf", 6, "This activity mentions chapter 1."),
            app.Chunk("Chapter 10 - Review.pdf", 1, "Review questions."),
        ]
        self.index = faiss.IndexFlatIP(2)
        # Other chapters deliberately rank above Chapter 1.
        self.index.add(np.array([[0, 1], [0.1, 0.9], [1, 0], [0.9, 0.1]], dtype="float32"))
        self.embedding = patch.object(
            app, "embed", return_value=np.array([[1, 0]], dtype="float32")
        )
        self.embedding.start()
        self.addCleanup(self.embedding.stop)

    def test_named_chapter_excludes_other_chapters(self):
        for question in ("explain chapter 1", "summarize Chapter 01", "explain chapter1"):
            with self.subTest(question=question):
                results = app.retrieve(question, self.index, self.chunks)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(chunk.filename.startswith("Chapter 1 -") for chunk in results))

    def test_missing_chapter_does_not_fall_back_to_other_chapters(self):
        results = app.retrieve("explain chapter 2", self.index, self.chunks)
        self.assertEqual(results, [])
        with patch.object(app.requests, "post") as post:
            self.assertEqual(app.generate_answer("explain chapter 2", results), (app.INSUFFICIENT, []))
            post.assert_not_called()

    def test_general_question_still_returns_top_three(self):
        results = app.retrieve("What does the activity describe?", self.index, self.chunks)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].filename, "Chapter 3 - Activity.pdf")


if __name__ == "__main__":
    unittest.main()
