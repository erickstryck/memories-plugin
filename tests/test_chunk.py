"""Tests for the slicing. Pure, no network — it is the module that decides search
quality, so it is the one that most deserves tests with TEETH."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunk import (Chunk, chunk_text, is_probably_binary, mode_for_suffix,
                        pack_chunks, split_blocks)


class TestBoundaries(unittest.TestCase):
    def test_markdown_heading_opens_a_block(self):
        lines = ["intro\n", "# Title\n", "body\n"]
        self.assertIn(1, [start for start, _ in split_blocks(lines)])

    def test_paragraph_after_blank_line_opens_a_block(self):
        lines = ["one\n", "\n", "two\n"]
        starts = [start for start, _ in split_blocks(lines)]
        self.assertIn(2, starts)

    def test_top_level_code_definition_opens_a_block(self):
        lines = ["def a():\n", "    return 1\n", "def b():\n", "    return 2\n"]
        starts = [start for start, _ in split_blocks(lines)]
        self.assertIn(2, starts, "a line at column 0 after an indented line is a boundary")

    def test_empty_file_produces_no_block(self):
        self.assertEqual(split_blocks([]), [])

    def test_blocks_cover_the_whole_file_with_no_gap(self):
        lines = [f"line {i}\n" for i in range(20)]
        lines[5] = "\n"
        lines[12] = "# section\n"
        blocks = split_blocks(lines)
        self.assertEqual(blocks[0][0], 0)
        self.assertEqual(blocks[-1][1], len(lines))
        for previous, next_ in zip(blocks, blocks[1:]):
            self.assertEqual(previous[1], next_[0], "one block's end is the next one's start")


class TestPacking(unittest.TestCase):
    def test_line_numbers_are_1_based_and_inclusive(self):
        chunks = pack_chunks(["a\n", "b\n", "c\n"], target=10_000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 3))

    def test_chunk_text_matches_the_declared_lines(self):
        lines = [f"L{i}\n" for i in range(1, 31)]
        lines[9] = "\n"
        for chunk in pack_chunks(lines, target=40):
            expected = "".join(lines[chunk.start_line - 1:chunk.end_line]).strip("\n")
            self.assertEqual(chunk.text, expected,
                             "the line range has to reproduce the text — that is the contract "
                             "of locator mode, which tells someone to re-read that region")

    def test_oversized_block_falls_back_to_the_fixed_window(self):
        oversized = ["x" * 200 + "\n" for _ in range(100)]  # 20k chars in a single block
        chunks = pack_chunks(oversized, target=2000, hard_max=4000)
        self.assertGreater(len(chunks), 1, "a block above the ceiling has to be split")
        for t in chunks:
            self.assertLessEqual(len(t.text), 5000)

    def test_fixed_window_overlaps(self):
        oversized = ["y" * 300 + "\n" for _ in range(40)]
        chunks = pack_chunks(oversized, target=1500, hard_max=2000)
        pairs = list(zip(chunks, chunks[1:]))
        self.assertTrue(any(b.start_line <= a.end_line for a, b in pairs),
                        "consecutive windows have to overlap so the seam is not lost")

    def test_whitespace_only_chunk_is_discarded(self):
        self.assertEqual(pack_chunks(["\n", "   \n", "\t\n"]), [])

    def test_respects_the_target_size(self):
        lines = []
        for i in range(60):
            lines += [f"paragraph {i}\n", "\n"]
        for t in pack_chunks(lines, target=200, hard_max=1000):
            self.assertLessEqual(len(t.text), 1000)

    def test_no_chunk_loses_meaningful_content(self):
        lines = [f"unique content {i}\n" for i in range(50)]
        chunks = pack_chunks(lines, target=100)
        together = "\n".join(t.text for t in chunks)
        for i in range(50):
            self.assertIn(f"unique content {i}", together)


class TestMode(unittest.TestCase):
    def test_text_extension_becomes_a_locator(self):
        for suf in (".py", ".md", ".ts", ".LOG"):
            self.assertEqual(mode_for_suffix(suf), "locator")

    def test_unknown_extension_becomes_a_snapshot(self):
        for suf in (".pdf", ".docx", ""):
            self.assertEqual(mode_for_suffix(suf), "snapshot")


class TestBinary(unittest.TestCase):
    def test_detects_a_null_byte(self):
        self.assertTrue(is_probably_binary("abc\x00def"))
        self.assertFalse(is_probably_binary("ordinary text with an accent é"))


class TestChunkText(unittest.TestCase):
    def test_realistic_markdown_document(self):
        doc = "\n".join(
            f"## Section {i}\n\nContent of section {i}. " + ("word " * 40)
            for i in range(10)
        )
        chunks = chunk_text(doc, target=600, hard_max=2000)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(isinstance(t, Chunk) for t in chunks))
        self.assertEqual(chunks[0].start_line, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
