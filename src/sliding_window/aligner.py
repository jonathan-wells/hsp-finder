import sys

import numpy as np
from numpy.typing import ArrayLike
import numba
from Bio import SeqIO
from Bio.Align import substitution_matrices

from sliding_window.formatter import Alignment


class WindowSearcher:
    def __init__(
        self,
        window_size: int = 80,
        identity_threshold: float = 40.0,
        scoring_system: str = "BLOSUM62",
    ):
        self.window: int = window_size
        self.idt = identity_threshold
        self.minscore = np.ceil((self.window * self.idt) / 100).astype(np.int8)
        self.scoring_system = scoring_system
        self.match_scores = self.load_scores_dict(scoring_system)

    def align(
        self,
        q_arr: ArrayLike,
        s_arr: ArrayLike,
        qseqid: str,
        sseqid: str
    ) -> Alignment | None:
        scores = self.calc_scoring_matrix(q_arr, s_arr)
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[i, j] > self.minscore:
            alignment = Alignment(
                q_arr, s_arr, scores[i, j], i - 1, j - 1, self.window, qseqid, sseqid
            )
            return alignment

    def calc_scoring_matrix(self, q_arr: ArrayLike, s_arr: ArrayLike) -> ArrayLike:
        m = q_arr.shape[0]
        n = s_arr.shape[0]
        scores = self._sum_diagonals(q_arr, s_arr, m, n, self.match_scores)
        window = self.window
        scores[window:, window:] -= scores[:-window, :-window]
        return scores

    @staticmethod
    def load_match_scores(scoring_system: str) -> dict[str, int]:
        matrix = substitution_matrices.load(scoring_system)
        match_scores = {}
        for i, aa1 in enumerate(matrix.alphabet):
            for j, aa2 in enumerate(matrix.alphabet):
                match_scores[(aa1, aa2)] = matrix[aa1, aa2]
        return match_scores

    @staticmethod
    @numba.njit(fastmath=True, cache=True)
    def _sum_diagonals(q_arr, s_arr, m, n, match_scores) -> ArrayLike:
        scores = np.zeros((m + 1, n + 1), dtype=np.int8)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                scores[i, j] = scores[i - 1, j - 1] + \
                    match_scores[q_arr[i - 1],s_arr[j - 1]]
        return scores


def search_database(
    query_fasta: str,
    database_fasta: str,
    window_size: int = 80,
    identity_threshold: float = 40.0,
):
    """Search a query sequence against a database of sequences in a FASTA file."""
    query_records = list(SeqIO.parse(query_fasta, "fasta"))
    subject_records = list(SeqIO.parse(database_fasta, "fasta"))
    query_data = [
        (rec.id, np.array([ord(c) for c in str(rec.seq)], dtype=np.int8))
        for rec in query_records
        if len(rec.seq) >= window_size
    ]
    subject_data = [
        (rec.id, np.array([ord(c) for c in str(rec.seq)], dtype=np.int8))
        for rec in subject_records
        if len(rec.seq) >= window_size
    ]

    n, m = len(query_data), len(subject_data)
    nsearch = n * m
    sys.stderr.write(f"Loaded {n} query sequences and {m} subject sequences.\n")
    sys.stderr.write(f"Performing {nsearch} searches...\n")

    c = 0
    searcher = WindowSearcher(window_size, identity_threshold)
    for qseqid, q_arr in query_data:
        for sseqid, s_arr in subject_data:
            aln = searcher.align(q_arr, s_arr, qseqid=qseqid, sseqid=sseqid)
            if aln:
                print(aln.to_blast_tab())
            c += 1
            if c % 1000000 == 0 or c == nsearch:
                prc = 100 * c / nsearch
                sys.stderr.write(f"Completed {c}/{nsearch}. {prc:.2f}%\n")

if __name__ == "__main__":
    query_db = sys.argv[1]
    subject_db = sys.argv[2]
    results = search_database(query_db, subject_db, 80, 40.0)

    # s1 = 'YQLPKSISELNLERGAPFSHYDQHSNNPTPMTKETI'
    # s2 = 'MERGAPFSHYMDDGAVGAPFSHYDQHSNNPTEIREEQRRLYMDDGAVEIIVAQQQPKETI'
    # s1_arr = np.array([ord(c) for c in s1], dtype=np.int8)
    # s2_arr = np.array([ord(c) for c in s2], dtype=np.int8)
    # sws = WindowSearcher(10, 30)
    # sws.align(s1_arr, s2_arr)
