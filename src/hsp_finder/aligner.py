import sys

import numpy as np
from numpy.typing import ArrayLike
import numba
from Bio import SeqIO
from Bio.Align import substitution_matrices

from hsp_finder.formatter import Alignment


class HSPFinder:
    def __init__(
        self,
        window_size: int = 80,
        minevalue: float = 0.01,
        scoring_system: str = "BLOSUM62",
        total_qlen: int | None = None,
        total_slen: int | None = None,
    ):
        self.window: int = window_size
        self.minevalue = minevalue
        self.scoring_system = scoring_system
        self.total_qlen = total_qlen
        self.total_slen = total_slen
        self.match_scores = self.load_match_scores(scoring_system)

    def align(
        self,
        q_arr: ArrayLike,
        s_arr: ArrayLike,
        qseqid: str,
        sseqid: str
    ) -> Alignment | None:
        scores = self.calc_scores_matrix(q_arr, s_arr)
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        evalue = self.calc_evalue(
            scores[i, j],
            self.total_qlen or len(q_arr),
            self.total_slen or len(s_arr)
        )
        if evalue < self.minevalue:
            alignment = Alignment(
                q_arr, s_arr, scores[i, j], evalue, i - 1, j - 1, self.window, qseqid, sseqid
            )
            return alignment

    def calc_scores_matrix(self, q_arr: ArrayLike, s_arr: ArrayLike) -> ArrayLike:
        m = q_arr.shape[0]
        n = s_arr.shape[0]
        scores = self._sum_diagonals(q_arr, s_arr, m, n, self.match_scores)
        window = self.window
        scores[window:, window:] -= scores[:-window, :-window]
        return scores


    @staticmethod
    def calc_evalue(
        score: int,
        query_len: int,
        subject_len: int,
        k: float = 0.134,
        lambda_: float = 0.318,
    ) -> float:
        """Calculate the E-value for a given alignment score."""
        evalue = k * query_len * subject_len * np.exp(-lambda_ * score)
        return evalue

    @staticmethod
    def load_match_scores(scoring_system: str) -> ArrayLike:
        matrix = substitution_matrices.load(scoring_system)
        match_scores = np.zeros((256, 256), dtype=np.int8)
        for i, aa1 in enumerate(matrix.alphabet):
            for j, aa2 in enumerate(matrix.alphabet):
                match_scores[ord(aa1), ord(aa2)] = matrix[i, j]
        return match_scores

    @staticmethod
    @numba.njit(fastmath=True, cache=True)
    def _sum_diagonals(q_arr, s_arr, m, n, match_scores) -> ArrayLike:
        scores = np.zeros((m + 1, n + 1), dtype=np.int16)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                scores[i, j] = scores[i - 1, j - 1] + \
                    match_scores[q_arr[i - 1], s_arr[j - 1]]
        return scores


def search_database(
    query_fasta: str,
    database_fasta: str,
    window_size: int = 80,
    evalue_threshold: float = 1e-2,
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
    total_qlen = sum(len(seq) for seq in query_data)
    total_slen = sum(len(seq) for seq in subject_data)


    n, m = len(query_data), len(subject_data)
    nsearch = n * m
    sys.stderr.write(f"Loaded {n} query sequences and {m} subject sequences.\n")
    sys.stderr.write(f"Performing {nsearch} searches...\n")

    c = 0

    searcher = HSPFinder(
        window_size,
        evalue_threshold,
        total_qlen=total_qlen,
        total_slen=total_slen
    )

    for qseqid, q_arr in query_data:
        for sseqid, s_arr in subject_data:
            aln = searcher.align(q_arr, s_arr, qseqid=qseqid, sseqid=sseqid)
            if aln and aln.identity >= 40.0 and aln.aln_length == window_size:
                print(aln.to_blast_tab())
            c += 1
            if c % 1000000 == 0 or c == nsearch:
                prc = 100 * c / nsearch
                sys.stderr.write(f"Completed {c}/{nsearch}. {prc:.2f}%\n")

if __name__ == "__main__":
    query_db = sys.argv[1]
    subject_db = sys.argv[2]
    search_database(query_db, subject_db)

    # s1 = 'YQLPKSISELNLERKAPFSHYDQHSNNPKPMTKETI'
    # s2 = 'MERGAPFSHYMDDGAVGAPFSHYDQHSNNPTEIREEQRRLYMDDGAVEIIVAQQQPKETI'
    # s1_arr = np.array([ord(c) for c in s1], dtype=np.int8)
    # s2_arr = np.array([ord(c) for c in s2], dtype=np.int8)
    # sws = HSPFinder(20, 0.01)
    # alignment = sws.align(s1_arr, s2_arr, 's1', 's2')
    # if alignment:
    #     print(alignment.pretty_print())
    #     print(alignment.to_blast_tab())
