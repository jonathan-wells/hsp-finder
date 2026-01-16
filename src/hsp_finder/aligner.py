"""High-Scoring Pair (HSP) alignment module for sequence similarity search.

This module implements a sliding window-based alignment algorithm for finding
high-scoring local alignments between protein sequences, similar to BLAST.
"""

import sys

import numpy as np
from numpy.typing import ArrayLike
import numba
from Bio import SeqIO
from Bio.Align import substitution_matrices

from hsp_finder.formatter import Alignment


class HSPFinder:
    """Finds high-scoring pairs (HSPs) between query and subject sequences.

    Uses a sliding window approach with dynamic programming to identify
    local alignments and calculates E-values for statistical significance.
    """

    def __init__(
        self,
        window_size: int = 80,
        minevalue: float = 0.01,
        scoring_system: str = "BLOSUM62",
        total_qlen: int | None = None,
        total_slen: int | None = None,
    ):
        """Initialize the HSPFinder with alignment parameters.

        Args:
            window_size: Length of the alignment window in amino acids.
            minevalue: Maximum E-value threshold for reporting alignments.
            scoring_system: Name of the substitution matrix (e.g., BLOSUM62).
            total_qlen: Total length of all query sequences for E-value calculation.
            total_slen: Total length of all subject sequences for E-value calculation.
        """
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
        """Align two sequences and return the best HSP if significant.

        Args:
            q_arr: Query sequence as a NumPy array of ASCII byte values.
            s_arr: Subject sequence as a NumPy array of ASCII byte values.
            qseqid: Query sequence identifier.
            sseqid: Subject sequence identifier.

        Returns:
            An Alignment object if E-value is below threshold, None otherwise.
        """
        scores = self.calc_scores_matrix(q_arr, s_arr)
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        evalue = self.calc_evalue(
            scores[i, j],
            self.total_qlen or len(q_arr),
            self.total_slen or len(s_arr)
        )
        if evalue < self.minevalue:
            alignment = Alignment(
                q_arr,
                s_arr,
                scores[i, j],
                evalue,
                i - 1,
                j - 1,
                self.window,
                qseqid,
                sseqid
            )
            return alignment

    def calc_scores_matrix(self, q_arr: ArrayLike, s_arr: ArrayLike) -> ArrayLike:
        """Calculate cumulative scoring matrix with sliding window adjustment.

        Computes cumulative alignment scores and subtracts scores from positions
        outside the sliding window to create a local alignment effect.

        Args:
            q_arr: Query sequence as a NumPy array of ASCII byte values.
            s_arr: Subject sequence as a NumPy array of ASCII byte values.

        Returns:
            A 2D array of alignment scores for each position pair.
        """
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
        """Calculate the E-value for a given alignment score.

        Uses the Karlin-Altschul statistics formula to compute the expected
        number of alignments with the given score occurring by chance.

        Args:
            score: Raw alignment score.
            query_len: Length of the query sequence.
            subject_len: Length of the subject sequence.
            k: Karlin-Altschul K parameter.
            lambda_: Karlin-Altschul lambda parameter.

        Returns:
            The E-value (expected number of chance alignments).
        """
        evalue = k * query_len * subject_len * np.exp(-lambda_ * score)
        return evalue

    @staticmethod
    def load_match_scores(scoring_system: str) -> ArrayLike:
        """Load a substitution matrix into a lookup table.

        Creates a 256x256 array indexed by ASCII values for fast score lookups
        during alignment.

        Args:
            scoring_system: Name of the substitution matrix (e.g., BLOSUM62).

        Returns:
            A 2D array of match scores indexed by ASCII byte values.
        """
        matrix = substitution_matrices.load(scoring_system)
        match_scores = np.zeros((256, 256), dtype=np.int8)
        for i, aa1 in enumerate(matrix.alphabet):
            for j, aa2 in enumerate(matrix.alphabet):
                match_scores[ord(aa1), ord(aa2)] = matrix[i, j]
        return match_scores

    @staticmethod
    @numba.njit(fastmath=True, cache=True)
    def _sum_diagonals(q_arr, s_arr, m, n, match_scores) -> ArrayLike:
        """Compute cumulative diagonal alignment scores using dynamic programming.

        JIT-compiled function that efficiently calculates cumulative scores
        along all diagonals of the alignment matrix.

        Args:
            q_arr: Query sequence as a NumPy array of ASCII byte values.
            s_arr: Subject sequence as a NumPy array of ASCII byte values.
            m: Length of the query sequence.
            n: Length of the subject sequence.
            match_scores: 2D lookup table of match scores indexed by ASCII values.

        Returns:
            A (m+1) x (n+1) array of cumulative alignment scores.
        """
        scores = np.zeros((m + 1, n + 1), dtype=np.int16)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                scores[i, j] = scores[i - 1, j - 1] + \
                    match_scores[q_arr[i - 1], s_arr[j - 1]]
        return scores


def search_database(
    query_fasta: str,
    subject_fasta: str,
    window_size: int = 80,
    evalue_threshold: float = 1e-2,
    identity_threshold: float = 40.0,
    output_file: str | None = None,
):
    """Search a query sequence against a database of sequences in a FASTA file.

    Performs an all-vs-all comparison between query and subject sequences,
    reporting alignments that meet identity and length criteria.

    Args:
        query_fasta: Path to the FASTA file containing query sequences.
        subject_fasta: Path to the FASTA file containing subject sequences.
        window_size: Length of the alignment window in amino acids.
        evalue_threshold: Maximum E-value for reporting alignments.
        identity_threshold: Minimum percent identity for reporting alignments.
        output_file: Optional path to write results. If None, writes to stdout.
    """
    query_records = list(SeqIO.parse(query_fasta, "fasta"))
    subject_records = list(SeqIO.parse(subject_fasta, "fasta"))

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

    output_handle = open(output_file, 'w') if output_file else sys.stdout

    try:
        for qseqid, q_arr in query_data:
            for sseqid, s_arr in subject_data:
                aln = searcher.align(q_arr, s_arr, qseqid=qseqid, sseqid=sseqid)
                if aln and aln.identity >= identity_threshold and aln.aln_length == window_size:
                    output_handle.write(aln.to_blast_tab() + '\n')
                c += 1
                if c % 1000000 == 0 or c == nsearch:
                    prc = 100 * c / nsearch
                    sys.stderr.write(f"Completed {c}/{nsearch}. {prc:.2f}%\n")
    finally:
        if output_file:
            output_handle.close()

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
