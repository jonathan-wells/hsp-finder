import numpy as np


class Alignment:
    def __init__(
        self,
        query: str | np.ndarray,
        subject: str | np.ndarray,
        score: int,
        evalue: float,
        i: int,
        j: int,
        window: int,
        qseqid: str,
        sseqid: str
    ):
        if isinstance(query, np.ndarray):
            query = "".join([chr(c) for c in query])
        if isinstance(subject, np.ndarray):
            subject = "".join([chr(c) for c in subject])
        self.query = query
        self.subject = subject
        self.m = len(query)
        self.n = len(subject)
        self.score = score
        self.evalue = evalue
        self.window = window

        # Calculate the actual alignment boundaries (0-indexed)
        qstart_0 = max(i - self.window + 1, 0)
        qend_0 = min(i + 1, self.m)
        sstart_0 = max(j - self.window + 1, 0)
        send_0 = min(j + 1, self.n)

        # Calculate actual lengths available in each sequence
        qlen = qend_0 - qstart_0
        slen = send_0 - sstart_0

        # Use the minimum length to ensure alignments match
        self.aln_length = min(qlen, slen)

        # Adjust boundaries to match the alignment length (keeping the end positions)
        qstart_0 = qend_0 - self.aln_length
        sstart_0 = send_0 - self.aln_length

        # Convert to 1-indexed for display
        self.qstart = qstart_0 + 1
        self.qend = qend_0
        self.sstart = sstart_0 + 1
        self.send = send_0

        self.qseqid = qseqid
        self.sseqid = sseqid
        self.query_aln = self.query[qstart_0:qend_0]
        self.subject_aln = self.subject[sstart_0:send_0]

    @property
    def identity(self) -> float:
        """Calculate the percent identity for a given alignment."""
        matches = sum(1 for q, s in zip(self.query_aln, self.subject_aln) if q == s)
        identity = 100.0 * matches / self.aln_length
        return identity

    @property
    def mismatches(self) -> float:
        """Calculate the number of mismatches for a given alignment."""
        return sum(1 for q, s in zip(self.query_aln, self.subject_aln) if q != s)

    def to_blast_tab(self) -> str:
        """Output alignment in BLAST outfmt 6 style (tab-delimited).

        Columns: qseqid sseqid pident length mismatch gapopen qstart qend sstart send score
        """
        return "\t".join(
            [
                str(self.qseqid),
                str(self.sseqid),
                f"{self.identity:.2f}",
                str(self.aln_length),
                str(self.mismatches),
                "0",  # gapopen is always 0 for this aligner
                str(self.qstart),
                str(self.qend),
                str(self.sstart),
                str(self.send),
                f"{self.evalue:.2e}",
                str(self.score),
            ]
        )

    def pretty_print(self) -> str:
        idmarkers = "".join(
            [
                "|" if self.query_aln[k] == self.subject_aln[k] else " "
                for k in range(self.aln_length)
            ]
        )

        output = ""
        output += f"{self.identity:.1f}%\n"
        output += f"{self.qstart:<4} {self.query_aln} {self.qend}\n"
        output += f"{'':<4} {idmarkers}\n"
        output += f"{self.sstart:<4} {self.subject_aln} {self.send}\n"

        return output
