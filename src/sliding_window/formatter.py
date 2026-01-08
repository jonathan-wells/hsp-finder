import numpy as np


class Alignment:
    def __init__(
        self,
        query: str | np.ndarray,
        subject: str | np.ndarray,
        score: int,
        i: int,
        j: int,
        window: int,
        qseqid: str | None = None,
        sseqid: str | None = None,
    ):
        if isinstance(query, np.ndarray):
            query = "".join([chr(c) for c in query])
        if isinstance(subject, np.ndarray):
            subject = "".join([chr(c) for c in subject])
        self.query = query
        self.subject = subject
        self.score = score
        self.i = i
        self.j = j
        self.window = window
        self.m = len(query)
        self.n = len(subject)
        self.qseqid = qseqid if qseqid else "query"
        self.sseqid = sseqid if sseqid else "subject"

    def to_blast_tab(self) -> str:
        """Output alignment in BLAST outfmt 6 style (tab-delimited).

        Columns: qseqid sseqid pident length mismatch gapopen qstart qend sstart send score
        """
        pident = 100.0 * self.score / self.window
        length = self.window

        # Calculate alignment coordinates (1-indexed)
        qstart = max(self.i - self.window + 1, 0) + 1
        qend = min(self.i, self.m) + 1
        sstart = max(self.j - self.window + 1, 0) + 1
        send = min(self.j, self.n) + 1

        return "\t".join(
            [
                str(self.qseqid),
                str(self.sseqid),
                f"{pident:.2f}",
                str(length),
                str(qstart),
                str(qend),
                str(sstart),
                str(send),
                str(self.score),
            ]
        )

    def pretty_print(self) -> str:
        start1, end1 = max(self.i - self.window, 0), min(self.i, self.m)
        start2, end2 = max(self.j - self.window, 0), min(self.j, self.n)

        lpad1, lpad2 = 0, 0
        rpad1, rpad2 = 0, 0
        if start1 == 0:
            lpad1 = self.window - self.i
        if start2 == 0:
            lpad2 = self.window - self.j
        if end1 == self.m:
            rpad1 = self.i - self.m
        if end2 == self.n:
            rpad2 = self.j - self.n

        sq1 = " " * lpad1 + self.query[start1:end1] + " " * rpad1
        sq2 = " " * lpad2 + self.subject[start2:end2] + " " * rpad2
        identity = 100.0 * self.score / self.window
        idmarkers = "".join(
            [
                "|" if sq1[k] == sq2[k] and sq1[k] != " " else " "
                for k in range(self.window)
            ]
        )

        output = ""
        output += f"{identity:.1f}%\n"
        output += f"{start1:<4}{sq1} {end1}\n"
        output += f"{'':<4}{idmarkers}\n"
        output += f"{start2:<4}{sq2} {end2}\n"

        return output
