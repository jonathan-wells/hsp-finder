# HSP Finder

Fast(ish) protein sequence alignment using sliding windows and E-value filtering.

## Install

```bash
uv sync
```

## Usage

```bash
uv run src/hsp_finder/aligner.py query.fasta database.fasta
```

## Output

BLAST-style tab-delimited format with columns: qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, score
