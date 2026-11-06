"""Subset local GWAS tables and translate coordinate IDs for MetaXcan."""

from collections import Counter
import csv
import gzip
from pathlib import Path
import re

import numpy as np


COORDINATE_ID = re.compile(r"^(?:chr)?([0-9]+)[:_]([0-9]+)[:_]([ACGT])[:_]([ACGT])$", re.IGNORECASE)


def coordinate_key(chrom, position, effect, other):
    return int(str(chrom).lower().removeprefix("chr")), int(position), tuple(sorted((effect.upper(), other.upper())))


def read_gwas(path, catalog, *, snp_column="ID", effect_column="ALT", other_column="REF", beta_column="ES", se_column="SE"):
    """Return {model SNP ID: (effect allele, other allele, beta, SE)}.

    Matching uses an exact ID or coordinate plus both alleles. Allele swaps
    are recorded but left to MetaXcan to align. No strand complementation or
    genome-build conversion is inferred; inputs must use the same build.
    """
    coordinates = {}
    for snp, (chrom, position, effect, other) in catalog.items():
        key = coordinate_key(chrom, position, effect, other)
        if key in coordinates and coordinates[key] != snp:
            raise ValueError(f"Ambiguous model SNP IDs at {key}")
        coordinates[key] = snp
    columns = [snp_column, effect_column, other_column, beta_column, se_column]
    if len(set(columns)) != len(columns):
        raise ValueError("GWAS column names must be distinct")
    counts = Counter(rows=0, unmatched=0, allele_mismatch=0, invalid_statistics=0,
                     exact_id_matches=0, coordinate_matches=0, allele_swaps=0)
    aligned, seen = {}, set()
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        if len(set(header)) != len(header) or not set(columns) <= set(header):
            raise ValueError(f"GWAS must have unique columns including {columns}; found {header}")
        indices = [header.index(column) for column in columns]
        for line_number, fields in enumerate(reader, start=2):
            if not fields:
                continue
            counts["rows"] += 1
            if len(fields) != len(header):
                raise ValueError(f"GWAS row {line_number} has {len(fields)} fields; expected {len(header)}")
            source, effect, other, beta_text, se_text = (fields[i].strip() for i in indices)
            effect, other = effect.upper(), other.upper()
            snp = source if source in catalog else None
            category = "exact_id_matches"
            if snp is None:
                match = COORDINATE_ID.fullmatch(source)
                if match:
                    chrom, position, first, second = match.groups()
                    key = coordinate_key(chrom, position, first, second)
                    if key[2] != tuple(sorted((effect, other))):
                        counts["allele_mismatch"] += 1
                        continue
                    snp = coordinates.get(key)
                    category = "coordinate_matches"
            if snp is None:
                counts["unmatched"] += 1
                continue
            _, _, expected_effect, expected_other = catalog[snp]
            if (effect, other) == (expected_effect, expected_other):
                swapped = False
            elif (effect, other) == (expected_other, expected_effect):
                swapped = True
            else:
                counts["allele_mismatch"] += 1
                continue
            if snp in seen:
                raise ValueError(f"Multiple GWAS rows map to model SNP {snp}; resolve duplicate statistics")
            seen.add(snp)
            try:
                beta, se = float(beta_text), float(se_text)
            except ValueError:
                counts["invalid_statistics"] += 1
                continue
            if not np.isfinite(beta) or not np.isfinite(se) or se <= 0 or not np.isfinite(beta / se):
                counts["invalid_statistics"] += 1
                continue
            aligned[snp] = (effect, other, beta, se)
            counts[category] += 1
            counts["allele_swaps"] += swapped
    if not aligned:
        raise ValueError("No valid GWAS statistics match the model SNP IDs/coordinates and alleles; check columns and genome build")
    return aligned, dict(counts, matched_snps=len(aligned), model_snps=len(catalog))


def write_gwas(path, records):
    """Keep original GWAS effects; MetaXcan performs the actual alignment."""
    partial = Path(str(path) + ".partial")
    with gzip.open(partial, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["SNP", "A1", "A2", "BETA", "SE"])
        writer.writerows((snp, *values) for snp, values in sorted(records.items()))
    partial.replace(path)
