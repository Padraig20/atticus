"""Read the single PLINK fileset layout produced by preprocessing/geuvadis.py."""

from pathlib import Path

from bed_reader import open_bed
import numpy as np


TEMPLATE = "chr{chrom}.GRCh38.biallelic"


class Chromosome:
    def __init__(self, directory, chrom, samples):
        self.prefix = Path(directory) / TEMPLATE.format(chrom=chrom)
        self.bed = open_bed(f"{self.prefix}.bed", count_A1=True, num_threads=1)
        ids = self.bed.iid.tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate individual IDs in {self.prefix}.fam")
        lookup = {sample: i for i, sample in enumerate(ids)}
        missing = set(samples) - set(lookup)
        if missing:
            raise ValueError(f"Genotype samples missing on chromosome {chrom}: {sorted(missing)}")
        self.rows = np.asarray([lookup[sample] for sample in samples])
        self.positions = self.bed.bp_position
        if (np.diff(self.positions) < 0).any():
            raise ValueError(f"BIM positions must be sorted in {self.prefix}.bim")

    def cis(self, tss, half_window, maf=0.05):
        start = np.searchsorted(self.positions, max(1, tss - half_window), side="left")
        stop = np.searchsorted(self.positions, tss + half_window, side="right")
        indices = np.arange(start, stop)
        if not indices.size:
            return np.empty((len(self.rows), 0)), {}
        dosage = self.bed.read(index=np.s_[self.rows, indices], dtype="float32", num_threads=1)
        # QC is defined on the genotype reference panel, independent of teacher targets.
        complete = np.isfinite(dosage).all(axis=0)
        frequencies = np.zeros(len(indices))
        frequencies[complete] = dosage[:, complete].mean(axis=0, dtype=np.float64) / 2
        a1, a2 = self.bed.allele_1[indices], self.bed.allele_2[indices]
        keep = complete & (np.minimum(frequencies, 1 - frequencies) >= maf)
        keep &= np.isin(a1, list("ACGT")) & np.isin(a2, list("ACGT")) & (a1 != a2)
        indices, dosage = indices[keep], dosage[:, keep]
        snps = self.bed.sid[indices]
        if len(set(snps)) != len(snps):
            raise ValueError(f"Duplicate SNP IDs in the cis-window at {self.prefix}:{tss}")
        return dosage, {"snp_ids": snps, "positions": self.positions[indices],
                        "effect_alleles": self.bed.allele_1[indices],
                        "other_alleles": self.bed.allele_2[indices]}
