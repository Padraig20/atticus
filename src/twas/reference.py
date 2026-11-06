"""Cache aligned reference genotypes for rebuilding student covariance matrices."""

import json
from pathlib import Path

import h5py
import numpy as np

from src.distillation.genotypes import Chromosome, TEMPLATE


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def write_strings(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())


def model_snps(group, chrom):
    snps = group["snp_ids"].asstr()[:].tolist()
    positions = group["positions"][:]
    effects, others = group["effect_alleles"].asstr()[:], group["other_alleles"].asstr()[:]
    if len(set(snps)) != len(snps) or any(len(values) != len(snps) for values in (positions, effects, others)):
        raise ValueError(f"Duplicate or unaligned SNP metadata in {group.name}")
    if group.attrs.get("coefficient_scale") != "raw_A1_dosage":
        raise ValueError(f"Expected raw additive dosage coefficients in {group.name}")
    if (not np.isfinite(positions).all() or (positions < 1).any()
            or (positions != positions.astype(np.int64)).any()):
        raise ValueError(f"Invalid SNP positions in {group.name}")
    for effect, other in zip(effects, others):
        if effect not in ("A", "C", "G", "T") or other not in ("A", "C", "G", "T") or effect == other:
            raise ValueError(f"Expected biallelic SNPs in {group.name}")
    return {snp: (str(chrom), int(position), str(effect), str(other))
            for snp, position, effect, other in zip(snps, positions, effects, others)}


class LDReference:
    """Cache aligned hard calls to reconstruct complete LD across draws and traits."""

    def __init__(self, cache, genotypes, samples, chromosomes, model_path):
        if len(samples) < 2 or len(set(samples)) != len(samples):
            raise ValueError("LD reference requires at least two unique individuals")
        self.directory, self.samples = Path(genotypes), samples
        files = [fingerprint(self.directory / f"{TEMPLATE.format(chrom=chrom)}.{ext}")
                 for chrom in sorted(set(chromosomes), key=int) for ext in ("bed", "bim", "fam")]
        config = {"format_version": 1, "models": fingerprint(model_path),
                  "genotype_files": files, "sample_ids": samples, "covariance_ddof": 1,
                  "storage": "complete hard calls in model-effect-allele coding"}
        encoded = json.dumps(config, sort_keys=True)
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(cache, "a")
        if "config" in self.handle:
            if self.handle["config"].asstr()[()] != encoded:
                self.close()
                raise ValueError("LD cache inputs changed; use a new --model-cache directory")
        elif len(self.handle) or len(self.handle.attrs):
            self.close()
            raise ValueError("Unrecognized LD cache file; use a new --model-cache directory")
        else:
            self.handle.create_dataset("config", data=encoded, dtype=h5py.string_dtype())
            write_strings(self.handle, "sample_ids", samples)
            self.handle.create_group("genes")
        self.chromosome = None
        self.chrom = None

    def close(self):
        self.handle.close()

    def get(self, gene, chrom, catalog):
        snps = sorted(catalog, key=lambda snp: (catalog[snp][1], snp))
        if gene in self.handle["genes"]:
            saved = self.handle["genes"][gene]
            if saved.attrs.get("complete", False):
                if saved["snp_ids"].asstr()[:].tolist() != snps:
                    raise ValueError(f"LD cache SNP support changed for {gene}")
                return snps, saved["dosages"][:]
            del self.handle["genes"][gene]
        if self.chrom != chrom:
            self.chromosome = Chromosome(self.directory, chrom, self.samples)
            self.chrom = chrom
        reference = self.chromosome
        indices, flip = [], []
        for snp in snps:
            _, position, effect, other = catalog[snp]
            start, stop = np.searchsorted(reference.positions, position, side="left"), np.searchsorted(reference.positions, position, side="right")
            matches = np.flatnonzero(reference.bed.sid[start:stop] == snp) + start
            if len(matches) != 1:
                raise ValueError(f"Model SNP {snp} is absent or duplicated in the genotype reference")
            index = int(matches[0])
            alleles = (reference.bed.allele_1[index], reference.bed.allele_2[index])
            if alleles == (effect, other):
                flip.append(False)
            elif alleles == (other, effect):
                flip.append(True)
            else:
                raise ValueError(f"Model/reference allele mismatch for {snp}: {effect}/{other} versus {alleles}")
            indices.append(index)
        dosage = reference.bed.read(index=np.s_[reference.rows, indices], dtype="float64", num_threads=1)
        if not np.isin(dosage, [0, 1, 2]).all():
            raise ValueError(f"Missing/invalid genotype calls for {gene}; expected the complete-call distillation reference")
        dosage[:, flip] = 2 - dosage[:, flip]
        saved = self.handle["genes"].create_group(gene)
        saved.attrs.update(chrom=chrom, complete=False)
        write_strings(saved, "snp_ids", snps)
        write_strings(saved, "effect_alleles", [catalog[snp][2] for snp in snps])
        write_strings(saved, "other_alleles", [catalog[snp][3] for snp in snps])
        saved.create_dataset("dosages", data=dosage.astype(np.uint8), compression="gzip", compression_opts=1)
        saved.attrs["complete"] = True
        self.handle.flush()
        return snps, dosage
