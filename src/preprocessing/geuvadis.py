"""Prepare autosomal Geuvadis genotypes from Zenodo record 16538482."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RECORD = "https://zenodo.org/api/records/16538482"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def read_ids(path):
    ids = Path(path).read_text().split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected a nonempty list of unique IDs in {path}")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item) for item in ids):
        raise ValueError(f"Invalid ID in {path}")
    return ids


def download(entry, destination):
    if destination.exists():
        with destination.open("rb") as handle:
            if "md5:" + hashlib.file_digest(handle, "md5").hexdigest() == entry["checksum"]:
                return
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    run("curl", "--fail", "--location", "--retry", 5, "--continue-at", "-",
        "--output", partial, entry["links"]["self"])
    with partial.open("rb") as handle:
        checksum = "md5:" + hashlib.file_digest(handle, "md5").hexdigest()
    if checksum != entry["checksum"]:
        partial.unlink()
        raise ValueError(f"Checksum mismatch for {destination.name}; rerun to download again")
    partial.replace(destination)


def chromosome_complete(output, chrom):
    vcf = output / "cohort" / f"chr{chrom}.vcf.gz"
    bed = output / "plink" / f"chr{chrom}.GRCh38.biallelic"
    products = [vcf, Path(f"{vcf}.tbi"), *(Path(f"{bed}.{ext}") for ext in ("bed", "bim", "fam"))]
    return (output / f".chr{chrom}.done").exists() and all(path.is_file() for path in products)


def prepare_chromosome(source, chrom, output, samples_file, threads):
    vcf = output / "cohort" / f"chr{chrom}.vcf.gz"
    bed = output / "plink" / f"chr{chrom}.GRCh38.biallelic"
    marker = output / f".chr{chrom}.done"
    if chromosome_complete(output, chrom):
        return vcf
    marker.unlink(missing_ok=True)
    vcf.parent.mkdir(parents=True, exist_ok=True)
    bed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        subset = temporary / "subset.vcf.gz"
        rename = temporary / "chromosomes.tsv"
        rename.write_text(f"{chrom}\tchr{chrom}\n")
        # Subset before conversion: missing samples must fail, never be silently dropped.
        with subprocess.Popen(
            ["bcftools", "view", "-S", str(samples_file), "-Ou", str(source)],
            stdout=subprocess.PIPE,
        ) as view:
            try:
                run("bcftools", "annotate", "--rename-chrs", rename, "-Oz",
                    "--threads", threads, "-o", subset, stdin=view.stdout)
            finally:
                view.stdout.close()
            if view.wait():
                raise RuntimeError(f"Sample subsetting failed for chromosome {chrom}")
        run("bcftools", "index", "-t", subset)
        prefix = temporary / "genotypes"
        run("plink2", "--vcf", subset, "--make-bed", "--max-alleles", 2,
            "--snps-only", "just-acgt", "--set-all-var-ids", "@:#:$r:$a",
            "--const-fid", 0, "--output-chr", 26, "--threads", threads,
            "--memory", 4096, "--out", prefix)
        subset.replace(vcf)
        Path(f"{subset}.tbi").replace(f"{vcf}.tbi")
        for ext in ("bed", "bim", "fam"):
            Path(f"{prefix}.{ext}").replace(f"{bed}.{ext}")
    marker.touch()
    return vcf


def split_individuals(vcfs, samples_file, output):
    individuals = output / "individuals"
    samples = read_ids(samples_file)
    marker = output / ".individuals.done"
    if marker.exists() and all(
        (individuals / f"{sample}.vcf.gz{suffix}").is_file()
        for sample in samples for suffix in ("", ".tbi")
    ):
        return
    marker.unlink(missing_ok=True)
    individuals.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        with subprocess.Popen(
            ["bcftools", "concat", "-Ou", *map(str, vcfs)], stdout=subprocess.PIPE,
        ) as concat:
            try:
                # Hom-reference records do not change consensus; retain missing calls.
                run("bcftools", "+split", "-", "-S", samples_file,
                    "-i", 'GT="alt" || GT="mis"', "-k", "FMT/GT", "-Oz", "-Wtbi", "-o", temporary,
                    stdin=concat.stdout)
            finally:
                concat.stdout.close()
            if concat.wait():
                raise RuntimeError("Chromosome concatenation failed")
        for sample in samples:
            for suffix in (".vcf.gz", ".vcf.gz.tbi"):
                path = Path(temporary) / f"{sample}{suffix}"
                path.replace(individuals / path.name)
    marker.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "data/1000G_EUR_503.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/geuvadis")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    samples = read_ids(args.samples)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples_file = output / "samples.txt"
    if samples_file.exists() and read_ids(samples_file) != samples:
        parser.error("Sample list changed; use a new --output directory")
    samples_file.write_text("\n".join(samples) + "\n")
    metadata = json.loads(run("curl", "-fsSL", "--retry", 5, RECORD, capture_output=True).stdout)
    files = {entry["key"]: entry for entry in metadata["files"]}
    vcfs = []
    for chrom in range(1, 23):
        print(f"Preparing chromosome {chrom}/22", flush=True)
        name = f"ALL.chr{chrom}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz"
        source = output / name
        # Completed chromosomes need no download. Incomplete products are rebuilt.
        if not chromosome_complete(output, chrom):
            (output / ".individuals.done").unlink(missing_ok=True)
            download(files[name], source)
        vcfs.append(prepare_chromosome(source, chrom, output, samples_file, args.threads))
        source.unlink(missing_ok=True)
    split_individuals(vcfs, samples_file, output)
    print(f"Ready: {output / 'plink'} and {output / 'individuals'}")


if __name__ == "__main__":
    main()
