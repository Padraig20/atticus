"""Persist each fitted student's reference covariance before releasing genotypes."""

import numpy as np


def covariance_blocks(dosage, block_size=128):
    """Yield packed upper-triangle row blocks without allocating a dense LD matrix."""
    dosage = np.asarray(dosage, dtype=np.float64)
    if dosage.ndim != 2 or len(dosage) < 2 or not np.isfinite(dosage).all():
        raise ValueError("Covariance requires complete dosages for at least two reference donors")
    centered = dosage - dosage.mean(axis=0)
    n_snps = dosage.shape[1]
    for start in range(0, n_snps, block_size):
        stop = min(start + block_size, n_snps)
        block = centered[:, start:stop].T @ centered[:, start:] / (len(dosage) - 1)
        values = np.concatenate([block[i, i:] for i in range(stop - start)])
        yield start, stop, values


def save_covariances(student, dosage):
    """Save packed upper triangles for each draw's nonzero projected weights.

    Dosage columns must follow student/snp_ids and include ALL reference donors,
    not just the fitting fold or a bootstrap sample. Packing by draw avoids a
    dense covariance over the much larger union of all selected SNPs.
    """
    if "ld" in student and student["ld"].attrs.get("complete", False):
        return
    weights = student["coefficient_draws"][:]
    dosage = np.asarray(dosage, dtype=np.float64)
    if (dosage.ndim != 2 or dosage.shape[0] < 2 or dosage.shape[1] != weights.shape[2]
            or not np.isfinite(dosage).all()):
        raise ValueError(f"Expected complete, aligned reference dosages for {student.name}")
    if "ld" in student:
        del student["ld"]
    group = student.create_group("ld")
    group.attrs.update(format_version=1, complete=False, n_samples=len(dosage), ddof=1,
                       reference="all_distillation_individuals", triangle="upper_including_diagonal")
    indices = [np.flatnonzero(draw) for draw in weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])]
    counts = np.array([len(index) for index in indices], dtype=np.int64)
    snp_offsets = np.r_[0, np.cumsum(counts)]
    value_offsets = np.r_[0, np.cumsum(counts * (counts + 1) // 2)]
    group.create_dataset("snp_offsets", data=snp_offsets)
    group.create_dataset("value_offsets", data=value_offsets)
    group.create_dataset("snp_indices", data=np.concatenate(indices))
    values = group.create_dataset("values", shape=(int(value_offsets[-1]),), dtype=np.float64,
                                  **({"compression": "gzip", "compression_opts": 1} if value_offsets[-1] else {}))
    for draw, selected in enumerate(indices):
        if not len(selected):
            continue
        offset = int(value_offsets[draw])
        for _, _, block in covariance_blocks(dosage[:, selected]):
            values[offset:offset + len(block)] = block
            offset += len(block)
    group.attrs["complete"] = True
    student.file.flush()


def triangle_layout(student, member, bootstrap):
    group = student["ld"]
    if (not group.attrs.get("complete", False) or group.attrs.get("format_version") != 1
            or group.attrs.get("ddof") != 1 or group.attrs.get("n_samples", 0) < 2):
        raise ValueError(f"Incomplete or unsupported stored covariance in {student.name}")
    draw = member * student["coefficient_draws"].shape[1] + bootstrap
    start, stop = group["snp_offsets"][draw:draw + 2]
    indices = group["snp_indices"][start:stop]
    start, stop = group["value_offsets"][draw:draw + 2]
    if stop - start != len(indices) * (len(indices) + 1) // 2 or start < 0 or stop > len(group["values"]):
        raise ValueError(f"Invalid stored covariance triangle in {student.name}")
    return group, indices, int(start), int(stop)


def saved_triangle_blocks(student, member, bootstrap, block_size=128):
    """Return SNP indices and a bounded-memory iterator over the saved triangle."""
    group, indices, offset, _ = triangle_layout(student, member, bootstrap)

    def blocks():
        position = offset
        n_snps = len(indices)
        for start in range(0, n_snps, block_size):
            stop = min(start + block_size, n_snps)
            count = (stop - start) * (2 * n_snps - start - stop + 1) // 2
            values = group["values"][position:position + count]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite stored covariance in {student.name}")
            yield start, stop, values
            position += count

    return indices, blocks()
