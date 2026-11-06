"""Run bundled S-PrediXcan in the gene order of native covariance files.

Only input order and SNP support are adapted; the bundled program performs
all association calculations. This adapter runs in its own subprocess.
"""

import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("gene_order", type=Path)
    args, forwarded = parser.parse_known_args()
    order = args.gene_order.read_text().splitlines()
    if not order or len(set(order)) != len(order) or any(not gene or any(c.isspace() for c in gene) for gene in order):
        raise ValueError("Expected unique gene IDs in the covariance order file")
    sys.path.insert(0, str(args.script.resolve().parent))
    from metax.metaxcan import Utilities

    build_context = Utilities.build_context

    def ordered_context(*context_args, **context_kwargs):
        context = build_context(*context_args, **context_kwargs)
        genes = set(context.extra.gene)
        if genes - set(order):
            raise ValueError("Model genes are missing from the covariance order file")
        context.extra = context.extra.set_index("gene").loc[[gene for gene in order if gene in genes]].reset_index()
        streamer = context.covariance.streamer

        def checked_covariance():
            for frame in streamer:
                gene = frame.GENE.iloc[0]
                if gene in context.weight_data:
                    expected = set(context.get_weights(gene).rsid)
                    actual = set(frame.RSID1) | set(frame.RSID2)
                    if actual != expected:
                        raise ValueError(f"Covariance SNP support differs from model weights for {gene}")
                yield frame

        context.covariance.streamer = checked_covariance()
        return context

    Utilities.build_context = ordered_context
    sys.argv = [str(args.script), *forwarded]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
