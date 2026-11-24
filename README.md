# ThirdAllele
A python package for estimating error rates in sequence alignments with third alleles. 

## Science

Errors in sequencing data can occur in a variety of ways, including optical error during sequencing,
PCR or ligation artifacts introduced during library prep, or bioinformatic/biological errors
related to incorrect alignment and paralogous regions of the genome. Standard bioinformatic
approaches like baseQ only address the simple optical error model, but for many purposes we
need more accurate error estimates covering the full spectrum of sources. 

Third allele borrows an approach from GATK's `CalculateContamination` method to estimate 
error across mechanisms while being robust to cross-contamination and (most)
somatic mutation. The idea is to count the rate of "third" alleles observed at germline
heterozygous sites. Because most common SNP positions have just two alleles circulating
in the population at significant frequencies, cross-contamination is unlikely to introduce
new alleles not observed in the primary patient data. 

## Installation

ThirdAllele needs Python 3.10+ and [bcftools](https://samtools.github.io/bcftools/) on your `PATH`.
It isn't on PyPI yet, so install it from a clone of this repository.

### With conda (recommended)

`environment.yml` pins bcftools and the Python dependencies:

```bash
git clone https://github.com/cjbattey/ThirdAllele.git
cd ThirdAllele
conda env create -f environment.yml   # or: mamba env create -f environment.yml
conda activate third-allele-errors
pip install --no-deps -e .
```

### With pip

If bcftools is already installed some other way:

```bash
git clone https://github.com/cjbattey/ThirdAllele.git
cd ThirdAllele
python -m venv .venv && source .venv/bin/activate
pip install .
```

Use `pip install -e .` instead if you plan to edit the code.

Either way, check the install with:

```bash
third-allele-errors --help
```

## Usage

```bash
third-allele-errors sample.bam ref.fa -o errors.tsv -s sites.tsv
```

`python -m third_allele` runs the same command. From Python:

```python
from third_allele import ThirdAlleleErrorEstimator

est = ThirdAlleleErrorEstimator(bam="sample.bam", reference="ref.fa", regions="targets.bed")
table = est.error_table()   # polars DataFrame
sites = est.sites_report()
```

### Inputs

* **BAM/CRAM**: coordinate-sorted and indexed (`.bai`/`.crai`), one sample.
* **Reference FASTA**: the one the reads were aligned to, indexed (`.fai`).
* **Regions** (optional, `-r`): a region string such as `chr1:1-1000000`, or a BED file. Restricting to
  well-covered targets (e.g. a capture panel) is much faster than a whole-genome pileup.

Het SNPs are called from the BAM itself with `bcftools mpileup | bcftools call`, so no VCF is needed.
Run `third-allele-errors --help` for all filters (mapping/base quality, depth, het VAF range) and model
settings.

### Outputs

The error table is always printed to stdout. `-o` also writes it as TSV, with one row per
substitution class plus an `overall` row:

| column | meaning |
|---|---|
| `mutation_type` | substitution folded onto the pyrimidine strand: `C>A`, `C>G`, `C>T`, `T>A`, `T>C`, `T>G` |
| `errors` | third-allele reads assigned to this class, down-weighted by their artefact probability |
| `opportunities` | reads in which this substitution could have been seen, weighted the same way |
| `error_rate`, `ci_low`, `ci_high` | rate per base and its exact (Clopper–Pearson) 95% interval |
| `lob_alt_reads` | alt reads needed to call a variant at `lod_depth` with ≤ 5% false positives |
| `lod_vaf` | VAF detected with 95% probability at that threshold |
| `lod_depth` | depth used for the LoD (default: median depth of the sites used) |

`overall` is the mean per-substitution rate. The chance that a base is miscalled as *any* other base
is about 3× this.

`-s` writes a per-site table: depth, germline allele counts, third-allele reads, QC flags, and how
many of the site's third-allele reads the model attributes to artefacts (mis-mapping, multi-allelic
loci, contamination) rather than sequencing error.

### Caveats

* The `overall` rate is the most reliable number. Each third-allele read could have come from either
  germline allele, and the per-class rates depend on how the model splits those reads. With fewer than
  a few thousand het sites that split is noisy, especially for the rarer classes, and the per-class
  confidence intervals don't account for it.
* At low depth (~30×), low-VAF artefacts leave only one or two reads and can't be told apart from
  errors, so rates may come out slightly high (about 5% in simulations).

## Docker

```bash
docker build -t third-allele-errors .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data -w /data \
    third-allele-errors sample.bam ref.fa -o errors.tsv
```

## Development

```bash
pip install -e ".[test]"
pytest                    # full suite (~20 s)
pytest -m "not slow"      # skip the large simulations (~10 s)
python scripts/benchmark.py   # compare estimators on simulated data
```

The tests check that the estimator recovers known error rates from simulated data
(`third_allele.simulate`) across WGS, capture and duplex-like regimes, with and without artefacts.
End-to-end tests run the real bcftools pipeline on simulated reads and are skipped if bcftools
isn't on `PATH`.
