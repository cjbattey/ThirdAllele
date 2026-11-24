# Image for the third-allele-errors CLI (bcftools + polars + scipy + defopt).
#
# Build: docker build -t third-allele-errors .
# Run:   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data -w /data \
#            third-allele-errors sample.bam ref.fa -o errors.tsv
# (--user lets the container write outputs into the mounted directory.)
#
# BAM/CRAM and FASTA must be indexed (.bai/.crai, .fai) next to the inputs.
FROM mambaorg/micromamba:1.5.10-bookworm-slim

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Dependencies come from conda above; pip installs only the package itself.
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE /tmp/third-allele/
COPY --chown=$MAMBA_USER:$MAMBA_USER src /tmp/third-allele/src
RUN pip install --no-deps --no-build-isolation --no-cache-dir /tmp/third-allele \
    && rm -rf /tmp/third-allele

# Smoke test: imports resolve and bcftools runs.
RUN bcftools --version | head -1 && third-allele-errors --help > /dev/null

# The base image's entrypoint activates the env before running CMD/args.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "third-allele-errors"]
CMD ["--help"]
