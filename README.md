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

## Implementation

### Inputs

### Outputs

### Docker

## Examples


