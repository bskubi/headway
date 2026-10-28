// Trims adapters and low-quality ends from each sample's reads, before the Snakefile aligns them.
params.reads = "raw/*.fq.gz"

process TRIM {
    tag "$sample"
    publishDir "trimmed", mode: "copy"

    input:
    tuple val(sample), path(reads)

    output:
    path "${sample}.fq.gz"

    script:
    """
    cutadapt -q 20 -m 25 -a AGATCGGAAGAGC -o ${sample}.fq.gz ${reads}
    """
}

workflow {
    Channel.fromPath(params.reads)
        | map { f -> tuple(f.simpleName, f) }
        | TRIM
}
