"""The headway of a pipeline written in five languages, none of them Python: code references by file path.

Each step points to its code by path (relative to this folder): "R/normalize.R:size_factors" names a definition,
found with tree-sitter (R, Rust, bash, Snakemake, ... ); "main.nf#process TRIM" anchors on text, for Nextflow, which
tree-sitter lacks; "config.yaml:2" is a line. `headway check spec` checks that they all resolve.
"""
from headway import Headway

G = Headway("signature score", "A score per RNA-seq sample for a gene signature, from raw reads: trimmed (Nextflow), "
            "aligned and counted (Snakemake), filtered (bash), normalized (R) and scored (Rust).")
s = G.index("s", "s", "sample")
g = G.index("g", "g", "gene")
R = G.symbol("R", "R_s", "reads", "Sample $s$'s raw sequencing reads.", dims=(s,), role="input")
Rt = G.symbol("Rt", r"\tilde R_s", "trimmed reads", "Without adapters and low-quality ends.", dims=(s,))
K = G.symbol("K", "K_{sg}", "counts", "Reads of sample $s$ aligned to gene $g$.", dims=(s, g))
m = G.constant("min_reads", "m", "minimum reads", "Samples with fewer counted reads are dropped.",
               value=1000000, code="config.yaml:2")
keep = G.symbol("keep", "k_s", "kept", "1 if sample $s$ has enough counted reads.", dims=(s,))
f = G.symbol("f", "f_s", "size factor", "Sample $s$'s depth relative to the typical sample.", dims=(s,))
c = G.constant("pseudocount", "c", "pseudocount", value=1, code="R/normalize.R#PSEUDOCOUNT <-")
L = G.symbol("L", "L_{sg}", "log expression", dims=(s, g))
w = G.symbol("w", "w_g", "signature weight", "Zero for genes outside the signature.", dims=(g,), role="input",
             code="rust/src/lib.rs:Scorer")
S = G.symbol("S", "S_s", "signature score", dims=(s,), role="output")

G.group("reads", "reads to counts")
G.group("norm", "normalization")
G.op("trim", "trim", r"\tilde R_s = \operatorname{cutadapt}(R_s)", "Quality 20, minimum length 25.",
     inputs=[R], outputs=[Rt], code="main.nf#process TRIM", group="reads")
G.op("count", "align and count", r"K_{sg} = \#\{\text{reads of } \tilde R_s \text{ aligned to } g\}",
     inputs=[Rt], outputs=[K], code=["Snakefile:align", "Snakefile:count"], group="reads")
G.op("qc", "filter samples", r"k_s = [\textstyle\sum_g K_{sg} \ge m]", inputs=[K, m], outputs=[keep],
     code="scripts/qc.sh:filter_samples", group="reads")
G.op("sizefactor", "size factors",
     r"f_s = \operatorname{median}_g \frac{K_{sg}}{\left(\prod_{s'} K_{s'g}\right)^{1/n}}",
     "Median of ratios to the geometric mean, over the genes counted in every sample.",
     inputs=[K, keep], outputs=[f], code="R/normalize.R:size_factors", group="norm",
     where={"n": "the number of kept samples"})
G.op("log", "log normalize", r"L_{sg} = \log_2\left(\frac{K_{sg}}{f_s} + c\right)", inputs=[K, f, c], outputs=[L],
     code="R/normalize.R:log_normalize", group="norm")
G.op("score", "signature score", r"S_s = \frac{\sum_g w_g L_{sg}}{\sum_g |w_g|}", inputs=[L, w], outputs=[S],
     code="rust/src/lib.rs:Scorer.score")
