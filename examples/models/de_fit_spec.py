"""A draft of glm_em's headway, from `headway init glm_em.steps:de_fit --clean`.

Every step is a call to one of the package's functions or, in clean code, a line of other code between them; the symbols are the variables that pass between
them. Still to do (by hand or with an AI assistant): the TeX of each symbol and the equation of each step,
the descriptions (every TODO), the indices each symbol has (dims=), and what to leave out: steps that
are bookkeeping rather than the method. `headway check` this module as you go.
"""
from headway import Headway

G = Headway('glm_em', "Each gene's log fold changes, from its counts: a negative binomial GLM, its counts normalized by the samples' size factors.",
             draft={"entry": 'glm_em.steps:de_fit', "clean": True}, hints=True)   # headway redraft: draft it again

G.index('samples', 's', 'samples', "from the type hints")
G.index('genes', 'g', 'genes', "from the type hints")
G.index('coefs', 'c', 'coefs', "from the type hints")

# inputs
counts = G.symbol('counts', 'y', 'counts', 'The counts of each gene in each sample.', role='input', dims=('samples', 'genes'))
X = G.symbol('X', 'X', 'design', "The design matrix: each sample's covariates.", role='input', dims=('samples', 'coefs'))
alpha = G.symbol('alpha', '\\alpha', 'dispersions', "Each gene's dispersion: its variance is mu + alpha mu^2.", role='input', dims=('genes',))

# values
s = G.symbol('s', 's', 'size factors', "Each sample's depth relative to the others.", dims=('samples',))

# outputs
irls = G.symbol('irls', '\\beta', 'coefficients', "Each gene's coefficients: its log mean, and its log fold changes.", role='output', per='genes', dims=('coefs',))

G.op('size_factors', 'size factors', tex="", desc="The median of ratios.", inputs=[counts], outputs=[s],
     code=['glm_em.steps:size_factors', 'glm_em.steps:de_fit#s = size_factors(counts)..s'],
     names={'counts': counts, 'return': s})
G.op('irls', 'fit the GLM', tex="", desc="Iteratively reweighted least squares.", inputs=[X, alpha, counts, s], outputs=[irls],
     code=['glm_em.steps:irls', 'glm_em.steps:de_fit#return [irls(counts[:,..return'], per='genes',
     relations=[{'kind': 'minimizes', 'target': 'glm_em.steps:penalized_nll', 'over': ['beta'], 'tol': 1e-06}],
     names={'s': s, 'X': X, 'return': irls})
# drafted by headway init; checksum f7eed97dfe6e2084
