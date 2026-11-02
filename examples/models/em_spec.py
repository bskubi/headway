"""A draft of glm_em's headway, from `headway init glm_em.steps:em --clean`.

Every step is a call to one of the package's functions or, in clean code, a line of other code between them; the symbols are the variables that pass between
them. Still to do (by hand or with an AI assistant): the TeX of each symbol and the equation of each step,
the descriptions (every TODO), the indices each symbol has (dims=), and what to leave out: steps that
are bookkeeping rather than the method. `headway check` this module as you go.
"""
from headway import Headway

G = Headway('glm_em', "TODO: one sentence on what the method computes, from what.",
             draft={"entry": 'glm_em.steps:em', "clean": True}, hints=True)   # headway redraft: draft it again

G.index('reads', 'r', 'reads', "TODO (an axis named in the type hints of glm_em.steps:em)")
G.index('components', 'c', 'components', "TODO (an axis named in the type hints of glm_em.steps:em)")

# inputs
m = G.symbol('m', 'm', 'm', 'TODO', role='input', dims=('reads',))
n = G.symbol('n', 'n', 'n', 'TODO', role='input', dims=('reads',))
p = G.symbol('p', 'p', 'p', 'TODO', role='input', dims=('components',))
steps = G.symbol('steps', 's', 'steps', 'TODO', role='input')

# values
weights = G.symbol('weights', 'w', 'weights', 'TODO')

# outputs
weights_2 = G.symbol('weights_2', "w'", 'weights', 'TODO', role='output')
p_2 = G.symbol('p_2', "p'", 'p', 'TODO', role='output')

G.op('weights', 'weights', tex="", desc="TODO", inputs=[p], outputs=[weights],
     code='glm_em.steps:em#weights = np.full(len(p),..weights',
     names={'p': p, 'weights': weights, 'return': weights})
G.op('weights_p', 'weights, p', tex="", desc="TODO", inputs=[m, n, p, steps, weights], outputs=[weights_2, p_2],
     code='glm_em.steps:em#for step..p =', names={'steps': steps, 'm': m, 'n': n})
# drafted by headway init; checksum 80424ab96c2e1465
