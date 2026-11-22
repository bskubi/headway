---
name: plain-text
description: How to write the text headway shows to people, in plain English at a measured reading level. Use whenever you write or change text a user reads - the viewer's labels, tooltips, panels and tour (viewer.html), command-line help, errors and check reports, GUIDE.md and the README - and to audit or rewrite that text.
---

# Plain text for headway

## The reader

A scientist or engineer who knows their own method well and is new to headway. They may read English as a second
language. They read a tooltip in a glance and the guide when they are stuck. Technical words of their field
(equation, metric, dataset, replicate) are fine. Words that are headway's own are not, until the glossary below
defines them.

## Targets (checked by `readability.py`)

Run `python .claude/skills/plain-text/readability.py` before and after a change; `-v` lists every sentence over a
limit, with its part named (`viewer`, `cli`, `guide`).

| | Grade (Flesch-Kincaid) | Words in a sentence | Colons and semicolons in a sentence |
|---|---|---|---|
| Tooltips, panel text, messages | 7 or lower | at most 25, aim for under 15 | at most 1 |
| The guide | 8 to 9 | at most 35, average under 20 | at most 1 |

Grade formulas count only word and sentence length, and headway's words are short. Most of what makes its text
hard is structure, so the rules below matter more than the grade.

## Rules

1. **Whole sentences.** A subject and a verb. Write "Click to read this version", not "click: read this version".
   A label (a button, a column) may be a noun phrase; a tooltip or a message is a sentence.
2. **One idea per sentence.** Do not chain clauses with colons, semicolons and middle dots. Split them.
3. **Action first in a tooltip.** Say what happens: "Hides this version from the graph and the results."
   Add one more sentence for why or when, if needed. Move anything longer to the guide.
4. **At most one aside in brackets per sentence.** If the aside matters, make it a sentence.
5. **No possessive chains.** Write "the weights in the suite's objective", not "the suite's objective's weights".
   One possessive is fine.
6. **Say who acts.** Prefer the active voice: "headway marks the changes", not "the changes are marked".
7. **Name things the same way every time**, with the glossary's word. Do not call a lane a "column" in one place
   and a "version" in another.
8. **Plain words.** use, not utilize. show, not display. about, not regarding. Keep the exact technical word when
   a plain one would change the meaning.
9. **Keep every fact.** A rewrite must say all that the old text said, and nothing false. If a fact does not fit
   a tooltip, move it to the guide and say where.
10. **Symbols need words.** Explain ●, ◐, ○, Δ and ± the first time they appear in a place, or point to the legend.

## Not held to these rules

- **Prompts the viewer writes for an AI agent** (the Ask AI presets and their context, in `viewer.html`). They are
  read by a model and need precision more than short sentences. Their buttons' tooltips are for people, and follow
  the rules. `readability.py` leaves the prompts out.
- **Key strips**: a list of keys and actions separated by `·`, such as "Click to select · drag to pan". Each item
  is a short phrase. `readability.py` counts each item as a sentence.

## Do not change

- Identifiers, command names, flags, keys, file names, code and TeX.
- Text that the tests match. Run the tests after a rewrite.
- The command line's report text without the user's agreement: `examples/build.sh` writes `changes.txt` and
  `results.txt`, which are tracked and must stay unchanged unless the user agrees to a new wording.

## Glossary

Use these words, with these meanings. Define a term the first time the guide or the tour uses it.

- **spec**: the Python module you write that describes a method: its values, steps, equations and code references.
- **headway**: what `headway build` makes from a spec: a JSON file or an HTML page that draws the method.
- **step**: one operation of the method, with its equation and a reference to the code that does it.
- **value**: something a step uses or makes. It is a data input, a setting, a constant, an intermediate value (made by
  one step and used by later ones) or an output (a result the method gives back).
- **symbol**: the math name of a value, such as X with indices i and g.
- **group**: a set of steps, usually one function or module of the code. Groups can be collapsed.
- **version**: one built headway of a method, for example at one commit.
- **lane**: one open version in the viewer. Lanes are drawn side by side.
- **baseline**: the version that every other open version is compared with.
- **suite**: a benchmark's declaration: its metrics, datasets, figures and objectives.
- **metric**: a number that measures how well a version does on a dataset.
- **objective**: a rule that combines several metrics into one score or ranking.
- **voter**: one metric, or one category, as an objective sees it: it orders the versions, and the objective's rule
  combines the voters' orders.
- **category**: a named set of metrics, for example "biological" or "scalability". An objective can treat each
  category as one vote.
- **replicate**: one run of a benchmark on a dataset. Several replicates give a spread.
- **atlas**: a folder that holds many versions of a method, their results and their relations.
- **relation**: a link between versions in an atlas, for example "made from" or "alternatives".
- **shelf**: in the atlas view, the versions you have collected to open together.
