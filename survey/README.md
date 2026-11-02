# Survey of codebases

A running record of existing codebases read to learn what headway needs to handle them well.

- `traits.yaml` is the vocabulary: the properties of a codebase that matter for writing its method as checked math.
  Each trait has a category, a description, headway's status (`handled`, `partial`, `missing`, `n/a`) and what
  handling it would take.
- `codebases/<id>.yaml` is one record per codebase. It holds the repository and the commit read, the methods it
  implements and their papers, where its computation lives, how it installs and tests, the method's steps, its
  traits with evidence, headway's results on it, and findings.
- `survey.py` checks the records against the vocabulary, tabulates them, and renames traits:
  ```
  uv run survey.py check
  uv run survey.py table
  uv run survey.py trait optimizer-closures
  uv run survey.py rename OLD NEW
  ```

The clones and their environments are in `~/targets/<id>` (each with its own `.venv`). Install and test logs are in
`~/targets/survey_logs`.

When a trait is reframed (split, merged, renamed), change `traits.yaml` first, then migrate the records and run
`check`. When a new property turns up, add it to `traits.yaml` before using it in a record.
