# Working on QuantStream

Use small, truthful changes that a reviewer can reproduce. An unsuccessful experiment is a valid result when its hypothesis, configuration and evaluation are recorded accurately.

## Branch and commit flow

```bash
git switch main
git pull --ff-only
git switch -c feat/kaggle-tcn-run-001
# Make the actual change or run the planned experiment.
python -m pytest -q
python -m ruff check quantstream app.py tests scripts
python scripts/check_notebooks.py
git diff
git add <specific-files>
git diff --staged
git commit -m "feat: record first Kaggle TCN experiment"
git push -u origin feat/kaggle-tcn-run-001
```

Open a PR describing the problem/hypothesis, resulting behavior, checks, observed results and limits. Suggested prefixes: `feat`, `fix`, `test`, `docs`, `ci`, `refactor`, `chore`.

Before training, commit code/config and export from a clean tree. Record dataset hash, Git SHA, seed, device, time windows and costs using `docs/experiments/TEMPLATE.md`. Choose models using development data; a revised hypothesis needs a new independent final period.

Keep datasets, credentials, weights, optimizer states and paper state outside Git. Commit small deliberate metrics/figures under `reports/`. Large artifacts belong in Kaggle outputs or a deliberate release.

CI runs offline logic tests/lint/notebook validation on Linux and Windows; it does not redownload financial data or train full models. A commit is local until pushed. Work outside this repository does not automatically enter its history.
