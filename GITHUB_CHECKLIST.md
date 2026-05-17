# GitHub Readiness Checklist

Before pushing to GitHub, please ensure the following:

- [ ] Run `pytest tests/` (All tests must pass).
- [ ] Run `python scripts/02_train_models.py` (Verify model trains locally).
- [ ] Run `python scripts/03_evaluate_models.py` (Verify plots generate).
- [ ] Open Streamlit app via `streamlit run app/streamlit_app.py` (Check UI).
- [ ] Open FastAPI docs via `http://127.0.0.1:8000/docs`.
- [ ] Review `README.md` for any typos.
- [ ] Verify you have captured and placed screenshots in `reports/screenshots/`.
- [ ] Check that no `.env` or secrets are committed.
- [ ] Verify `.gitignore` is correctly ignoring `venv`, `__pycache__`, and raw data.
- [ ] Ensure `data/sample/sample_pollution_data.csv` is committed for demo purposes.
- [ ] Check large files (e.g., if `.pkl` model is >100MB, add it to `.gitignore` and use Git LFS or just rely on scripts).

## Git Push Commands

```bash
git status
git add .
git commit -m "Finalize IITD Air Intelligence ML project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
*Note: Replace `<YOUR_GITHUB_REPO_URL>` with your actual repository link.*
