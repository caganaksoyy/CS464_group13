# Running on Google Colab

## One-time setup

1. **GitHub access.** The repo is `caganaksoyy/CS464_group13`. The setup cell clones it
   in one of two ways:
   - **With a token** (needed while the repo is private): create a fine-grained token
     under GitHub → Settings → Developer settings → Personal access tokens →
     *Fine-grained tokens* → Generate. Repository access: *Only select repositories* →
     `CS464_group13`; permissions: *Contents: Read-only*. Group members use their own
     token; they must be collaborators on the repo.
     Then add the Colab secrets (key icon in the left sidebar) and switch on *Notebook
     access* for them:
     - `GITHUB_TOKEN`: the token
     - `GITHUB_USER`: your GitHub user name (optional)

     The setup cell reads them with `google.colab.userdata`, so the token never appears
     in the notebook.
   - **Without a token**: if `GITHUB_TOKEN` is missing (or notebook access to it is
     off), the cell clones `https://github.com/caganaksoyy/CS464_group13.git` with no
     credentials. This only works when the repo is public; for a private repo the clone
     fails with an authentication error.
2. **Data.** Nothing to do by hand. If `MyDrive/pianist-id/raw/` holds no `.zip`, the
   first notebook downloads ATEPP v1.2 (`ATEPP-1.2.zip`, ~220 MB) with `gdown` from the
   public Google Drive link of the ATEPP repository (id `1Df2KUdqvXtgvhvzx2D10YaQRbpP23rOS`)
   and stores it there, so later sessions reuse it. To use your own copy instead, upload
   the zip to `MyDrive/pianist-id/raw/` before the first run. ATEPP is licensed CC BY 4.0
   (citation in the README).

   Everything under `MyDrive/pianist-id/` is created by the notebooks:

   ```
   MyDrive/pianist-id/
   ├── raw/        ATEPP zip (downloaded automatically if missing)
   ├── cache/      parsed note/pedal arrays (later milestones)
   ├── splits/     split CSVs (notebook 01)
   ├── features/   feature tables (notebook 01)
   └── results/    one folder per experiment/run (audit, xgb_baseline, ...)
   ```

## Opening a notebook

- **Badge:** the *Open in Colab* badge in the README opens notebook 01 directly. For
  the private repo, Colab asks once to authorise access to your GitHub account.
- **By hand:** *File → Open notebook → GitHub*, tick *Include private repos*, choose
  `caganaksoyy/CS464_group13` and the notebook. Alternatively upload the `.ipynb` from
  your computer.

The setup cell always clones/pulls the latest code from GitHub, so the library code is
up to date however the notebook was opened.

## Every session

Run the setup cell first (or *Runtime → Run all*). It:

1. mounts Drive,
2. clones the repo into `/content/CS464_group13` (with the token if `GITHUB_TOKEN` is
   available, otherwise the public URL), or pulls if it is already there,
3. installs `requirements.txt` and the package (`pip install -e . --no-deps`),
4. sets `PIANIST_ID_ROOT=/content/drive/MyDrive/pianist-id` and
   `PIANIST_ID_LOCAL=/content/local`,
5. copies the cache from Drive to local disk if it exists and isn't there yet,
6. prints Python/package versions, the GPU and the git commit.

The first data cell then finds the ATEPP zip in `raw/` (downloading it if missing) and
extracts it to `/content/local/atepp` (~1 min per session).

If you changed code on GitHub during a session, re-run the setup cell and then
*Runtime → Restart session* so Python re-imports the package.

## Runtimes

| Notebook | Runtime | Notes |
|---|---|---|
| 00 data audit | CPU | ~2–5 min |
| 01 XGBoost baseline | CPU | features ~5 min (first run, then cached); 24 fits ~10–15 min (N=10 part optional via `RUN_N10`) |
| (later notebooks) | see each notebook's header | |
