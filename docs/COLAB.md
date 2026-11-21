# Running on Google Colab

## One-time setup

1. **Push this repo to GitHub** as a private repository called `pianist-id`.
2. **Create a GitHub token** that Colab can use to clone the private repo:
   GitHub → Settings → Developer settings → Personal access tokens → *Fine-grained
   tokens* → Generate. Repository access: *Only select repositories* → `pianist-id`.
   Permissions: *Contents: Read-only*. Copy the token.
3. **Add two Colab secrets** (key icon in the left sidebar of any Colab notebook):
   - `GITHUB_USER`: your GitHub user name
   - `GITHUB_TOKEN`: the token from step 2

   Switch on *Notebook access* for both. The setup cell reads them with
   `google.colab.userdata`, so the token never appears in the notebook.
4. **Drive layout.** Create `MyDrive/pianist-id/raw/` and upload the ATEPP v1.2 zip
   there (download it yourself from `github.com/tangjjbetsy/ATEPP` after accepting the
   disclaimer). Everything else under `MyDrive/pianist-id/` is created by the notebooks:

   ```
   MyDrive/pianist-id/
   ├── raw/        ATEPP zip (by hand)
   ├── cache/      parsed note/pedal arrays (notebook 02)
   ├── splits/     split CSVs (notebook 02)
   ├── features/   feature tables (notebook 03)
   └── results/    one folder per experiment/run, incl. checkpoints
   ```

## Opening a notebook

In Colab: *File → Open notebook → GitHub*, tick *Include private repos* (authorise
Colab on GitHub once), choose `pianist-id` and the notebook. Alternatively upload the
`.ipynb` from your computer. The setup cell always clones/pulls the latest code from
GitHub, so the library code is up to date either way.

## Every session

Run the setup cell first (or *Runtime → Run all*). It:

1. mounts Drive,
2. clones the repo into `/content/pianist-id`, or pulls if it is already there,
3. installs `requirements.txt` and the package (`pip install -e . --no-deps`),
4. sets `PIANIST_ID_ROOT=/content/drive/MyDrive/pianist-id` and
   `PIANIST_ID_LOCAL=/content/local`,
5. copies the cache from Drive to local disk if it exists and isn't there yet,
6. prints Python/package versions, the GPU and the git commit.

If you changed code on GitHub during a session, re-run the setup cell and then
*Runtime → Restart session* so Python re-imports the package.

## Runtimes

| Notebook | Runtime | Notes |
|---|---|---|
| 00 data audit | CPU | ~2–5 min |
| (later notebooks) | see each notebook's header | |
