# Hosted dashboard (Vercel)

The live training dashboard at one URL for the whole team, behind a shared password. It shows the same
pages as `sci-jepa serve` plus two hosted-only pages: **Notes** (a shared markdown document) and
**Queue** (the project's SLURM jobs).

Production: https://scijepa-dashboard.vercel.app (Vercel project `scijepa-dashboard`)

## How data gets there

```
training process (any cluster)   ── sci_jepa/dashboard/push.py ──┐
  runs/paper_a_results/dashboard/   (thread in every run,          │
  runs.json, runs/*/live.json, ...   every 30 s + at exit)         ├─> /api/push ──> Upstash Redis
Grete login node                  ── scripts/push_dashboard.py ───┘                    │
  (compute nodes have no internet:  (started by run_paper_a.sh                         v
   the login node pushes instead)    submit, exits with the jobs)  browser <── CDN <── /api/manifest, /api/f
```

**Every run updates the site by itself.** When `SCIJEPA_PUSH_URL` and `SCIJEPA_PUSH_TOKEN` are set in the
environment of a training job, each run (`LiveRun`) starts a background thread that uploads changed dashboard
files every30 s and once more when the  run ends. Several shard processes writing one folder share the work:
only the one holding `<dashboard>/.autopush.lock` uploads. Upload errors are logged and retried with back-off;
they never stop training. `SCIJEPA_AUTOPUSH=0` turns it off, `SCIJEPA_PUSH_EVERY` sets the interval.

On Grete the compute nodes have no internet (`COMPUTE_HAS_INTERNET=0` in `site.env`), so
`bash cluster/grete/run_paper_a.sh submit ...` starts `scripts/push_dashboard.py --until-idle` on the login
node instead (once; it exits when none of your jobs is left; log `${LOGS}/push_dashboard.log`), and the jobs
run with `SCIJEPA_AUTOPUSH=0`.

In the browser, `static/hosted.js` answers the unchanged pages' `fetch("runs.json")` and
`fetch("runs/<name>/live.json")` from the synced copies. Every file version has its own URL, so the CDN
serves repeated polls and Redis is read about once per version. This keeps usage inside the free plans.
Long `history` arrays in `live.json` are thinned to 1,500 rows before upload; local files are never changed.

## Results folders

Every results folder has its own place on the site, so runs with the same name in different folders (S and M model
sizes, 2-epoch runs) no longer overwrite each other. A dashboard folder `<results>/dashboard` is pushed under
`d/<results>/` and shown at `https://scijepa-dashboard.vercel.app/d/<results>/` (for example
`/d/arxiv_full_M_e2/`). The menu next to the site links switches folders. `/` opens the folder viewed last, else
`arxiv_full_S` (`DEFAULT_FOLDERS` in `static/hosted.js`).

- The folder name comes from the results folder. `--folder <name>` or `SCIJEPA_PUSH_FOLDER` sets another, and
  `"."` pushes to the top level. Archives from `import_logs.py` have no `runs.json` and always go to the top level.
- Extra-epoch folders `<base>_e<N>` (e.g. `arxiv_full_M_e2`) hold N-epoch reruns under the same run names. The
  pages of `<base>` and `<base>_e<N>` both show the runs of both folders: `hosted.js` merges their `runs.json`
  and renames the N-epoch runs to `<arm> (N ep)`, so they are their own arm in the Benchmark and All runs tabs.
- `cluster/`, `archive/` and `runs_archive.json` exist once and are shown in every folder.
- `--prune` deletes only inside the folder's own place, never another folder or the top level.
- **The top level** holds everything pushed before 7 Nov 2026, when every folder still pushed there. S and M runs
  with the same names overwrote each other there, so it mixes them. It is kept, under "top level" in the menu. A
  full copy taken before the change is in `/nfs/home/alastia/SCI-JEPA-data/dashboard_backup_20261107/`.
- Jobs started before the change still push to the top level. Push their folders by hand meanwhile:
  `python3 scripts/push_dashboard.py <results>/dashboard --until-idle`.

## Credentials

```bash
# in ~/.bashrc on the cluster (values in deploy/vercel/.env.secrets, ask the project lead);
# sbatch --export=ALL carries them into the jobs
export SCIJEPA_PUSH_URL=https://scijepa-dashboard.vercel.app
export SCIJEPA_PUSH_TOKEN=<PUSH_TOKEN>
```

By hand (for example a results folder copied lsewheere): `python3 scripts/push_dashboard.py /path/to/dashboard`
(every 60 s; `--once` for a single push). The header shows "synced N s ago". It turns orange after 10 minutes
without a push.

## Earlier runs from job logs

Runs that predate the hosted site, or whose dashboard folder is gone, can be rebuilt from their SLURM logs:

```bash
python3 scripts/import_logs.py ../log /tmp/archive                  # archive/<run>/live.json + runs_archive.json
python3 scripts/push_dashboard.py /tmp/archive --once --no-squeue
```

They are listed before the live runs (names start with `log-`) with their loss curves, per-epoch validation and
final test line. They live outside `runs.json`, so new runs never drop them, and `--prune` never deletes them.

## Deploy and settings

```bash
cd SCI-JEPA-training/deploy/vercel
npm install
npm run deploy            # copies the package's dashboard pages into public/, then vercel deploy --prod
```

Redeploy after changing `src/sci_jepa/dashboard/dashboard.html` or `infra.html`, so the hosted copy matches.

Project environment variables (production):

| Variable | Purpose |
|---|---|
| `DASHBOARD_PASSWORD` | the team password; changing it signs everyone out |
| `SESSION_SECRET` | signs the session cookie |
| `PUSH_TOKEN` | the cluster's upload credential (only `/api/push` accepts it) |
| `KV_REST_API_URL`, `KV_REST_API_TOKEN` | set by the Upstash for Redis integration |

Rotate a secret with `vercel env rm NAME production && vercel env add NAME production`, then `npm run deploy`.
