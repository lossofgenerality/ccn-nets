# The standard layout of a net

Every net, in whichever project's repository it lives, has one folder in this form (for example `nets/subjectnet/` in `ccn-agent-free-form-language`):

| Path | Holds |
|---|---|
| `net.json` | The structure, as saved from the net-diagram editor: the single source the code and pictures come from |
| `card.json` | The summary the published pages are built from (below) |
| `data/` | The data generator and its seeds, or where they are; the fixed test sets |
| `train/` | The training: a Python script using `ccnlab`, and the Wolfram notebook (or where it is) |
| `runs/<date>-<name>/` | One folder per training run, never overwritten: `weights.npz`, `run.json` (settings, timings, toolkit, Python, NumPy, machine), `metrics.json`, `history.json` |
| `analysis/` | Notebooks and reports that look inside the net |
| `viz/` | The visualisations made from the latest run: the 3D view's data, the matrix drawing |
| `writeup.md` | Aim, data, design, results and limits |

## card.json

| Key | Meaning |
|---|---|
| `name`, `slug` | The net's name, and the folder name on the site |
| `project` | The project's key in `publish.json` |
| `status` | For example `drawn`, `trained`, `published` |
| `purpose`, `structure` | One sentence each |
| `numbers` | How many numbers (weights and biases) the net has |
| `editor_example` | The editor's example key, if the editor has the net |
| `viewer` | The 3D viewer to use, a file in `viewer3d/` (for example `dense2`) |
| `latest_run` | The run folder the pages are built from |
| `files` | `diagram`, `writeup`, `viewer_data`, `matrices`, and `downloads` (paths from the repository root) |
| `headline` | The results shown as large figures on the net's page |

Only what a card names is copied to the published site.

---
© Computational Classnotes · CC BY-NC-SA 4.0
