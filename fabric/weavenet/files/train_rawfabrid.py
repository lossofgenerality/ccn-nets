"""WeaveNet on RAW-FABRID: trained on clean fabric only, and measured against the dataset's masks.

    RAW_FABRID=<folder holding RAW_FABRID/MVTec> python3 projects/fabric/nets/weavenet/train/train_rawfabrid.py [run-name]

RAW-FABRID (Pérez-Llorens, Albero-Albero and Silvestre-Blanes, Mendeley Data, DOI 10.17632/db6g85xsyg.1,
CC BY 4.0): 256 x 256 greyscale patches of raw fabric from seven rolls, in the MVTec layout. This run uses
a fixed subset: every 4th clean training patch, every 5th clean test patch, and all 687 defective test
patches with their masks.

What changes from train.py, and why:
  1. Only clean fabric is used for training, so no robust setting-aside is needed.
  2. Some clean patches are kept apart for calibration: the threshold for "defect" is set on them, as
     the level that clean fabric almost never reaches. A threshold set by the image itself (as in the
     first run) always finds something, so it cannot be used to measure false alarms.
  3. Every number is measured: the share of defective patches found, the share of clean patches with a
     false alarm, the area under the ROC curve for whole patches and for pixels, and the overlap of the
     boxes with the masks.

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Laboratory

© Computational Classnotes · CC BY-NC-SA 4.0
"""
import glob, json, os, sys, time

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as T                      # noqa: E402  (patches, redraw, panels, filters, paths)
from ccnlab import nn, export          # noqa: E402

SETTINGS = {"data": "RAW-FABRID (CC BY 4.0), fixed subset", "patch": 16, "stride (training)": 8,
            "training patches per image": 60, "stride (scoring)": 4, "hidden": 16, "activation": "Tanh",
            "output": "linear, half mean squared error", "epochs": 10, "batch_size": 512, "lr": 0.003,
            "l2": 1e-5, "seed": 7, "calibration images (clean, not trained on)": 400,
            "smoothing of the error map (pixels)": 2.0, "start level (clean images reaching it)": 0.05,
            "grow level (fraction of the start level)": 0.5, "joining gap (pixels)": 4,
            "smallest defect (pixels)": 40, "border left out (pixels)": 6}


def load(folder, n=None):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))[:n]
    return [os.path.basename(f) for f in files], [np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0 for f in files]


def error_map(net, img, s):
    rec = T.redraw(net, img, s["patch"], s["stride (scoring)"])
    err = ndimage.gaussian_filter((img - rec) ** 2, s["smoothing of the error map (pixels)"])
    b = s["border left out (pixels)"]
    # Each fabric is redrawn more or less easily, so the error is measured against the image's own
    # typical error (its median): a defect stands out from its own fabric, whatever the fabric.
    err = err / (np.median(err[b:-b, b:-b]) + 1e-12)
    err[:b, :] = err[-b:, :] = err[:, :b] = err[:, -b:] = 0.0
    return rec, err


def boxes_of(err, start, s):
    grow = ndimage.binary_dilation(err >= start * s["grow level (fraction of the start level)"], iterations=s["joining gap (pixels)"])
    lab, _ = ndimage.label(grow)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        part = lab[sl] == k
        if sl is not None and part.sum() >= s["smallest defect (pixels)"] and err[sl][part].max() >= start:
            out.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop, float(err[sl][part].max() / start)))
    return out


def auroc(pos, neg):
    """The chance that a random defective score is above a random clean one (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(name="rawfabrid-seed7"):
    s = SETTINGS
    root = os.path.join(os.environ["RAW_FABRID"], "RAW_FABRID", "MVTec")
    scratch = os.environ.get("CCN_SCRATCH")
    runs_dir = os.path.join(scratch, "runs") if scratch else os.path.join(T.NET, "runs")
    viz = os.path.join(scratch, "viz") if scratch else os.path.join(T.NET, "viz")
    os.makedirs(viz, exist_ok=True)
    rng = np.random.default_rng(s["seed"])
    _, good = load(os.path.join(root, "train", "good"))
    order = rng.permutation(len(good))
    cal = [T.standardise(good[i]) for i in order[:s["calibration images (clean, not trained on)"]]]
    trn = [T.standardise(good[i]) for i in order[s["calibration images (clean, not trained on)"]:]]
    p = s["patch"]
    X = []
    for im in trn:
        P, _ = T.patches(im, p, s["stride (training)"])
        X.append(P[rng.choice(len(P), s["training patches per image"], replace=False)])
    X = np.concatenate(X)
    net = nn.Net([p * p, s["hidden"], p * p], [s["activation"]], output="linear").initialise(s["seed"])
    t0 = time.perf_counter()
    history = nn.train(net, X, X, epochs=s["epochs"], batch_size=s["batch_size"], lr=s["lr"], l2=s["l2"], seed=s["seed"])
    t_train = time.perf_counter() - t0
    # Calibration on clean fabric the net has not seen: the start level is the highest error that only
    # 1% of clean images reach anywhere.
    cal_max = [error_map(net, im, s)[1].max() for im in cal]
    start = float(np.quantile(cal_max, 1 - s["start level (clean images reaching it)"]))
    # Test: clean and defective patches, with masks
    gn, gimgs = load(os.path.join(root, "test", "good"))
    dn, dimgs = load(os.path.join(root, "test", "defect"))
    masks = [np.asarray(Image.open(os.path.join(root, "ground_truth", "defect", n)).convert("L")) > 127 for n in dn]
    t0 = time.perf_counter()
    g_scores, g_alarm, d_scores, d_found, d_iou, pix_pos, pix_neg = [], 0, [], 0, [], [], []
    shown = []
    for im in gimgs:
        _, err = error_map(net, T.standardise(im), s)
        g_scores.append(err.max()); g_alarm += bool(boxes_of(err, start, s))
        pix_neg.append(err[::8, ::8].ravel())
    for k, (im, m) in enumerate(zip(dimgs, masks)):
        sim = T.standardise(im)
        rec, err = error_map(net, sim, s)
        bx = boxes_of(err, start, s)
        d_scores.append(err.max())
        hit = np.zeros_like(m)
        for (y0, x0, y1, x1, _) in bx:
            hit[y0:y1, x0:x1] = True
        found = bool((hit & m).any())
        d_found += found
        if m.any():
            d_iou.append(float((hit & m).sum() / max(1, (hit | m).sum())))
        pix_pos.append(err[m]); pix_neg.append(err[~m][::8])
        if found and len(shown) < 6 and m.sum() > 150 and k % 7 == 0:
            shown.append(k)
            T.panels(dn[k][:-4], sim, rec, err, err / start, bx, os.path.join(viz, f"raw-panels-{len(shown)}.png"), T.SETTINGS,
                     score_title="Error ÷ the level clean fabric rarely reaches", mask=m)
    t_score = time.perf_counter() - t0
    pp, pn = np.concatenate(pix_pos), np.concatenate(pix_neg)
    sub = np.random.default_rng(1)
    pp = pp[sub.choice(len(pp), min(len(pp), 60000), replace=False)]
    pn = pn[sub.choice(len(pn), min(len(pn), 60000), replace=False)]
    metrics = {
        "Training patches (16 × 16, clean only)": int(len(X)), "Clean images for calibration": len(cal),
        "Loss, first and last epoch": [round(history[0], 4), round(history[-1], 4)],
        "Gradient check (linear output), largest difference": nn.gradient_check(activations=("Tanh",), classes=5, output="linear"),
        "Defective test patches found": f"{d_found} of {len(dimgs)}",
        "Defective test patches found, share": round(d_found / len(dimgs), 3),
        "Clean test patches with a false alarm": f"{g_alarm} of {len(gimgs)}",
        "Clean test patches with a false alarm, share": round(g_alarm / len(gimgs), 3),
        "Patch-level AUROC (largest error)": round(auroc(d_scores, g_scores), 3),
        "Pixel-level AUROC": round(auroc(pp, pn), 3),
        "Box overlap with the masks (mean IoU, defective patches)": round(float(np.mean(d_iou)), 3),
        "Numbers in the net": net.weight_count}
    path = export.save_run(net, runs_dir, name, settings=s, metrics=metrics, history=history,
                           timings={"training, s": round(t_train, 2), "scoring all test patches, s": round(t_score, 1)})
    T.filters(net, os.path.join(viz, "raw-filters.png"), p)
    if not scratch:
        card_p = os.path.join(T.NET, "card.json")
        card = json.load(open(card_p, encoding="utf-8"))
        card["latest_run"] = os.path.basename(path)
        export.write_json(card_p, card)
    print(os.path.basename(path), json.dumps({k: v for k, v in metrics.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
