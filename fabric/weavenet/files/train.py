"""WeaveNet: a patch autoencoder that learns the weave of normal fabric, so that what it cannot
redraw is the defect. Trained with our own code (ccnlab.nn), no neural-net library.

    python3 projects/fabric/nets/weavenet/train/train.py [run-name]      (from the ccn-net-lab root)

The steps, each written out below:
  1. Every image is put on one scale (mean 0, spread 1), so lighting does not matter.
  2. Training patches of 16 x 16 pixels are cut from the training images, every 4 pixels.
  3. The net squeezes each patch (256 numbers) through 48 hidden units and redraws it.
  4. Robust training: after the first passes, the 5% of patches it redraws worst (most of them on
     defects) are set aside, and training goes on with the rest; so it learns the weave, not the flaws.
  5. Scoring an image: every patch is redrawn, the redrawn patches are put back together (overlaps
     averaged), and the squared difference from the image is the error map.
  6. The error map becomes a robust z-score (median and median absolute deviation of the image), and
     areas with z above the threshold are boxed as defects, as in the classical trials.

By Dara O Shayda and Hikari Noor Hoshino (星野ヌール)
AI System Programming Group

© Computational Classnotes · CC BY-NC-SA 4.0
"""
import glob, json, os, sys, time

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.dirname(HERE)
PROJECT = os.path.dirname(os.path.dirname(NET))
LAB = os.path.dirname(os.path.dirname(PROJECT))
sys.path.insert(0, LAB)
from ccnlab import nn, export          # noqa: E402

SETTINGS = {"patch": 16, "stride (training)": 4, "stride (scoring)": 4, "hidden": 16, "activation": "Tanh",
            "output": "linear, half mean squared error", "epochs": 14, "robust after epoch": 6,
            "set aside (worst redrawn)": 0.05, "batch_size": 256, "lr": 0.003, "l2": 1e-5, "seed": 7,
            "smoothing of the error map (pixels)": 2.0, "z to start a defect": 9.0, "z to grow it": 4.0,
            "joining gap (pixels)": 4, "smallest defect (pixels)": 60, "border left out (pixels)": 6,
            "test images": 6}


def load_images():
    files = sorted(glob.glob(os.path.join(PROJECT, "data", "dataset-2", "*.png")))
    imgs = [np.asarray(Image.open(f).convert("L"), dtype=float) / 255.0 for f in files]
    return [os.path.basename(f)[:-4] for f in files], imgs


def standardise(img):
    return (img - img.mean()) / (img.std() + 1e-8)


def patches(img, p, stride):
    """All p x p patches every `stride` pixels, as rows of p*p numbers, with their top-left corners."""
    h, w = img.shape
    ys, xs = range(0, h - p + 1, stride), range(0, w - p + 1, stride)
    return np.array([img[y:y + p, x:x + p].ravel() for y in ys for x in xs]), [(y, x) for y in ys for x in xs]


def redraw(net, img, p, stride):
    """The image as the net redraws it: every patch redrawn, overlaps averaged."""
    P, corners = patches(img, p, stride)
    R = net.probabilities(P)                 # with a linear output this is the redrawn patch
    out, count = np.zeros_like(img), np.zeros_like(img)
    for r, (y, x) in zip(R, corners):
        out[y:y + p, x:x + p] += r.reshape(p, p)
        count[y:y + p, x:x + p] += 1
    return out / np.maximum(count, 1)


def anomaly(img, rec, s):
    err = ndimage.gaussian_filter((img - rec) ** 2, s["smoothing of the error map (pixels)"])
    med = np.median(err)
    mad = np.median(np.abs(err - med)) * 1.4826 + 1e-12
    z = (err - med) / mad
    b = s["border left out (pixels)"]          # the edges are covered by fewer patches, so their error is less sure
    z[:b, :] = z[-b:, :] = z[:, :b] = z[:, -b:] = 0.0
    # Hysteresis, as in edge detection: a defect starts where z is very high, and grows over the
    # pixels around it where z is still clearly high; nearby pieces are joined across small gaps.
    grow = ndimage.binary_dilation(z > s["z to grow it"], iterations=s["joining gap (pixels)"])
    lab, n = ndimage.label(grow)
    boxes = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        part = lab[sl] == k
        if sl is not None and part.sum() >= s["smallest defect (pixels)"] and z[sl][part].max() >= s["z to start a defect"]:
            boxes.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop, float(z[sl][part].max())))
    return err, z, boxes


def split(names):
    """Test images: one from every few, so every roll is in both sets."""
    test = set(range(1, len(names), max(1, len(names) // SETTINGS["test images"])))
    test = sorted(test)[:SETTINGS["test images"]]
    return [i for i in range(len(names)) if i not in test], test


def panels(name, img, rec, err, z, boxes, path, s, score_title="Anomaly score (robust z)", mask=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 5, figsize=(19.2, 4.3), dpi=100)
    show = [(img, "gray", "Input fabric"), (rec, "gray", "Redrawn by WeaveNet (the weave it learnt)"),
            (err, "viridis", "Squared difference (smoothed)"), (z, "inferno", score_title), (img, "gray", f"Detected defects: {len(boxes)}")]
    for a, (m, cm, t) in zip(ax, show):
        if cm == "inferno":
            im = a.imshow(np.clip(z, 0, None), cmap=cm)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        else:
            a.imshow(m, cmap=cm)
        a.set_title(t, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    if mask is not None:                      # the dataset's own outline of the defect, for comparison
        ax[4].contour(mask, levels=[0.5], colors="#ff3b6b", linewidths=1.2)
        ax[4].set_title(f"Detected defects: {len(boxes)} (green); true outline (pink)", fontsize=11)
    for (y0, x0, y1, x1, _) in boxes:
        ax[4].add_patch(Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, fill=False, ec="#00e05a", lw=2))
    fig.text(0.005, 0.02, name, fontsize=8, color="#888")
    fig.text(0.995, 0.02, "© Computational Classnotes · CC BY-NC-SA 4.0", ha="right", fontsize=8, color="#888")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def filters(net, path, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = net.params[0]["W"]                    # (256, hidden): one column per hidden unit
    n = W.shape[1]; cols = 8; rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.75 + 0.7), dpi=100)
    for k, a in enumerate(ax.ravel()):
        a.set_xticks([]); a.set_yticks([])
        if k < n:
            w = W[:, k].reshape(p, p); m = np.abs(w).max()
            a.imshow(w, cmap="RdBu_r", vmin=-m, vmax=m)
            a.set_title(str(k + 1), fontsize=8)
        else:
            a.axis("off")
    fig.suptitle(f"What the {n} hidden units look for: the first weights of each, as a 16 × 16 picture (red positive, blue negative)", fontsize=11)
    fig.text(0.995, 0.005, "© Computational Classnotes · CC BY-NC-SA 4.0", ha="right", fontsize=8, color="#888")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(name="seed7"):
    """CCN_SCRATCH=<folder> sends the run and pictures there instead, for trials that are not kept."""
    s = SETTINGS
    scratch = os.environ.get("CCN_SCRATCH")
    runs_dir = os.path.join(scratch, "runs") if scratch else os.path.join(NET, "runs")
    viz = os.path.join(scratch, "viz") if scratch else os.path.join(NET, "viz")
    os.makedirs(viz, exist_ok=True)
    names, raw = load_images()
    imgs = [standardise(im) for im in raw]
    tr, te = split(names)
    p = s["patch"]
    X = np.concatenate([patches(imgs[i], p, s["stride (training)"])[0] for i in tr])
    net = nn.Net([p * p, s["hidden"], p * p], [s["activation"]], output="linear").initialise(s["seed"])
    t0 = time.perf_counter()
    h1 = nn.train(net, X, X, epochs=s["robust after epoch"], batch_size=s["batch_size"], lr=s["lr"], l2=s["l2"], seed=s["seed"])
    e = np.sum((net.probabilities(X) - X) ** 2, axis=1)
    keep = e <= np.quantile(e, 1 - s["set aside (worst redrawn)"])
    h2 = nn.train(net, X[keep], X[keep], epochs=s["epochs"] - s["robust after epoch"], batch_size=s["batch_size"], lr=s["lr"], l2=s["l2"], seed=s["seed"] + 1)
    t_train = time.perf_counter() - t0
    results, test_err = [], []
    for i, (n_, im) in enumerate(zip(names, imgs)):
        rec = redraw(net, im, p, s["stride (scoring)"])
        err, z, boxes = anomaly(im, rec, s)
        results.append({"image": n_, "set": "test" if i in te else "train", "boxes": len(boxes),
                        "largest z": round(max([b[4] for b in boxes], default=float(z.max())), 1),
                        "mean squared difference": round(float(np.mean((im - rec) ** 2)), 4)})
        if i in te:
            test_err.append(float(np.mean((im - rec) ** 2)))
            panels(n_, im, rec, err, z, boxes, os.path.join(viz, f"panels-{len(test_err)}.png"), s)
    filters(net, os.path.join(viz, "filters.png"), p)
    history = h1 + h2
    test = [r for r in results if r["set"] == "test"]
    metrics = {
        "Training patches": int(len(X)), "Set aside as hardest to redraw": int((~keep).sum()),
        "Loss, first and last epoch": [round(history[0], 4), round(history[-1], 4)],
        "Gradient check (linear output), largest difference": nn.gradient_check(activations=("Tanh",), classes=5, output="linear"),
        "Test images with at least one detection": f"{sum(r['boxes'] > 0 for r in test)} of {len(test)}",
        "All images with at least one detection": f"{sum(r['boxes'] > 0 for r in results)} of {len(results)}",
        "Detections per image, test (mean)": round(float(np.mean([r['boxes'] for r in test])), 2),
        "Mean squared difference, test images": round(float(np.mean(test_err)), 4),
        "Numbers in the net": net.weight_count, "Per image": results}
    path = export.save_run(net, runs_dir, name, settings=s, metrics=metrics, history=history,
                           timings={"training, s": round(t_train, 2)})
    if not scratch:
        card_p = os.path.join(NET, "card.json")
        card = json.load(open(card_p, encoding="utf-8"))
        card["latest_run"] = os.path.basename(path)
        export.write_json(card_p, card)
    print(os.path.basename(path), "| train", round(t_train, 1), "s |", metrics["Test images with at least one detection"],
          "|", metrics["All images with at least one detection"], "| per image", [r["boxes"] for r in results])


if __name__ == "__main__":
    main(*sys.argv[1:])
