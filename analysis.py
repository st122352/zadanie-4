"""Linear and small NumPy MLP comparison for the CCPP dataset.

Run with Python 3.10+ and NumPy:
    python analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ccpp.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20261117
HIDDEN = (32, 16)
ALPHAS = (1e-4, 1e-3, 1e-2)
MAX_EPOCHS = 1600
PATIENCE = 60
HOT_QUANTILE = 0.80
MODEL_NAMES = {"linear": "Linear", "relu": "MLP ReLU", "tanh": "MLP tanh"}
COLORS = {"linear": "#2878b5", "relu": "#d97827", "tanh": "#358b6b"}


def load_data() -> tuple[np.ndarray, np.ndarray]:
    with DATA_PATH.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["AT", "V", "AP", "RH", "PE"]
        if reader.fieldnames != expected:
            raise ValueError(f"Expected columns {expected}, found {reader.fieldnames}")
        rows = [[float(row[c]) for c in expected] for row in reader]
    data = np.asarray(rows, dtype=float)
    if data.shape != (9568, 5) or not np.isfinite(data).all():
        raise ValueError(f"Expected 9568 complete records and five columns; got {data.shape}")
    return data[:, :4], data[:, 4]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def standardize_y_fit(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean, scale = float(y.mean()), float(y.std(ddof=0))
    if scale < 1e-12:
        scale = 1.0
    return (y - mean) / scale, mean, scale


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)),
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    z, x_mean, x_scale = standardize_fit(x)
    y_z, y_mean, y_scale = standardize_y_fit(y)
    design = np.column_stack([np.ones(len(z)), z])
    weights = np.linalg.lstsq(design, y_z, rcond=None)[0]
    return {"weights": weights, "x_mean": x_mean, "x_scale": x_scale,
            "y_mean": y_mean, "y_scale": y_scale}


def predict_linear(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["x_mean"]) / model["x_scale"]
    y_z = np.column_stack([np.ones(len(z)), z]) @ model["weights"]
    return y_z * model["y_scale"] + model["y_mean"]


def activate(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(z, 0)
    if kind == "tanh":
        return np.tanh(z)
    raise ValueError(kind)


def activation_grad(z: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return (z > 0).astype(float)
    return 1 - a**2


def init_weights(kind: str, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    dimensions = [4, *HIDDEN, 1]
    weights = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
        if kind == "relu":
            scale = np.sqrt(2 / fan_in)
        else:
            scale = np.sqrt(2 / (fan_in + fan_out))
        weights.append(rng.normal(0, scale, (fan_in, fan_out)))
        weights.append(np.zeros(fan_out))
    return weights


def mlp_forward(x: np.ndarray, params: list[np.ndarray], kind: str, cache: bool = False):
    z1 = x @ params[0] + params[1]
    a1 = activate(z1, kind)
    z2 = a1 @ params[2] + params[3]
    a2 = activate(z2, kind)
    pred = a2 @ params[4] + params[5]
    if cache:
        return pred[:, 0], (x, z1, a1, z2, a2)
    return pred[:, 0]


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    alpha: float,
    *,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    epochs: int | None = None,
    seed: int = SEED,
) -> dict:
    """Full-batch Adam with validation-based early stopping when validation is supplied."""
    x_z, x_mean, x_scale = standardize_fit(x)
    y_z, y_mean, y_scale = standardize_y_fit(y)
    params = init_weights(kind, seed)
    first_m = [np.zeros_like(p) for p in params]
    second_m = [np.zeros_like(p) for p in params]
    beta1, beta2, eps, learning_rate = 0.9, 0.999, 1e-8, 0.01
    best_params = [p.copy() for p in params]
    best_epoch = 0
    best_score = float("inf")
    stale = 0
    max_epochs = epochs if epochs is not None else MAX_EPOCHS

    for step in range(1, max_epochs + 1):
        pred, cache = mlp_forward(x_z, params, kind, cache=True)
        inputs, z1, a1, z2, a2 = cache
        residual = pred - y_z
        n = len(x_z)
        d3 = residual[:, None] / n
        grad_w3 = a2.T @ d3 + alpha * params[4]
        grad_b3 = d3.sum(axis=0)
        d2 = d3 @ params[4].T * activation_grad(z2, a2, kind)
        grad_w2 = a1.T @ d2 + alpha * params[2]
        grad_b2 = d2.sum(axis=0)
        d1 = d2 @ params[2].T * activation_grad(z1, a1, kind)
        grad_w1 = inputs.T @ d1 + alpha * params[0]
        grad_b1 = d1.sum(axis=0)
        grads = [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]
        for i, grad in enumerate(grads):
            first_m[i] = beta1 * first_m[i] + (1 - beta1) * grad
            second_m[i] = beta2 * second_m[i] + (1 - beta2) * grad**2
            m_hat = first_m[i] / (1 - beta1**step)
            v_hat = second_m[i] / (1 - beta2**step)
            params[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        if validation is not None:
            xv, yv = validation
            xv_z = (xv - x_mean) / x_scale
            yv_prediction = mlp_forward(xv_z, params, kind) * y_scale + y_mean
            score = float(np.sqrt(np.mean((yv_prediction - yv)**2)))
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = step
                best_params = [p.copy() for p in params]
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    if validation is None:
        best_params, best_epoch = params, max_epochs
    return {"kind": kind, "params": best_params, "x_mean": x_mean, "x_scale": x_scale,
            "y_mean": y_mean, "y_scale": y_scale, "alpha": alpha, "epochs": best_epoch,
            "validation_rmse": best_score}


def predict_mlp(x: np.ndarray, model: dict) -> np.ndarray:
    x_z = (x - model["x_mean"]) / model["x_scale"]
    return mlp_forward(x_z, model["params"], model["kind"]) * model["y_scale"] + model["y_mean"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_curve(grid: np.ndarray, curves: dict[str, np.ndarray], hot_threshold: float) -> str:
    w, h, left, top, pw, ph = 850, 500, 84, 55, 700, 350
    all_y = np.concatenate(list(curves.values()))
    ymin, ymax = float(all_y.min()), float(all_y.max())
    pad = 0.08 * max(ymax-ymin, 1)
    ymin, ymax = ymin-pad, ymax+pad
    xmin, xmax = float(grid.min()), float(grid.max())
    sx = lambda a: left + (float(a)-xmin)/(xmax-xmin)*pw
    sy = lambda a: top + ph - (float(a)-ymin)/(ymax-ymin)*ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="425" y="27" text-anchor="middle" font-family="Arial" font-size="18">Power response to temperature (other inputs fixed)</text>']
    hot_x = sx(hot_threshold)
    out.append(f'<rect x="{hot_x:.1f}" y="{top}" width="{left+pw-hot_x:.1f}" height="{ph}" fill="#fbe9e7"/>')
    for i in range(5):
        value = ymin + (ymax-ymin)*i/4
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e6e6e6"/>',
                f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>']
    out += [f'<line x1="{hot_x:.1f}" y1="{top}" x2="{hot_x:.1f}" y2="{top+ph}" stroke="#aa5550" stroke-dasharray="5,4"/>',
            f'<text x="{min(hot_x+5,left+pw-5):.1f}" y="{top+15}" font-family="Arial" font-size="10" fill="#8d3c38">held-out hot range</text>']
    for key, values in curves.items():
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a,b in zip(grid,values))
        out.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[key]}" stroke-width="2.5"/>')
    out += [f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
            f'<text x="{left+pw/2}" y="{top+ph+40}" text-anchor="middle" font-family="Arial" font-size="13">Ambient temperature AT (°C)</text>',
            f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20 {top+ph/2})" text-anchor="middle" font-family="Arial" font-size="13">Net electrical power PE (MW)</text>']
    for i,key in enumerate(curves):
        x = 220 + i*160
        out += [f'<line x1="{x}" y1="462" x2="{x+25}" y2="462" stroke="{COLORS[key]}" stroke-width="3"/>',
                f'<text x="{x+31}" y="466" font-family="Arial" font-size="12">{MODEL_NAMES[key]}</text>']
    out.append('</svg>')
    return "\n".join(out)


def svg_error_bins(rows: list[dict]) -> str:
    models = ["linear", "relu", "tanh"]
    bins = list(dict.fromkeys(row["temperature_bin"] for row in rows))
    w, h, left, top, pw, ph = 960, 540, 100, 78, 810, 370
    vmax = max(row["rmse_mw"] for row in rows) * 1.12
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="480" y="27" text-anchor="middle" font-family="Arial" font-size="18">Prediction error by temperature interval</text>',
           '<text x="480" y="49" text-anchor="middle" font-family="Arial" font-size="11">Cooler bins: inner validation; shaded bins: completely held-out hot tail</text>']
    hot_start = next((i for i,b in enumerate(bins) if b.startswith("Q80")), len(bins))
    slot = pw / len(bins)
    if hot_start < len(bins):
        out.append(f'<rect x="{left+hot_start*slot:.1f}" y="{top}" width="{(len(bins)-hot_start)*slot:.1f}" height="{ph}" fill="#fbe9e7"/>')
    for i in range(5):
        val = vmax*i/4
        y = top+ph-val/vmax*ph
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#e6e6e6"/>',
                f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val:.1f}</text>']
    groupw = slot*0.68
    barw = groupw/len(models)
    for i,b in enumerate(bins):
        rows_b = {r["model"]:r for r in rows if r["temperature_bin"]==b}
        cx = left+(i+0.5)*slot
        for j,model in enumerate(models):
            if model not in rows_b:
                continue
            value = rows_b[model]["rmse_mw"]
            bh = value/vmax*ph
            x = cx-groupw/2+j*barw
            y=top+ph-bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barw-2:.1f}" height="{bh:.1f}" fill="{COLORS[model]}"/>')
        out.append(f'<text x="{cx:.1f}" y="{top+ph+20}" text-anchor="middle" font-family="Arial" font-size="10">{b}</text>')
    out += [f'<text x="19" y="{top+ph/2}" transform="rotate(-90 19 {top+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">RMSE (MW)</text>',
            f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>']
    for i,model in enumerate(models):
        x=245+i*165
        out += [f'<rect x="{x}" y="490" width="12" height="12" fill="{COLORS[model]}"/>',
                f'<text x="{x+18}" y="501" font-family="Arial" font-size="12">{MODEL_NAMES[model]}</text>']
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    x, y = load_data()
    threshold = float(np.quantile(x[:,0], HOT_QUANTILE))
    hot_mask = x[:,0] >= threshold
    cool_idx = np.flatnonzero(~hot_mask)
    hot_idx = np.flatnonzero(hot_mask)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(cool_idx)
    n_val = int(round(0.20*len(cool_idx)))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    if len(hot_idx) < 100:
        raise ValueError("Hot-condition holdout unexpectedly small")

    fitted_validation: dict[str, object] = {}
    configs, selection_rows = {}, []
    linear_val_model = fit_linear(x[train_idx], y[train_idx])
    fitted_validation["linear"] = linear_val_model
    linear_val_pred = predict_linear(x[val_idx], linear_val_model)
    selection_rows.append({"model":"linear","activation":"none","alpha":0.0,"epochs":0,
                           **metrics(y[val_idx],linear_val_pred)})
    best_config = {}
    for offset, kind in enumerate(("relu", "tanh")):
        candidates=[]
        for alpha_i, alpha in enumerate(ALPHAS):
            model=fit_mlp(x[train_idx],y[train_idx],kind,alpha,
                          validation=(x[val_idx],y[val_idx]),seed=SEED+offset*100+alpha_i)
            pred=predict_mlp(x[val_idx],model)
            score=metrics(y[val_idx],pred)
            row={"model":MODEL_NAMES[kind],"activation":kind,"alpha":alpha,"epochs":model["epochs"],**score}
            selection_rows.append(row)
            candidates.append((score["rmse_mw"],alpha,model,score))
        _,alpha,model,score=min(candidates,key=lambda z:z[0])
        best_config[kind]={"alpha":alpha,"epochs":model["epochs"],"validation_rmse_mw":score["rmse_mw"]}
        fitted_validation[kind]=model
    write_csv(RESULTS/"validation_hyperparameter_search.csv",selection_rows)

    # Refit after tuning on all non-hot records; the hot tail remains untouched until scoring.
    final_models={"linear":fit_linear(x[cool_idx],y[cool_idx])}
    for offset,kind in enumerate(("relu","tanh")):
        final_models[kind]=fit_mlp(x[cool_idx],y[cool_idx],kind,best_config[kind]["alpha"],
                                   epochs=best_config[kind]["epochs"],seed=SEED+1000+offset)
    results=[]
    predictions={}
    partitions={
        "train":(train_idx,True),
        "validation":(val_idx,False),
        "hot_test":(hot_idx,False),
    }
    for key,model in final_models.items():
        if key=="linear":
            train_pred=predict_linear(x[cool_idx],model)
            hot_pred=predict_linear(x[hot_idx],model)
            val_pred=predict_linear(x[val_idx],fitted_validation[key])
        else:
            train_pred=predict_mlp(x[cool_idx],model)
            hot_pred=predict_mlp(x[hot_idx],model)
            val_pred=predict_mlp(x[val_idx],fitted_validation[key])
        predictions[key]={"train":train_pred,"validation":val_pred,"hot_test":hot_pred}
        for partition,truth,pred in [
            ("train",y[cool_idx],train_pred),("validation",y[val_idx],val_pred),("hot_test",y[hot_idx],hot_pred)
        ]:
            row={"model":MODEL_NAMES[key],"model_key":key,"partition":partition,
                 "n":len(truth),**metrics(truth,pred)}
            if key in best_config:
                row.update(best_config[key])
            results.append(row)
    write_csv(RESULTS/"metrics.csv",results)

    # Temperature-bin chart: validation covers cooler conditions; the hottest bins are held out.
    edges=np.quantile(x[:,0],[0,.25,.50,.65,.80,.90,.95,1.0])
    labels=["Q00–25","Q25–50","Q50–65","Q65–80","Q80–90","Q90–95","Q95–100"]
    error_rows=[]
    for key in ("linear","relu","tanh"):
        for partition,idx,pred in [("inner_validation",val_idx,predictions[key]["validation"]),
                                   ("heldout_hot_test",hot_idx,predictions[key]["hot_test"])]:
            bins=np.digitize(x[idx,0],edges[1:-1],right=False)
            for b,label in enumerate(labels):
                mask=bins==b
                if np.any(mask):
                    error_rows.append({"model":key,"partition":partition,"temperature_bin":label,
                                       "at_min_c":float(x[idx[mask],0].min()),"at_max_c":float(x[idx[mask],0].max()),
                                       "n":int(mask.sum()),"rmse_mw":metrics(y[idx[mask]],pred[mask])["rmse_mw"]})
    write_csv(RESULTS/"temperature_bin_errors.csv",error_rows)

    # Partial response curve at the median V, AP and RH of non-hot training conditions.
    grid=np.linspace(float(x[:,0].min()),float(x[:,0].max()),240)
    fixed=np.median(x[cool_idx],axis=0)
    curve_rows=[]
    curves={}
    for key,model in final_models.items():
        curve_x=np.tile(fixed,(len(grid),1))
        curve_x[:,0]=grid
        pred=predict_linear(curve_x,model) if key=="linear" else predict_mlp(curve_x,model)
        curves[key]=pred
        delta=np.diff(pred)
        curve_rows.extend([
            {"model":key,"AT_c":float(at),"predicted_PE_mw":float(power),
             "V_fixed":float(fixed[1]),"AP_fixed":float(fixed[2]),"RH_fixed":float(fixed[3])}
            for at,power in zip(grid,pred)
        ])
        best_config.setdefault(key,{})
        best_config[key]["curve_positive_slope_share"]=float(np.mean(delta>1e-8))
        best_config[key]["curve_total_change_mw"]=float(pred[-1]-pred[0])
        best_config[key]["curve_range_mw"]=float(pred.max()-pred.min())
    write_csv(RESULTS/"temperature_response_curve.csv",curve_rows)

    # Dataset/experiment summary for report and reproducibility.
    train_at_bins=np.quantile(x[:,0],[.5,.65,.8,.9,.95,1])
    summary={
        "rows":int(len(x)),"input_count":4,"target":"PE (MW)","seed":SEED,
        "hot_quantile":HOT_QUANTILE,"hot_threshold_c":threshold,
        "split_counts":{"cool_pool":int(len(cool_idx)),"tuning_train":int(len(train_idx)),
                        "validation":int(len(val_idx)),"hot_test":int(len(hot_idx))},
        "temperature_quantiles_c":{"50":float(train_at_bins[0]),"65":float(train_at_bins[1]),
                                   "80":float(train_at_bins[2]),"90":float(train_at_bins[3]),
                                   "95":float(train_at_bins[4]),"100":float(train_at_bins[5])},
        "architecture":list(HIDDEN),"alpha_grid":list(ALPHAS),"selected":best_config,
        "model_metrics":results,"temperature_error_bins":error_rows,
        "fixed_inputs":{"V_cm_Hg":float(fixed[1]),"AP_mbar":float(fixed[2]),"RH_percent":float(fixed[3])},
    }
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (FIGURES/"temperature_response.svg").write_text(svg_curve(grid,curves,threshold),encoding="utf-8")
    (FIGURES/"temperature_error_bins.svg").write_text(svg_error_bins(error_rows),encoding="utf-8")
    print(f"Rows={len(x)}; hot cutoff AT >= {threshold:.3f} °C; hot test n={len(hot_idx)}")
    for row in results:
        print(f"{row['model']:10s} {row['partition']:10s} n={row['n']:4d} "
              f"MAE={row['mae_mw']:.3f} RMSE={row['rmse_mw']:.3f} R2={row['r2']:.4f}")
    print("Selected MLP settings:")
    for key in ("relu","tanh"):
        print(f"  {key}: alpha={best_config[key]['alpha']}, epochs={best_config[key]['epochs']}, "
              f"curve positive-slope share={best_config[key]['curve_positive_slope_share']:.3f}, "
              f"end-to-end change={best_config[key]['curve_total_change_mw']:.3f} MW")


if __name__ == "__main__":
    main()
