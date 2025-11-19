# build_rewards_index.py  (one-time indexer)
import h5py, numpy as np, pandas as pd, re, os, argparse
SIM_RX = re.compile(r"(?:^|/)(?:sim_type|algorithm)=(?P<sim>[^/]+)(?:/|$)")
PEN_RX = re.compile(r"(?:^|/)(?:failure_penalty|penalty|fp)=(?P<pen>[-+]?\d*\.?\d+)(?:/|$)")

def get_attr_up(g, k):
    cur=g
    while True:
        if k in cur.attrs: return cur.attrs[k]
        if cur.parent is None or cur.parent.name==cur.name: return None
        cur=cur.parent

def decode(x): return x.decode() if isinstance(x,(bytes,bytearray)) else x

def main(h5, reward_name, out):
    rows=[]
    with h5py.File(h5, "r") as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Group): return
            if reward_name in obj and isinstance(obj[reward_name], h5py.Dataset):
                ds = obj[reward_name]
                if ds.size!=1: return
                val = float(np.ravel(ds[()])[0])
                sim = decode(get_attr_up(obj,"sim_type") or get_attr_up(obj,"algorithm"))
                pen = get_attr_up(obj,"failure_penalty") or get_attr_up(obj,"penalty") or get_attr_up(obj,"fp")
                if pen is not None:
                    try: pen=float(pen)
                    except: pen=None
                if sim is None:
                    m=SIM_RX.search(obj.name); sim = m.group("sim") if m else None
                if pen is None:
                    m=PEN_RX.search(obj.name); pen=float(m.group("pen")) if m else None
                if sim is None or pen is None: return
                rows.append((obj.name, sim, pen, val))
        f.visititems(visit)
    df = pd.DataFrame(rows, columns=["episode_path","sim_type","failure_penalty","total_reward"])
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--reward-name", required=True)  # e.g., total_reward
    ap.add_argument("--out", required=True)          # index.csv or index.parquet
    main(**vars(ap.parse_args()))
