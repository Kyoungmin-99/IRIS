"""Open Images 받을 사진 목록 만들기(openimages.py 입력): 사람이 확인한 라벨 → 사진 ID, 사진 정보(작성자·원본 주소·라이선스).

  python oi_prep.py --out /tmp/claude-0/collect/oi
- tattoo_ids.csv: 타투(/m/07j2f) 확인된 사진(train · validation · test)
- neg_ids.csv: 타투 없는 팔·손·소매·스웨터·손톱·팔찌·손목 사진(타투가 '없음'으로 확인된 것 먼저, 그다음 갈래마다 몫만큼).
  맨살 팔·니트 소매 무늬·손톱을 잉크로 잘못 잡던 것을 줄이는 '잉크 없음' 학습 조각용
- meta.csv: 위 사진들의 정보 줄(2018_04/image_ids_and_rotation.csv 에서 같은 ID 만)
큰 CSV(2~3GB)는 디스크에 두지 않고 받으면서 거른다. 끊기면 받은 바이트부터 Range 로 이어 받는다(컨테이너 재시작 · 연결 끊김).
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/openimages"
TATTOO = "/m/07j2f"
NEG = {"/m/0dzf4": "arm", "/m/0k65p": "hand", "/m/062581": "sleeve", "/m/019b80": "sweater", "/m/023j4r": "nail", "/m/01j48b": "wrist", "/m/01llx7": "bracelet"}
QUOTA = [("sweater", 300), ("sleeve", 300), ("nail", 300), ("bracelet", 150), ("wrist", 150), ("arm", 400), ("hand", 200)]


def lines(url: str):
    """큰 CSV 를 줄 단위로(끊기면 이어 받기)"""
    got, buf = 0, b""
    for _ in range(30):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={got}-"} if got else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        if buf:
                            yield buf.decode("utf-8", "replace")
                        return
                    got += len(chunk)
                    buf += chunk
                    *full, buf = buf.split(b"\n")
                    for l in full:
                        yield l.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # 없는 파일은 다시 받아도 없다
                raise
            print("  이어 받기:", str(e)[:80], got, flush=True)
            time.sleep(10)
        except Exception as e:
            print("  이어 받기:", str(e)[:80], got, flush=True)
            time.sleep(10)
    raise RuntimeError(url)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pos: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    tat1, tat0 = set(), set()
    want = set(NEG) | {TATTOO}
    for split, fn in (("train", "train"), ("validation", "val"), ("test", "test")):
        n = 0
        for l in lines(f"{BASE}/v7/oidv7-{fn}-annotations-human-imagelabels.csv"):
            r = l.rstrip("\r").split(",")
            if len(r) < 4 or r[2] not in want:
                continue
            k, conf = (split, r[0]), float(r[3])
            if r[2] == TATTOO:
                (tat1 if conf > 0.5 else tat0).add(k)
            elif conf > 0.5:
                pos[k].add(NEG[r[2]])
            n += 1
        print(split, "라벨 줄", n, flush=True)
    with open(out / "tattoo_ids.csv", "w") as f:
        for s, i in sorted(tat1):
            f.write(f"{s},{i}\n")
    cand = [k for k in pos if k not in tat1]
    random.seed(a.seed)
    random.shuffle(cand)
    sel = [k for k in cand if k in tat0]
    S = set(sel)
    for cls, q in QUOTA:
        add = [k for k in cand if cls in pos[k] and k not in S][:q]
        sel += add
        S.update(add)
    with open(out / "neg_ids.csv", "w") as f:
        for s, i in sel:
            f.write(f"{s},{i},{'+'.join(sorted(pos[(s, i)]))}\n")
    print("타투", len(tat1), "· 타투 없음", len(sel), collections.Counter(c for k in sel for c in pos[k]), flush=True)
    ids = {i for _, i in tat1} | {i for _, i in sel}
    n = 0
    with open(out / "meta.csv", "w", encoding="utf-8") as f:
        for l in lines(f"{BASE}/2018_04/image_ids_and_rotation.csv"):
            if l[:16] in ids:
                f.write(l.rstrip("\r") + "\n")
                n += 1
    print("사진 정보", n, "/", len(ids), flush=True)


if __name__ == "__main__":
    main()
