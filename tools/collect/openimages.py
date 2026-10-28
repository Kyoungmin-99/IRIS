"""Open Images(구글, 사진 CC BY 2.0 · 라벨 CC BY 4.0)에서 사람이 확인한 라벨로 사진 직접 모으기.

  # 1) 라벨에서 원하는 분류 줄만(예: Tattoo /m/07j2f). 큰 파일은 저장하지 않고 받으면서 거른다
  curl -s https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv | grep ",/m/07j2f,1" | sed "s/^/val,/" > ids.csv
  # 2) 사진 정보(작성자·원본 주소·라이선스) 줄: {split}-images-with-rotation.csv 에서 같은 ID 줄만 → meta_*.csv
  # 3) 받기
  python openimages.py --ids ids.csv --meta 'meta_*.csv' --out /tmp/claude-0/collect/openimages --label tattoo
- 사진은 open-images-dataset S3 에서 받아 긴 변 1024 로 줄여 저장. 라이선스 줄이 CC BY / CC BY-SA 가 아니면 받지 않음
- ledger.jsonl: sha256 · 출처(원본 Flickr 주소) · 작성자 · 라이선스 · 받은 날 · commercial_ok · person_consent(unknown)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

UA = "IRIS-research/1.0 (virtual try-on research; rudals906377@gmail.com)"
SPLIT = {"val": "validation", "validation": "validation", "test": "test", "train": "train"}


def get(url: str, tries: int = 4) -> bytes:
    wait = 10
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):  # 없는 사진은 다시 받아도 없다
                raise RuntimeError(url) from e
            print("  다시:", str(e)[:80], flush=True)
            time.sleep(wait)
            wait *= 2
        except Exception as e:
            print("  다시:", str(e)[:80], flush=True)
            time.sleep(wait)
            wait *= 2
    raise RuntimeError(url)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", action="append", required=True, help="split,ImageID,... 줄(여러 개 가능)")
    ap.add_argument("--meta", required=True, help="사진 정보 CSV 글롭(ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="tattoo")
    ap.add_argument("--max", type=int, default=100000)
    a = ap.parse_args()
    cols = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL", "Author", "Title"]
    meta = {}
    for f in glob.glob(a.meta):
        for row in csv.reader(open(f, encoding="utf-8")):
            if row and row[0] != "ImageID":
                meta[row[0]] = dict(zip(cols, row))
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    led = out / "ledger.jsonl"
    seen = {json.loads(l)["image_id"] for l in led.read_text().splitlines()} if led.exists() else set()
    f = open(led, "a", encoding="utf-8")
    n = 0
    for ids in a.ids:
        for row in csv.reader(open(ids, encoding="utf-8")):
            if n >= a.max:
                break
            split, iid = SPLIT.get(row[0], row[0]), row[1]
            m = meta.get(iid)
            if iid in seen or not m or "creativecommons.org/licenses/by" not in m["License"] or "-nc" in m["License"] or "-nd" in m["License"]:
                continue
            data = None
            # S3 에 없으면(훈련 묶음 일부) 원본 Flickr 주소로
            for url in (f"https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg", m["OriginalURL"]):
                try:
                    data = get(url)
                    break
                except RuntimeError:
                    continue
            if data is None:
                continue
            im = Image.open(io.BytesIO(data)).convert("RGB")
            s = min(1, 1024 / max(im.size))
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=92)
            sha = hashlib.sha256(data).hexdigest()
            (out / "images" / f"{iid}.jpg").write_bytes(buf.getvalue())
            lic = m["License"].rstrip("/").split("/licenses/")[-1]
            f.write(json.dumps({
                "sha256": sha, "image_id": iid, "file": f"images/{iid}.jpg", "source": "open-images", "source_url": m["OriginalLandingURL"],
                "author": m["Author"], "author_url": m["AuthorProfileURL"], "license": "CC " + lic.upper().replace("/", " "), "license_url": m["License"],
                "label": a.label, "downloaded": datetime.date.today().isoformat(), "commercial_ok": True, "person_consent": "unknown",
            }, ensure_ascii=False) + "\n")
            f.flush()
            seen.add(iid)
            n += 1
            if n % 20 == 0:
                print(f"{n}장", flush=True)
            time.sleep(0.3)
    print(f"끝 {n}장", flush=True)


if __name__ == "__main__":
    main()
