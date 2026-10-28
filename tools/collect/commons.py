"""위키미디어 커먼즈에서 학습 사진 직접 모으기(이용 조건이 공개된 사진만, 출처·라이선스를 함께 기록).

  python commons.py --out /tmp/claude-0/collect/commons --query "tattoo arm" --query "tattooed forearm" --max 400
- 검색(파일 이름공간) + 사진 정보(라이선스 · 작성자 · 원본 주소)를 한 번에 받는다(50장씩). 긴 변 1024 미리보기만 받는다
- 라이선스: CC0 · 퍼블릭 도메인 · CC BY · CC BY-SA 만(상업 이용 가능). 그 밖(NC · ND · 공정 이용 등)은 받지 않음
- 기록: out/ledger.jsonl 한 줄 = 사진 하나(sha256 · 출처 주소 · 작성자 · 라이선스 · 라이선스 주소 · 받은 날 · commercial_ok)
  → FINDE-photos 비공개 장부(rights/ledger.jsonl) 형식과 같음
- 예의: 요청 사이 1.5초, 거절(429·5xx)되면 30초 → 60초 … 기다렸다 다시. 이어서 하기(받은 것은 건너뜀)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "IRIS-research/1.0 (virtual try-on research; rudals906377@gmail.com)"
OK = re.compile(r"^(cc0|public domain|pd|cc[- ]by(-sa)?( \d\.\d)?( [a-z\-]+)?)$", re.I)


def get(url: str, tries: int = 6) -> bytes:
    wait = 30
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # 429 · 5xx · 끊김
            print("  기다림", wait, "초:", str(e)[:80], flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 300)
    raise RuntimeError("계속 실패: " + url[:120])


def search(query: str, offset: int) -> tuple[list[dict], int | None]:
    q = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6",
        "gsrlimit": "50", "gsroffset": str(offset), "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1024",
    }
    d = json.loads(get(API + "?" + urllib.parse.urlencode(q)))
    pages = list(d.get("query", {}).get("pages", {}).values())
    nxt = d.get("continue", {}).get("gsroffset")
    return pages, (int(nxt) if nxt is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", action="append", required=True)
    ap.add_argument("--max", type=int, default=400, help="검색어마다 최대 장수")
    ap.add_argument("--minside", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    led = out / "ledger.jsonl"
    seen = {json.loads(l)["source_url"] for l in led.read_text().splitlines()} if led.exists() else set()
    f = open(led, "a", encoding="utf-8")
    for query in a.query:
        got, offset = 0, 0
        while offset is not None and got < a.max:
            pages, offset = search(query, offset)
            time.sleep(1.5)
            for p in pages:
                ii = (p.get("imageinfo") or [{}])[0]
                meta = {k: v.get("value", "") for k, v in (ii.get("extmetadata") or {}).items()}
                lic = re.sub(r"<[^>]+>", "", meta.get("LicenseShortName", "")).strip()
                src = ii.get("descriptionurl", "")
                if src in seen or ii.get("mime") not in ("image/jpeg", "image/png") or min(ii.get("width", 0), ii.get("height", 0)) < a.minside:
                    continue
                if not OK.match(lic):
                    continue
                url = ii.get("thumburl") or ii.get("url")
                try:
                    data = get(url)
                except RuntimeError as e:
                    print(e, flush=True)
                    continue
                time.sleep(1.5)
                sha = hashlib.sha256(data).hexdigest()
                ext = ".png" if ii.get("mime") == "image/png" else ".jpg"
                (out / "images" / f"{sha[:16]}{ext}").write_bytes(data)
                f.write(json.dumps({
                    "sha256": sha, "file": f"images/{sha[:16]}{ext}", "source": "wikimedia-commons", "source_url": src,
                    "title": p.get("title", ""), "author": re.sub(r"<[^>]+>", "", meta.get("Artist", "")).strip()[:200],
                    "license": lic, "license_url": meta.get("LicenseUrl", ""), "query": query,
                    "downloaded": datetime.date.today().isoformat(), "commercial_ok": True,
                    "person_consent": "unknown",  # 인물 사진은 초상권·동의 확인 필요(상업판 학습 전 법률 확인)
                }, ensure_ascii=False) + "\n")
                f.flush()
                seen.add(src)
                got += 1
            print(f"{query}: {got}장", flush=True)
    print("끝", flush=True)


if __name__ == "__main__":
    main()
