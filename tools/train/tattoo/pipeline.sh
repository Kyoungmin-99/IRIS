#!/bin/bash
# 타투 잉크 학습 전체를 처음부터(또는 이어서) 띄운다 — 컨테이너가 바뀌어 /tmp 의 사진·정답·저장본이 모두 사라져도 이 파일 하나로 다시 선다.
#   bash tools/train/tattoo/pipeline.sh        (몇 번 불러도 됨: 이미 도는 것·끝난 것은 건너뜀)
# 단계: 파이썬 패키지 → FINDE-photos(비공개 사진 저장소) → Open Images 목록·사진 받기(출처·라이선스 장부) → 글꼴·도안 → 합성 사진
#       → 선생님 정답(teacher.py) → 실제 사진 조각(crops.py) → '잉크 없음' 조각 → 학습 순환(600 배치마다 저장본 + 실제 사진 점수)
# 기록: $W/trainloop.log(바퀴마다 점수), 각 단계 로그는 $W 아래
set -u
W=${W:-/tmp/claude-0}
T=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$T/../../.." && pwd)
FP=${FP:-/home/user/finde-photos}
mkdir -p $W/collect $W/teach $W/tatout5
export HF_HOME=$W/hfcache
running() { pgrep -f "$1" > /dev/null; }

# 1) 파이썬 패키지(시스템 pip 이 다른 파이썬 것이라 python3 -m pip)
if ! python3 -c "import torch, transformers, onnx, onnxruntime, cv2, pyarrow" 2> /dev/null; then
  python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
  python3 -m pip install -q --ignore-installed pillow numpy setuptools
  python3 -m pip install -q onnx onnxscript onnxruntime transformers opencv-python-headless scipy huggingface_hub accelerate pyarrow
fi
# 2) FINDE-photos
[ -d $FP/collections ] || git clone -q --depth 1 https://github.com/rudals906377/FINDE-photos $FP
# 3) Open Images: 목록(타투 / 타투 없는 팔·손·소매·스웨터·손톱) → 사진(긴 변 1024, 장부 ledger.jsonl)
C=$W/collect
if [ ! -s $C/oi/meta.csv ] && ! running "[o]i_prep.py"; then
  (setsid nohup python3 $R/tools/collect/oi_prep.py --out $C/oi > $C/oi_prep.log 2>&1 < /dev/null &)
fi
if [ -s $C/oi/meta.csv ]; then
  for job in "tattoo_ids openimages tattoo oi" "neg_ids openimages_neg no-tattoo oi_neg"; do set -- $job
    if ! grep -q "^끝" $C/$4.log 2> /dev/null && ! running "[o]i/$1.csv"; then
      (setsid nohup python3 $R/tools/collect/openimages.py --ids $C/oi/$1.csv --meta "$C/oi/meta.csv" --out $C/$2 --label $3 >> $C/$4.log 2>&1 < /dev/null &)
    fi
  done
fi
# 4) 글꼴(OFL) · 실제 도안(Hugging Face Drozdik/tattoo_v0, CC BY-NC-SA — 연구용 판만)
if [ $(ls $W/fonts/*.ttf 2> /dev/null | wc -l) -lt 5 ]; then
  mkdir -p $W/fonts
  for f in nanumpenscript/NanumPenScript-Regular.ttf nanumbrushscript/NanumBrushScript-Regular.ttf nanummyeongjo/NanumMyeongjo-Regular.ttf \
           nanumgothic/NanumGothic-Regular.ttf gaegu/Gaegu-Regular.ttf greatvibes/GreatVibes-Regular.ttf pacifico/Pacifico-Regular.ttf \
           unifrakturmaguntia/UnifrakturMaguntia-Book.ttf; do
    curl -sSL -o $W/fonts/$(basename $f) https://raw.githubusercontent.com/google/fonts/main/ofl/$f
  done
fi
if [ $(ls $W/designs 2> /dev/null | wc -l) -lt 4000 ]; then
  mkdir -p $W/designs
  python3 - "$W" << 'PY'
import io, sys
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
w = sys.argv[1]
d = snapshot_download('Drozdik/tattoo_v0', repo_type='dataset', local_dir=f'{w}/hfdata/tattoo_v0')
import glob
for f in glob.glob(f'{d}/data/*.parquet'):
    for i, v in enumerate(pq.read_table(f).column('image').to_pylist()):
        Image.open(io.BytesIO(v['bytes'])).convert('RGB').save(f'{w}/designs/{i:05d}.jpg', quality=90)
PY
fi
# 5) 합성 사진 3,000장(배경: FINDE 네일·메이크업·헤어·손 사진)
B=$FP/collections/beauty-2026-10-20
if [ $(ls $W/tatdata3s/masks 2> /dev/null | wc -l) -lt 3000 ] && ! running "[s]ynth.py"; then
  (cd $T && setsid nohup nice -n 5 python3 synth.py --bg $B/네일 $B/메이크업 $B/헤어 $FP/collections/hands-2026-10-26 --fonts $W/fonts \
     --designs $W/designs --design-frac 0.5 --out $W/tatdata3s --n 3000 --seed 5 >> $W/synth.log 2>&1 < /dev/null &)
fi
# 6) 선생님 정답(끝난 사진은 건너뜀 → 새로 받은 사진만). 학습과 CPU 를 나눠 3 스레드
if ! running "[t]eacher.py"; then
  (cd $T && THREADS=3 setsid nohup python3 teacher.py --pc $FP --collect $C --out $W/teach/lab >> $W/teach/label.log 2>&1 < /dev/null &)
fi
# 7) 디스크 정리(매번): 다시 받을 수 있는 캐시 · 도안을 꺼낸 뒤의 원본 · 커진 로그
rm -rf /root/.cache/pip /root/.cache/uv /root/.npm/_cacache
[ $(ls $W/designs 2> /dev/null | wc -l) -ge 4000 ] && rm -rf $W/hfdata
running "[c]rops.py" || rm -rf $W/tatreal_new $W/tatneg_new
for f in $W/tattrain.log $W/teach/label.log $C/oi.log $C/oi_neg.log; do
  [ -f $f ] && [ $(stat -c %s $f) -gt 5000000 ] && tail -c 1000000 $f > $f.t && mv $f.t $f
done
# 8) 학습 순환
if ! running "[t]rain_loop"; then
  (setsid nohup bash $T/train_loop.sh >> $W/trainloop.log 2>&1 < /dev/null &)
fi
