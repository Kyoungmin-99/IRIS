#!/bin/bash
# 타투 잉크 5단계 U-Net 을 멈추지 않고 학습(pipeline.sh 가 띄움): 정답이 200장 늘면 실제 사진 조각 새로, '잉크 없음' 사진이 늘면 그 조각도 새로
# → 600 배치(부하 없을 때 약 35분) 이어 학습 → 저장본을 앱과 같은 calib 0.85 로 지금 앱과 함께 채점($W/trainloop.log)
# 저장본(.pt)이 없으면(컨테이너 교체) 앱 ONNX 에서 가중치를 되살려(unet.from_onnx) BN 고정으로 이어 학습
W=${W:-/tmp/claude-0}
T=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$T/../../.." && pwd)
FP=${FP:-/home/user/finde-photos}
O=$W/tatout5
L=$W/trainloop.log
APP=$R/web/public/models/tattoo-ink.onnx
cd $T
mkdir -p $O
log() { echo "$(date -u +%m-%d\ %H:%M) $*" >> $L; }
# 끝난 바퀴의 저장본인데 ONNX 나 점수가 없는 것(내보내기·채점 도중 컨테이너가 꺼짐)을 마저 내보내고 채점
score_pending() {
  for pt in $(ls -t $O/snap_*.pt 2> /dev/null | head -3); do
    s=$(basename $pt .pt)
    grep -q "^${s}_c85 {" $L 2> /dev/null && continue
    if [ ! -s $O/${s}_c85.onnx ]; then
      python3 export.py $pt $O/${s}_c85.onnx.tmp --calib 0.85 > /dev/null 2>&1 && mv $O/${s}_c85.onnx.tmp $O/${s}_c85.onnx || { rm -f $O/${s}_c85.onnx.tmp; continue; }
    fi
    nice -n 5 python3 eval_real.py --lab $W/teach/lab ${s}_c85=$O/${s}_c85.onnx app=$APP >> $L 2>&1
  done
}
while true; do
  # 실제 사진 조각(선생님 정답, 시험 전용 사진 제외)
  n=$(ls $W/teach/lab/*.json 2> /dev/null | wc -l)
  last=$(cat $O/crops_n 2> /dev/null || echo 0)
  if [ $n -ge 100 ] && { [ $((n - last)) -ge 200 ] || [ ! -d $W/tatreal ]; }; then
    rm -rf $W/tatreal_new && nice -n 5 python3 crops.py --lab $W/teach/lab --out $W/tatreal_new >> $L 2>&1 \
      && rm -rf $W/tatreal && mv $W/tatreal_new $W/tatreal && echo $n > $O/crops_n && log "조각 새로(정답 $n)"
  fi
  # '잉크 없음' 조각: 타투 없는 맨살 팔(FINDE dot-arms, 학습용) + Open Images 타투 없는 팔·손·소매·스웨터·손톱 사진
  python3 - "$W" "$FP" > $W/neg_list.new << 'PY'
import glob, hashlib, json, os, sys
w, fp = sys.argv[1], sys.argv[2]
out = []
for mf in sorted(glob.glob(f'{fp}/dot-candidates/dot-arms-*/*.json')):
    d = json.load(open(mf)); items = d['items'] if isinstance(d, dict) and 'items' in d else (d if isinstance(d, list) else d.get('candidates', []))
    for it in items:
        p = os.path.join(os.path.dirname(mf), it['file'])
        # teacher.py 와 같은 시험 전용 규칙(이름 해시 4장에 1장)은 빼고
        if os.path.exists(p) and not it.get('has_tattoo') and int(hashlib.sha1(os.path.basename(p).encode()).hexdigest(), 16) % 4:
            out.append(p)
# Open Images 타투 없는 사진: 사람 확인 '타투 없음'이 아닌 갈래 몫 사진에는 선생님이 잉크를 붙인 것이 있다(1,233장 중 126장).
# 그 사진이 실제 사진 조각(잉크)과 잉크 없음 조각(잉크 0)에 함께 들어가 서로 반대 정답이 됐다 → 선생님 정답이 train 이고 잉크가 없는 것만
# (시험 전용 eval 사진도 여기로 새고 있었다)
import numpy as np
from PIL import Image
lab = f'{w}/teach/lab'
for jf in sorted(glob.glob(f'{lab}/*.json')):
    m = json.load(open(jf))
    if 'openimages_neg/' not in m.get('src', '') or 'error' in m or m.get('split') != 'train' or not os.path.exists(m['src']):
        continue
    if (np.asarray(Image.open(jf[:-5] + '_ink.png')) > 128).mean() <= 0.002:
        out.append(m['src'])
print('\n'.join(out))
PY
  nn=$(wc -l < $W/neg_list.new)
  if [ $nn -ge 50 ] && [ $((nn - $(cat $O/neg_n 2> /dev/null || echo 0))) -ge 200 ]; then
    mv $W/neg_list.new $W/neg_list.txt
    rm -rf $W/tatneg_new && nice -n 5 python3 crops.py --neg $W/neg_list.txt --out $W/tatneg_new >> $L 2>&1 \
      && rm -rf $W/tatneg && mv $W/tatneg_new $W/tatneg && echo $nn > $O/neg_n && log "잉크 없음 조각 새로(사진 $nn)"
  fi
  D="$W/tatdata3s"
  [ -d $W/tatcompc ] && D="$D $W/tatcompc"
  [ -d $W/tatreal ] && D="$D $W/tatreal:2"
  # 잉크 없음 ×2: ×2(사진 763장) 때 맨살 팔 오검출이 2 → 5장이라 ×4 로 올렸지만, 사진이 1,300장을 넘은 뒤 ×4 는
  # 바퀴마다 잉크를 덜 잡아 IoU 가 0.53 → 0.44 로 떨어졌다(상자 밖 오검출 37 → 18%) → 앱 가중치에서 ×2 로 다시(첫 바퀴 IoU 0.50 · 맨살 팔 3장, 아직 앱 0.538 아래). NEGX 로 바꿔 시험
  [ -d $W/tatneg ] && D="$D $W/tatneg:${NEGX:-2}"
  score_pending
  RES=$O/last.pt
  # 잘린 저장본(저장 도중 꺼짐)으로 이어 하면 매번 불러오기에 실패해 학습이 영영 멈춘다 → 치워 두고 앱 가중치에서 다시
  if [ -f $RES ] && ! python3 -c "import torch; torch.load('$RES', map_location='cpu')" 2> /dev/null; then
    mv $RES $RES.bad; rm -f $O/round_steps; log "last.pt 가 깨져 치움 → 앱 가중치에서 다시"
  fi
  if [ ! -f $RES ]; then
    python3 -c "import sys, torch; sys.path.insert(0, '$T'); from unet import from_onnx; torch.save(from_onnx('$APP', 0.85).state_dict(), '$O/from_app.pt')" && log "앱 ONNX 에서 가중치 되살림"
    RES=$O/from_app.pt
  fi
  # 합성만으로 학습하면 실제 사진 점수가 떨어진다(합성 IoU ≠ 실제): 실제 사진 조각이 생길 때까지 기다림
  if [ ! -d $W/tatreal ] || [ ! -d $W/tatdata3s/masks ] || [ $(ls $W/tatdata3s/masks | wc -l) -lt 1000 ]; then sleep 300; continue; fi
  # 컨테이너가 재시작되면 바퀴가 0 배치부터 다시 돌아 600 배치를 못 채우고 점수가 안 나왔다: 이 바퀴에서 이미 한 배치(90초마다 저장)를 빼고 나머지만
  done_n=$(cat $O/round_steps 2> /dev/null || echo 0)
  left=$((600 - done_n)); [ $left -lt 60 ] && left=60
  python3 train.py --data $D --out $O --epochs 1 --steps $left --step0 $done_n --lr 2e-4 --ch 16,32,64,128,256 --freeze-bn --resume $RES >> $W/tattrain.log 2>&1 || { sleep 60; continue; }
  stamp=$(date -u +%m%d%H%M)
  # 저장본을 먼저 남긴 뒤에 바퀴 진행을 지운다(그 사이에 꺼져도 끝난 바퀴가 사라지지 않게). 내보내기·채점은 score_pending 이 맡는다
  cp $O/last.pt $O/snap_$stamp.pt.tmp && mv $O/snap_$stamp.pt.tmp $O/snap_$stamp.pt
  rm -f $O/round_steps
  log "바퀴 끝 snap_$stamp"
  score_pending
  # 저장본은 최근 6개만(디스크)
  ls -t $O/snap_*.pt 2> /dev/null | tail -n +7 | xargs -r rm -f
  ls -t $O/snap_*_c85.onnx 2> /dev/null | tail -n +7 | xargs -r rm -f
done
