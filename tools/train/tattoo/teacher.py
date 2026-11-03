"""선생님 모델로 실제 사진의 타투 정답 만들기(이어서 하기: 끝난 사진은 건너뜀). 모두 상업 이용 가능한 공개 모델, CPU 로 사진당 10초 안팎.

  python teacher.py --pc /tmp/claude-0/pc --out /tmp/claude-0/teach/lab   (FINDE-photos photo-collection 을 푼 폴더)
그다음 crops.py 로 학습 조각을 만들고 train.py --data <합성> <조각>:3 으로 섞어 학습, eval_real.py 로 실제 사진 점수.
"""
#  1) Grounding DINO tiny(Apache-2.0) 'a tattoo.' 상자  2) SigLIP2 base(Apache-2.0)로 상자 안을 다시 보고 타투/아님 판정
#  3) SAM 2.1 tiny(Apache-2.0) 상자 → 영역  4) 영역 안에서 잉크 픽셀 = 블랙햇(가는 어두운 획) 또는 피부색과 색 차이(색 잉크)
# 출력(768 크기): lab/{id}.jpg 그림, {id}_ink.png 잉크 정답(0~255), {id}_w.png 학습 가중치(0=모름), {id}.json 상자·점수
import argparse, glob, json, os, sys, time, hashlib
import numpy as np, torch, cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, Sam2Model, Sam2Processor
torch.set_num_threads(int(os.environ.get('THREADS', '3')))
ap = argparse.ArgumentParser()
ap.add_argument('--pc', default='/home/user/finde-photos', help='FINDE-photos 를 받은 폴더(collections · dot-candidates)')
ap.add_argument('--out', default='/tmp/claude-0/teach/lab')
ap.add_argument('--collect', default='/tmp/claude-0/collect', help='직접 모은 사진 폴더(tools/collect 출력)')
ap.add_argument('--holdout', default='/tmp/claude-0/inkeval/holdout_neg.txt', help='시험 전용 맨살 팔 목록(한 줄에 경로 하나)')
A = ap.parse_args()
OUT, PC = A.out, A.pc
os.makedirs(OUT, exist_ok=True)

def sources():
    tat = sorted(glob.glob(f'{PC}/collections/beauty-2026-10-20/타투/*/*'))
    ev = set(tat[::4])  # inkeval/eval.py 가 쓰는 타투 사진(4장에 1장) → 시험 전용
    L = [(p, 'tat', 'eval' if p in ev else 'train') for p in tat]
    L += [(p, 'tat', 'train') for p in sorted(glob.glob(f'{PC}/collections/style_kr-2026-10-27/tattoo/*'))]
    # 코덱스 요청 11(약한 유형: 일상 스냅 · 작은 · 어두운 · 컬러) · 요청 3/8(실제 타투) 묶음: 약한 유형 점수를 따로 재도록 이름 해시로 2장에 1장 시험 전용
    for p in sorted(glob.glob(f'{PC}/collections/tattoo_weak-*/images/*') + glob.glob(f'{PC}/collections/tattoo_real-*/*/*')):
        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            L.append((p, 'tat', 'eval' if int(hashlib.sha1(os.path.basename(p).encode()).hexdigest(), 16) % 2 == 0 else 'train'))
    # 코덱스 요청 7(타투로 잘못 읽히는 맨살): 맨살 팔과 같이(시험 전용이면 오검출로 채점)
    for p in sorted(glob.glob(f'{PC}/collections/tattoo_negative-*/images/*')):
        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            L.append((p, 'arm', 'eval' if int(hashlib.sha1(os.path.basename(p).encode()).hexdigest(), 16) % 2 == 0 else 'train'))
    # 시험 전용 맨살 팔: 목록 파일이 없으면(컨테이너가 바뀌어 사라짐) 파일 이름 해시로 4장에 1장 — 다시 만들어도 같은 사진
    hold = set(l for l in open(A.holdout).read().split('\n') if l) if os.path.exists(A.holdout) else None
    for mf in sorted(glob.glob(f'{PC}/dot-candidates/dot-arms-*/*.json')):
        d = json.load(open(mf)); items = d['items'] if isinstance(d, dict) and 'items' in d else (d if isinstance(d, list) else d.get('candidates', []))
        for it in items:
            p = os.path.join(os.path.dirname(mf), it['file'])
            held = p in hold if hold is not None else int(hashlib.sha1(os.path.basename(p).encode()).hexdigest(), 16) % 4 == 0
            if os.path.exists(p): L.append((p, 'arm_tat' if it.get('has_tattoo') else 'arm', 'eval' if (held or it.get('has_tattoo')) else 'train'))
    L += [(p, 'hand', 'train') for p in sorted(glob.glob(f'{PC}/collections/hands-2026-10-26/*/*.jpg'))]
    # 직접 모은 사진(tools/collect: 커먼즈 등, 라이선스 기록 있음). 10장에 1장은 시험 전용
    web = sorted(glob.glob(f'{A.collect}/*/images/*'))
    L += [(p, 'web', 'eval' if int(hashlib.sha1(os.path.basename(p).encode()).hexdigest(), 16) % 10 == 0 else 'train') for p in web]
    return L

D = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
G = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').eval()
SP = Sam2Processor.from_pretrained('facebook/sam2.1-hiera-tiny')
SM = Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny').eval()
VP = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224')
VM = AutoModel.from_pretrained('google/siglip2-base-patch16-224').eval()
LAB = ['a tattoo on skin', 'bare skin', 'cream or lotion smeared on skin', 'a mole, freckles or veins on skin', 'hair on skin', 'clothing or jewelry']
def feat(o):
    o = o.pooler_output if hasattr(o, 'pooler_output') else o
    return o / o.norm(dim=-1, keepdim=True)
with torch.no_grad():
    TE = feat(VM.get_text_features(**VP(text=LAB, padding='max_length', max_length=64, return_tensors='pt')))

def ink_rule(rgb, region, area):
    """area 안 잉크 정도(0~1): 블랙햇(주변보다 어두운 가는 획) + 피부 가운데 색(region 바로 바깥 띠)과의 Lab 거리(색 잉크·채운 면)"""
    h, w = region.shape; short = min(h, w)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    k = max(7, int(short * 0.03)) | 1
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # 피부 기준색: 영역 바로 바깥 띠(영역을 키운 것 − 영역)의 가운데값
    ring = cv2.dilate(region.astype(np.uint8), np.ones((k * 2 + 1, k * 2 + 1), np.uint8)).astype(bool) & ~region
    ref = lab[ring] if ring.sum() > 50 else lab[region]
    med = np.median(ref, 0)
    a = lab[..., 1] - 128, lab[..., 2] - 128
    dc = np.hypot(a[0] - (med[1] - 128), a[1] - (med[2] - 128))  # 색도 차이
    dl = (med[0] - lab[..., 0]) * 100 / 255  # 피부보다 어두운 정도
    sm = lambda x, a, b: np.clip((x - a) / (b - a), 0, 1)
    v = np.maximum(sm(bh, 6, 18), np.maximum(sm(dc, 14, 30), sm(dl, 18, 35)))
    return v * area

def label(p):
    im = Image.open(p).convert('RGB'); s = min(1, 768 / max(im.size)); im = im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))), Image.BILINEAR)
    W, H = im.size; rgb = np.asarray(im)
    with torch.no_grad():
        x = D(images=im, text='a tattoo.', return_tensors='pt', size={'shortest_edge': 512, 'longest_edge': 768}); o = G(**x)
        r = D.post_process_grounded_object_detection(o, x.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[(H, W)])[0]
    boxes = [[float(v) for v in b] for b in r['boxes'].tolist()]; ds = [float(v) for v in r['scores'].tolist()]
    vt = []
    if boxes:
        crops = []
        for x0, y0, x1, y1 in boxes:
            pw, ph = (x1 - x0) * 0.1, (y1 - y0) * 0.1
            crops.append(im.crop((max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph))))
        with torch.no_grad():
            pr = torch.softmax(feat(VM.get_image_features(**VP(images=crops, return_tensors='pt'))) @ TE.T * 100, -1).numpy()
        vt = [float(v) for v in pr[:, 0]]
    ink = np.zeros((H, W), np.float32); wgt = np.ones((H, W), np.float32)
    keep = [i for i, v in enumerate(vt) if v >= 0.5]
    unsure = [i for i, v in enumerate(vt) if 0.1 <= v < 0.5]
    for i in unsure:  # 애매한 상자는 배우지 않음
        x0, y0, x1, y1 = map(int, boxes[i]); wgt[y0:y1 + 1, x0:x1 + 1] = 0
    regions = []
    if keep:
        with torch.no_grad():
            si = SP(images=im, input_boxes=[[boxes[i] for i in keep]], return_tensors='pt'); so = SM(**si, multimask_output=False)
        ms = SP.post_process_masks(so.pred_masks, si['original_sizes'])[0][:, 0].numpy() > 0.5
        for j, i in enumerate(keep):
            x0, y0, x1, y1 = boxes[i]; bw, bh = x1 - x0, y1 - y0
            bx = np.zeros((H, W), bool)
            bx[max(0, int(y0 - bh * 0.04)):int(y1 + bh * 0.04) + 1, max(0, int(x0 - bw * 0.04)):int(x1 + bw * 0.04) + 1] = True
            m = ms[j] & bx
            full = m.sum() > 0.8 * bx.sum()  # SAM 이 상자 전체(팔 전체)를 잡으면 상자로
            reg = bx if full or m.sum() < 30 else cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & bx
            if reg is bx and bx.mean() > 0.3:  # 슬리브처럼 큰 상자: 바깥 띠가 옷·배경이라 피부색 기준이 없음 → 배우지 않음
                wgt[bx] = 0; regions.append({'box': i, 'sam_full': True, 'area': float(bx.mean()), 'skipped': True}); continue
            regions.append({'box': i, 'sam_full': bool(full), 'area': float(reg.mean())})
            v = ink_rule(rgb, reg, bx)
            ink = np.maximum(ink, v * reg)
            # 상자 안 SAM 영역 밖: 규칙도 맨살이면 '잉크 아님', 규칙은 잉크인데 SAM 이 놓친 곳(작은 딸기·발자국 등)은 모름
            out = bx & ~reg
            wgt[out & (v >= 0.1)] = 0
            wgt[reg] = 1
    return im, ink, wgt, {'boxes': boxes, 'dino': ds, 'siglip_tat': vt, 'regions': regions}

if __name__ == '__main__':
    L = sources(); done = 0; t0 = time.time()
    print(f'사진 {len(L)}장', flush=True)
    for p, kind, split in L:
        key = hashlib.sha1(p.encode()).hexdigest()[:12]
        if os.path.exists(f'{OUT}/{key}.json'): continue
        try:
            im, ink, wgt, meta = label(p)
        except Exception as e:
            print('실패', p, e, flush=True); json.dump({'src': p, 'error': str(e)}, open(f'{OUT}/{key}.json', 'w')); continue
        im.save(f'{OUT}/{key}.jpg', quality=92)
        Image.fromarray((ink * 255).astype(np.uint8)).save(f'{OUT}/{key}_ink.png')
        Image.fromarray((wgt * 255).astype(np.uint8)).save(f'{OUT}/{key}_w.png')
        meta.update({'src': p, 'kind': kind, 'split': split})
        json.dump(meta, open(f'{OUT}/{key}.json', 'w'), ensure_ascii=False)  # json 을 마지막에(끝난 표시)
        done += 1
        if done % 10 == 0: print(f'{done}장 ({(time.time() - t0) / done:.1f}초/장)', flush=True)
    print('끝', flush=True)
