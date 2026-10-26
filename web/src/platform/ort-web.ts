// 웹용 모델 실행기(core/runner.ts 의 ModelRunner): onnxruntime-web(WASM). 처음 쓸 때 한 번 불러온다.
// 앱(React Native · Expo)에서는 onnxruntime-react-native 로 같은 모양(run({data, dims}) → {data, dims})의 실행기를 만든다(docs/models.md).
import type { ModelRunner, Tensor } from '../core/runner.ts';

const ORT_VERSION = '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

type Session = { inputNames: string[]; outputNames: string[]; run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array; dims: number[] }>> };
type Ort = { env: { wasm: { wasmPaths: string } }; InferenceSession: { create(url: string, opts: { executionProviders: string[] }): Promise<Session> }; Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown };
let ortP: Promise<Ort> | null = null;

/** 모델 주소(문서 기준 상대 경로) → 실행기. 불러오기에 실패하면 다음 run 때 다시 시도한다 */
export function ortWebRunner(modelUrl: string): ModelRunner {
  let sessP: Promise<{ ort: Ort; session: Session }> | null = null;
  const load = () => {
    sessP ??= (async () => {
      ortP ??= import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`).then((m) => {
        const ort = m as unknown as Ort;
        ort.env.wasm.wasmPaths = ORT_BASE;
        return ort;
      });
      const ort = await ortP;
      const url = new URL(modelUrl, document.baseURI).href;
      return { ort, session: await ort.InferenceSession.create(url, { executionProviders: ['wasm'] }) };
    })();
    sessP.catch(() => {
      sessP = null;
    });
    return sessP;
  };
  return {
    async run(input: Tensor): Promise<Tensor> {
      const { ort, session } = await load();
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input.data, input.dims) });
      const o = out[session.outputNames[0]];
      return { data: o.data, dims: o.dims };
    },
  };
}
