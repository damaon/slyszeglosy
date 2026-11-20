// Runs the phonemizer (wymowa, through Pyodide) and the models (ONNX Runtime Web)
// off the main thread. Messages in: {type: "load"} and {type: "speak", id, text,
// model, voice, speed}; model is "v1" or "v2", and V2 is downloaded the first time it is asked for. Messages out: status, ready, sentence (with its audio, as soon as
// it is made), done (with the whole clip), error (plain: true when the message is
// meant for the reader as it is).

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize;

// The two models share one phonemizer: build_assets.sh checks that they use the same symbols.
const MODELS = {
  v1: { file: "assets/model.onnx", config: "assets/config.json", label: "Ładowanie modelu" },
  v2: { file: "assets/model-v2.onnx", config: "assets/config-v2.json", label: "Ładowanie głosu V2" },
};
const loaded = {};  // model name -> promise of {session, config}

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "pl-4";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    for (const name of await caches.keys()) {             // files of an older version are not needed again
      if (name.startsWith("slyszeglosy-") && name !== "slyszeglosy-" + VERSION) await caches.delete(name);
    }
    store = await caches.open("slyszeglosy-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("Ładowanie fonetyzera");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  // wymowa (rules, no data files) and tiny_voca/text.py as one module: the same tokens as in training.
  const source = await download("assets/phonemizer.zip", "Ładowanie fonetyzera");
  status("Uruchamianie fonetyzera");
  pyodide.unpackArchive(source.buffer, "zip", { extractDir: "/phonemizer" });
  pyodide.runPython(`
import json, sys
sys.path.insert(0, "/phonemizer")
import wymowa
import slyszeglosy_text as text

def phonemize_json(sentence):
    try:
        result = text.phonemize(sentence)
    except text.NothingToPronounce:
        # Nothing to pronounce (another script, a face, signs only) is no failure: "null", and the sentence is skipped.
        return "null"
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress), "text": wymowa.transcribe(sentence)},
                      ensure_ascii=False)

phonemize_json("Rozgrzewka.")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function startModel(name) {
  const { file, config: configFile, label } = MODELS[name];
  const config = await (await fetch(configFile + "?v=" + VERSION)).json();
  const bytes = await download(file, label);
  status("Uruchamianie modelu");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return { session, config };
}

// A model that failed to load is tried again on the next request.
function loadModel(name) {
  if (!MODELS[name]) throw new Error("unknown model " + name);
  if (!loaded[name]) loaded[name] = startModel(name).catch((error) => { delete loaded[name]; throw error; });
  return loaded[name];
}

// A full stop after a title, a common abbreviation or an initial does not end a sentence.
const ABBREVIATIONS = /(?:(?<![\p{L}\p{N}])(?:dr|prof|mgr|inż|hab|św|ks|np|m\.in|ul|al|pl|os|im|godz|tzw|ok|nr|ds|gen|kpt|płk|mjr|tel|str|woj|pow|gm|pt|por|zob|tj|tzn|tys|ws|wg)|(?<![\p{L}\p{N}])[A-ZĄĆĘŁŃÓŚŹŻ])\.$/iu;

// Sentences, without breaking after titles, initials or inside numbers.
// tiny_voca.infer.sentences in the Python package splits the same way; keep the two alike.
function sentences(text) {
  const parts = [];
  let current = "";
  // A line break is kept for the normalizer: a verse may start with the word "I" ("Częstochowy / I w Ostrej").
  text = text.replace(/\s+/g, (space) => (space.includes("\n") ? "\n" : " ")).trim();
  for (const piece of text.split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-ZĄĆĘŁŃÓŚŹŻ0-9])/u)) {
    current = current ? current + " " + piece : piece;
    if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// A full stop is added when the text ends without a mark: before a final mark the voice lengthens the last
// syllable and lets the sound fall, without one the audio breaks off. A face or another sign after the last word
// is no mark ("see you :)"), a comma or a colon is one. tiny_voca.infer.finished does the same; keep the two alike.
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : body + ".";
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut at commas, then by words.
// A sentence with nothing to pronounce gives no utterance.
function fit(sentence, config) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  if (result.ids.length <= config.model.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:—])\s+/);
  if (pieces.length > 1) return pieces.flatMap((piece) => fit(piece, config));
  const words = sentence.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(" "), config), ...fit(words.slice(half).join(" "), config)];
}

async function synthesize(session, phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  return output.audio.data;
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// Shown when no sentence of a text could be read. tiny_voca.infer.speak_text says the same.
const ENGLISH_ONLY = "słyszę głosy czyta na razie tylko po polsku.";

async function speak({ id, text, model, voice, speed }) {
  const { session, config } = await loadModel(model);
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index], config);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = await synthesize(session, phones, voice, speed);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: ENGLISH_ONLY, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const [, { config }] = await Promise.all([loadPhonemizer(), loadModel("v1")]);
      postMessage({ type: "ready", sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
