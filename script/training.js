import * as ort from "/dist/ort.training.wasm.min.js";

console.log(ort);
// ort.env.wasm.trace = true;
//
// console.log(ort);//
// ort.env.wasm.numThreads = 4;
ort.env.wasm.wasmPaths = "/dist/";

//import { InferenceSession } from "onnxruntime-web";
//import ortWasmThreaded from "../public/ort-training-wasm-simd-threaded.mjs";

//ort.env.wasm.wasmPaths["ort-training-wasm-simd.wasm"] = "../public/ort-training-wasm-simd.wasm";
//ort.env.wasm.wasmPaths["ort-wasm-simd.wasm."] = "../public/ort.training.wasm";

async function loadInferenceSession() {
  console.log("Trying to load inference session");

  try {
    const session = new InferenceSession(
      "./inference_artifacts/initial_model.onnx",
    );
    console.log("Inference session loaded");
  } catch (err) {
    console.log("Error loading inference session", err);
    throw err;
  }
}

async function loadTrainingSession() {
  console.log("Trying to load Training Session");

  const train = "/artifacts/training_model.onnx";
  const eval_ = "/artifacts/eval_model.onnx";
  const optimizer = "/artifacts/optimizer_model.onnx";
  const checkpoint = "/artifacts/checkpoint";

  try {
    const createOptions = {
      checkpointState: checkpoint,
      trainModel: train,
      evalModel: eval_,
      optimizerModel: optimizer,
    };
    const session = await ort.TrainingSession.create(createOptions);
    console.log("Training session loaded");
    console.log(session);
    return session;
  } catch (err) {
    console.log("Error loading the training session:", err);
    throw err;
  }
}

const session = await loadTrainingSession();

//async function infer() {
//    const inference = await loadInferenceSession();
//}
//
//async function train() {
//    const session = await loadTrainingSession();
//
//    let numEpochs = 4:
//
//    for (let epoch=0; epoch<numEpochs; epoch++) {
//        iter += await runTrainingEpoch(session)
//    }
//}
