import * as ort from "/dist/ort.training.wasm.min.js";

console.log(ort);

ort.env.wasm.wasmPaths = "/dist/";

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

export async function train() {

  const session = await loadTrainingSession();


}
