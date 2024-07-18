import * as ort from "/dist/ort.training.wasm.min.js";

// Set up wasm paths
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

let trainingSession = null;
let images = null;
let target_tensor = null;
const numEpochs = 1;

async function loadTrainingSession() {
  console.log("Trying to load Training Session");

  const train = "/artifacts/training_model.onnx";
  const eval_ = "/artifacts/eval_model.onnx";
  const optimizer = "/artifacts/optimizer_model.onnx";
  const checkpoint = "/artifacts/checkpoint";

  const createOptions = {
    checkpointState: checkpoint,
    trainModel: train,
    evalModel: eval_,
    optimizerModel: optimizer,
  };

  try {
    trainingSession = await ort.TrainingSession.create(createOptions);
    console.log("Training session loaded");
  } catch (err) {
    console.error("Error loading the training session:", err);
    throw err;
  }
}

async function runTrainingEpoch(images, epoch, target_tensor) {
  const epochStartTime = Date.now();
  const lossNodeName = trainingSession.handler.outputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${numEpochs} | Starting Training ... `
  );

  for (let image of images) {
    // create input
    const feeds = {
      pixel_values: image,
      target: target_tensor,
    };

    const results = await trainingSession.runTrainStep(feeds);

    const loss = results[lossNodeName].data;
    console.log(`LOSS: ${loss}`);

    await trainingSession.runOptimizerStep();
    await trainingSession.lazyResetGrad();
  }

  const epochTime = Date.now() - epochStartTime;
  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}

// Worker code for message handling
self.addEventListener("message", async (event) => {

  let data = event.data;

  if (!trainingSession) {
    await loadTrainingSession();
  }

  images = data.images.map(img => new ort.Tensor(img.type, img.data, img.dims));
  target_tensor = new ort.Tensor(data.target_tensor.type, data.target_tensor.data, data.target_tensor.dims);

  const startTrainingTime = Date.now();
  console.log("Training started");

  // Run training loop
  for (let epoch = 0; epoch < numEpochs; epoch++) {
    await runTrainingEpoch(images, epoch, target_tensor);
  }

  const trainingTime = Date.now() - startTrainingTime;

  self.postMessage({ status: "Training completed", trainingTime });

  await trainingSession.release();
});

self.onerror = function (error) {
  console.error("Worker error:", error);
};
