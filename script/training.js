import * as ort from "/dist/ort.training.wasm.min.js";
import { preprocessImage, loadJSON } from "./infer.js";

console.log(tf);

ort.env.wasm.wasmPaths = "/dist/";

console.log(ort);

var config = await loadJSON("/script/config.json");
var pre = await loadJSON("/script/preprocessor_config.json");
var numEpochs = 2;

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

function createTargetTensor(new_class) {
  const index = config.label2id[new_class];
  const shape = [1, 1000];
  const low_value = -3;

  const size = shape.reduce((a, b) => a * b);
  let data = new Float32Array(size).fill(low_value);
  data[index] = -low_value;

  const target_tensor = new ort.Tensor("float32", data, shape);

  return target_tensor;
}

async function preprocessImageTraining(base64Data, pre) {
  const numImages = 10;
  const images = [];
  const inputSize = { width: pre.size.width, height: pre.size.height };

  // Assuming preprocessImage(base64Data) gives you an ort.Tensor with shape [1, 3, 224, 224]
  // Let's create a tensor from this data for augmentation
  let image = await preprocessImage(base64Data);

  const imageData = await image.getData();
  console.log(imageData);

  for (let i = 0; i < numImages; i++) {
    // Clone the original tensor for augmentation
    let imageTensor = tf.tensor(imageData, [1,3,224,224]);
    console.log(imageTensor);
    // Perform random transformations
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = inputSize.width;
    canvas.height = inputSize.height;

    // Convert tensor to canvas to manipulate
    const pixels = await tf.browser.toPixels(imageTensor);
    console.log(pixels);
    ctx.putImageData(pixels, 0, 0);

    // Random transformations
    if (Math.random() > 0.5) ctx.scale(-1, 1); // Random horizontal flip
    if (Math.random() > 0.5) ctx.scale(1, -1); // Random vertical flip

    // Random rotation
    const rotations = [0, 90, 180, 270];
    const angle = rotations[Math.floor(Math.random() * rotations.length)];
    ctx.rotate((angle * Math.PI) / 180);

    // Convert canvas back to tensor
    imageTensor = tf.browser.fromPixelsAsync(canvas);

    // Transpose image tensor to [3, 224, 224]
    imageTensor = tf.transpose(imageTensor, [2, 0, 1]);
    imageTensor = tf.expandDims(imageTensor, 0);

    // Collect the image tensor
    images.push(imageTensor);
  }

  return images;
}

async function runTrainingEpoch(session, images, epoch, target_tensor) {
  let batchNum = 0;
  const epochStartTime = 0;
  // Obtain loss node name from session
  const lossNodeName = session.trainingOutputNames[0];

  console.log(
    `TRAINING | Epoch ${epoch + 1} / ${numEpochs} | Starting Training ... `
  );

  for (let image of images) {
    // create input
    const feeds = {
      pixel_values: image,
      target: target_tensor,
    };

    const results = await session.runTrainStep(feeds);

    const loss = results[lossNodeName].data;
    console.log(`LOSS ${loss}`);

    await session.runOptimizerStep();
    await session.lazyResetGrad();
  }

  const buffer = session.getContiguousParameters();
  console.log("CONTIGUOUS PARAMETERS", buffer);
}

export async function train(base64Data, new_class) {
  try {
    // Load Training Session
    const session = await loadTrainingSession();
    console.log("Training session loaded");

    // Preprocess input image
    let images = await preprocessImageTraining(base64Data, pre);
    console.log(images);
    console.log(images.shape);
    // Create target tensor
    const target_tensor = createTargetTensor(new_class);

    const startTrainingTime = Date.now();
    let accumulatedLoss = 0;
    let testAcc = 0;

    console.log("Training started");
    // Run training loop
    for (let epoch = 0; epoch < numEpochs; epoch++) {
      await runTrainingEpoch(session, images, epoch, target_tensor);
      // testAcc = await runTestingEpoch(session,images,epoch);

      //const loss = results[lossNodeName].data;
      //// Accumulate loss for evaluation purposes (if needed)
      //accumulatedLoss += parseFloat(loss);
    }

    const trainingTime = Math.abs(startTrainingTime - Date.now());
    console.log(`Training completed in ${trainingTime} milliseconds.`);
  } catch (err) {
    console.error("Error during training:", err);
    throw err; // Propagate the error for higher-level handling
  }
}
