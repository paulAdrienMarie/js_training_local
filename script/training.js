import * as ort from "/dist/ort.training.wasm.min.js";
import { preprocessImage, loadJSON, toTensorAndResize } from "./infer.js";

// Set up wasm paths 
ort.env.wasm.wasmPaths = "/dist/";
ort.env.wasm.numThreads = 1;

// Load configuration files
const config = await loadJSON("/script/config.json");
const pre = await loadJSON("/script/preprocessor_config.json");
const numEpochs = 2;
let trainingSession = null;

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

    trainingSession = await ort.TrainingSession.create(createOptions);
    console.log("Training session loaded");
  } catch (err) {
    console.error("Error loading the training session:", err);
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

  return new ort.Tensor("float32", data, shape);
}

function augmentImage(imageTensor) {
  // Add a batch dimension, imageTensor shape is: [224, 224, 3]
  let augmentedImage = tf.expandDims(imageTensor, 0);
  
  // augmentedImage shape is: [1, 224, 224, 3]
  // Random horizontal flip
  if (Math.random() > 0.5) {
    augmentedImage = tf.image.flipLeftRight(augmentedImage);
  }


  // Random rotation (0, 90, 180, 270 degrees)
  const rotations = [0, 90, 180, 270];
  const angle = rotations[Math.floor(Math.random() * rotations.length)];
  augmentedImage = tf.image.rotateWithOffset(augmentedImage, angle / 90);

  return augmentedImage; // shape: [1, 224, 224, 3]
}

async function preprocessImageTraining(base64Data, pre) {
  const numImages = 10;
  const images = [];
  const inputSize = {
    width: pre.size.width,
    height: pre.size.height,
  };

  // Preprocess the image, image shape is [1, 3, 224, 224]
  let image = await toTensorAndResize(base64Data);

  // Get the pixel values of the image
  const imageData_ = await image.getData();

  for (let i = 0; i < numImages; i++) {
    // Clone the original tensor for augmentation, imageTensor shape is [3, 224, 224]
    let imageTensor = tf.tensor(imageData_, [3, 224, 224], "float32");

    // Transpose image tensor to [224, 224, 3] for augmentation
    imageTensor = tf.transpose(imageTensor, [1, 2, 0]);

    // Apply augmentations, shape is [1, 224, 224, 3]
    let augmentedTensor = augmentImage(imageTensor);

    // Transpose image tensor back to [3, 224, 224]
    augmentedTensor = tf.transpose(augmentedTensor, [0, 3, 1, 2]);

    let data_ = await augmentedTensor.data();
    let shape = augmentedTensor.shape;

    augmentedTensor = new ort.Tensor("float32", data_, shape);

    augmentedTensor = await preprocessImage(augmentedTensor);

    // Collect the augmented tensor
    images.push(augmentedTensor);
  }

  return images;
}

async function runTrainingEpoch(session, images, epoch, target_tensor) {
  const epochStartTime = Date.now();
  const lossNodeName = session.trainingOutputNames[0];

  console.log(`TRAINING | Epoch ${epoch + 1} / ${numEpochs} | Starting Training ... `);

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

  const epochTime = Date.now() - epochStartTime;
  console.log(`Epoch ${epoch + 1} completed in ${epochTime} milliseconds.`);
}

export async function train(base64Data, new_class) {
  try {
    // Load Training Session
    if (!trainingSession) {
      console.log("Training session not loaded yet, waiting...");
      await loadTrainingSession();
    }
    
    // Preprocess input image - perform data augmentation
    let images = await preprocessImageTraining(base64Data, pre);

    // Create target tensor
    const target_tensor = createTargetTensor(new_class);

    const startTrainingTime = Date.now();
    console.log("Training started");

    // Run training loop
    for (let epoch = 0; epoch < numEpochs; epoch++) {
      await runTrainingEpoch(trainingSession, images, epoch, target_tensor);
    }

    const trainingTime = Date.now() - startTrainingTime;
    console.log(`Training completed in ${trainingTime} milliseconds.`);

    await trainingSession.release();

  } catch (err) {
    console.error("Error during training:", err);
    throw err; // Propagate the error for higher-level handling
  }
}
