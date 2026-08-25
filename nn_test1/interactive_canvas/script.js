
// SETTINGS
const modelPath = "./model.onnx"
const modelWeightsPath = "./model.onnx.data"

const updateInterval = 333;  // ms

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear");
const chartCtx = document.getElementById("chart");
console.log(Chart.version);


// DRAWING SETTINGS
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";


// DRAWING
let drawing = false;

let imageChanged = false;
let canvasClear = true;

function getMousePosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

canvas.addEventListener("mousedown", event => {
  drawing = true;
  const pos = getMousePosition(event);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener("mousemove", event => {
    if (!drawing) return;

    const pos = getMousePosition(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    imageChanged = true;
    canvasClear = false;
});

canvas.addEventListener("mouseup", () => {drawing = false});

canvas.addEventListener("mouseleave", () => {drawing = false});


// CLEAR CANVAS
clearButton.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  imageChanged = true;
  canvasClear = true;
});


// IMAGE PROCESSING
function downsample(canvas) {
  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCanvasCtx = smallCanvas.getContext("2d");

  smallCanvasCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = smallCanvasCtx.getImageData(0, 0, 28, 28);
  return imageData
}

function imageToTensor(imageData) {
  const imageVec = new Float32Array(28*28);
  for (let i=0; i < 28*28; i++) {

    // RGBA to grayscale, use first channel (red)
    imageVec[i] = imageData.data[i * 4] / 255;
  }
  const imageTensor = new ort.Tensor("float32", imageVec, [1, 1, 28, 28]);
  return imageTensor
}


// ANALYSIS
function argmax(array) {
  let maxI = 0;
  for (let i=1; i < array.length; i++) {
    if (array[i] > array[maxI]) {maxI = i}
  }
  return maxI;
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}


// LOAD MODEL
let session;

async function loadModel() {

  const modelResponse = await fetch(modelPath);
  console.log("model:", modelResponse.status, modelResponse.ok);
  const modelData = await modelResponse.arrayBuffer();

  const modelWeightsResponse = await fetch(modelWeightsPath);
  console.log("model data:", modelWeightsResponse.status, modelWeightsResponse.ok);
  const externalData = new Uint8Array(
      await modelWeightsResponse.arrayBuffer()
  );

  session = await ort.InferenceSession.create(
    modelData, {
      executionProviders: ["wasm"],
      externalData: [{
        path: modelWeightsPath,
        data: externalData
      } ]
    }
  );
}


// INFERENCE
async function predict(canvas) {
  const imageData = downsample(canvas);
  const imageTensor = imageToTensor(imageData);

  const sessionResults = await session.run({x: imageTensor});
  const output = sessionResults.linear_1.data;

  const probsVector = softmax(output);
  const pred = argmax(probsVector);

  return {
    probabilities: probsVector,
    prediction: pred,
  }
}


// CHART
Chart.defaults.color = "#eeeeee"

const chart = new Chart(chartCtx, {
  type: "bar",
  data: {
    labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    datasets: [{
      data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      backgroundColor: '#9BD0F5'
    }]
  },
  options: {
    indexAxis: "y",

    responsive: true,
    maintainAspectRatio: false,

    scales: {
      x: {title: {display: true, text: "Probability"}, min: 0, max: 1.0, } },
    plugins: {
      legend: {display: false},
      title: {display: true, text:"Prediction:"} }
  },
});

async function main() {
  await loadModel();

  setInterval(async () => {
    if (!imageChanged) return;

    const prediction = await predict(canvas);
    if (!canvasClear) {
      chart.data.datasets[0].data = prediction.probabilities;
      chart.options.plugins.title.text = "Prediction: ".concat(prediction.prediction);
    }
    else {
      chart.data.datasets[0].data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      chart.options.plugins.title.text = "Prediction:";
    }

    chart.update();

    imageChanged = false;
  },

  updateInterval);
}

main();