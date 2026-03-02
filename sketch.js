let network;
let trainingData;
let errorHistory = [];
let training = false;

let lrSlider;
let activationMode = "sigmoid"; // ou "relu"

function setup() {
  let canvas = createCanvas(1100, 650);
  canvas.position(0, 60);

  // SLIDER LEARNING RATE
  lrSlider = createSlider(0.001, 1, 0.1, 0.001);
  lrSlider.position(300, 25);
  lrSlider.style('width', '200px');

  resetNetwork();

  trainingData = [
    { inputs: [0, 0], target: [0] },
    { inputs: [0, 1], target: [1] },
    { inputs: [1, 0], target: [1] },
    { inputs: [1, 1], target: [0] }
  ];
}

function toggleTraining() {
  training = !training;
}

function resetNetwork() {
  network = new NeuralNetwork(2, 3, 1);
  errorHistory = [];
}

function toggleActivation() {
  activationMode = activationMode === "sigmoid" ? "relu" : "sigmoid";
}

function draw() {
  background(15);

  network.learning_rate = lrSlider.value();

  if (training) {
    let data = random(trainingData);
    let error = network.train(data.inputs, data.target);
    errorHistory.push(error);
    if (errorHistory.length > 300) errorHistory.shift();
  }

  network.draw(200, 150);
  drawGraph();
  drawPredictions();
  drawUIInfo();
}

function drawUIInfo() {
  fill(255);
  textSize(14);
  text("Learning Rate: " + nf(lrSlider.value(), 1, 3), 520, 40);
  text("Activation: " + activationMode.toUpperCase(), 750, 40);
}

function drawPredictions() {
  fill(255);
  textSize(14);

  let yBase = 520;

  for (let i = 0; i < trainingData.length; i++) {
    let out = network.predict(trainingData[i].inputs);
    text(
      trainingData[i].inputs + " → " + nf(out[0], 1, 3),
      200,
      yBase + i * 25
    );
  }
}

function drawGraph() {
  push();
  translate(700, 150);

  // EIXOS
  stroke(120);
  line(0, 0, 0, 200);
  line(0, 200, 300, 200);

  fill(200);
  noStroke();
  text("Erro", -40, 10);
  text("Iterações", 100, 230);

  // ESCALA
  for (let i = 0; i <= 5; i++) {
    let y = map(i, 0, 5, 200, 0);
    stroke(50);
    line(0, y, 300, y);
    noStroke();
    fill(150);
    text((i/5).toFixed(1), -30, y + 5);
  }

  // CURVA
  stroke(0, 255, 200);
  noFill();
  beginShape();
  for (let i = 0; i < errorHistory.length; i++) {
    let y = map(errorHistory[i], 0, 1, 200, 0);
    vertex(i, y);
  }
  endShape();

  pop();
}

/* ========================= */
/* ATIVAÇÕES */
/* ========================= */

function activate(x) {
  if (activationMode === "relu") return Math.max(0, x);
  return 1 / (1 + Math.exp(-x));
}

function dActivate(y) {
  if (activationMode === "relu") return y > 0 ? 1 : 0;
  return y * (1 - y);
}

/* ========================= */
/* REDE NEURAL */
/* ========================= */

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.learning_rate = 0.1;

    this.weights_ih = this.randomMatrix(hidden_nodes, input_nodes);
    this.weights_ho = this.randomMatrix(output_nodes, hidden_nodes);

    this.bias_h = this.randomArray(hidden_nodes);
    this.bias_o = this.randomArray(output_nodes);

    this.lastHidden = new Array(hidden_nodes).fill(0);
    this.lastOutput = new Array(output_nodes).fill(0);
  }

  randomMatrix(r, c) {
    return Array.from({ length: r }, () =>
      Array.from({ length: c }, () => random(-1, 1))
    );
  }

  randomArray(s) {
    return Array.from({ length: s }, () => random(-1, 1));
  }

  predict(inputs) {
    let hidden = this.feedHidden(inputs);
    let outputs = this.feedOutput(hidden);
    return outputs;
  }

  train(inputs, targets) {
    let hidden = this.feedHidden(inputs);
    let outputs = this.feedOutput(hidden);

    this.lastHidden = hidden;
    this.lastOutput = outputs;

    let output_errors = outputs.map((o, i) => targets[i] - o);

    // OUTPUT
    for (let i = 0; i < this.output_nodes; i++) {
      let gradient = dActivate(outputs[i]) * output_errors[i] * this.learning_rate;

      for (let j = 0; j < this.hidden_nodes; j++) {
        this.weights_ho[i][j] += gradient * hidden[j];
      }

      this.bias_o[i] += gradient;
    }

    // HIDDEN
    for (let i = 0; i < this.hidden_nodes; i++) {
      let error = 0;
      for (let j = 0; j < this.output_nodes; j++) {
        error += this.weights_ho[j][i] * output_errors[j];
      }

      let gradient = dActivate(hidden[i]) * error * this.learning_rate;

      for (let j = 0; j < this.input_nodes; j++) {
        this.weights_ih[i][j] += gradient * inputs[j];
      }

      this.bias_h[i] += gradient;
    }

    return Math.abs(output_errors[0]);
  }

  feedHidden(inputs) {
    let hidden = [];

    for (let i = 0; i < this.hidden_nodes; i++) {
      let sum = this.bias_h[i];
      for (let j = 0; j < this.input_nodes; j++) {
        sum += this.weights_ih[i][j] * inputs[j];
      }
      hidden[i] = activate(sum);
    }

    return hidden;
  }

  feedOutput(hidden) {
    let outputs = [];

    for (let i = 0; i < this.output_nodes; i++) {
      let sum = this.bias_o[i];
      for (let j = 0; j < this.hidden_nodes; j++) {
        sum += this.weights_ho[i][j] * hidden[j];
      }
      outputs[i] = activate(sum);
    }

    return outputs;
  }

  draw(x, y) {
    let spacing = 100;

    for (let i = 0; i < this.hidden_nodes; i++) {
      for (let j = 0; j < this.input_nodes; j++) {
        let w = this.weights_ih[i][j];
        stroke(w > 0 ? color(0,255,150) : color(255,80,80));
        strokeWeight(constrain(Math.abs(w)*2,1,4));
        line(x, y + j*spacing, x+200, y + i*spacing);
      }
    }

    for (let i = 0; i < this.output_nodes; i++) {
      for (let j = 0; j < this.hidden_nodes; j++) {
        let w = this.weights_ho[i][j];
        stroke(w > 0 ? color(0,255,150) : color(255,80,80));
        strokeWeight(constrain(Math.abs(w)*2,1,4));
        line(x+200, y + j*spacing, x+400, y + i*spacing);
      }
    }

    noStroke();

    for (let i = 0; i < this.input_nodes; i++) {
      fill(0,200,255);
      ellipse(x, y + i*spacing, 50);
    }

    for (let i = 0; i < this.hidden_nodes; i++) {
      fill(0,255*this.lastHidden[i],150);
      ellipse(x+200, y + i*spacing, 50);
    }

    for (let i = 0; i < this.output_nodes; i++) {
      fill(255*this.lastOutput[i],100,200);
      ellipse(x+400, y + i*spacing, 50);

      fill(255);
      textAlign(CENTER);
      text(nf(this.lastOutput[i],1,2),
           x+400, y + i*spacing + 35);
    }
  }
}