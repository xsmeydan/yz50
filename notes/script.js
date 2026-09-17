// ---------- Chapter nav active-state ----------
const tabs = Array.from(document.querySelectorAll('.chapter-tab'));
const sections = tabs.map(t => document.getElementById(t.dataset.target));
function updateActiveTab(){
  let idx = 0;
  const y = window.scrollY + 140;
  sections.forEach((s, i) => { if (s && s.offsetTop <= y) idx = i; });
  tabs.forEach((t, i) => t.classList.toggle('active', i === idx));
}
window.addEventListener('scroll', updateActiveTab, { passive: true });
updateActiveTab();

// ---------- Sigmoid widget ----------
function sigmoid(z){ return 1 / (1 + Math.exp(-z)); }

const sigPath = document.getElementById('sigmoidPath');
const sigDot = document.getElementById('sigmoidDot');
const zSlider = document.getElementById('zSlider');
const zVal = document.getElementById('zVal');
const zReadout = document.getElementById('zReadout');
const sigReadout = document.getElementById('sigReadout');

function sigX(z){ return 320 + z * 30; }        // z in [-10,10] -> x in [20,620]
function sigY(s){ return 190 - s * 170; }        // s in [0,1] -> y in [190,20]

(function drawSigmoidCurve(){
  let d = '';
  for(let z=-10; z<=10; z+=0.25){
    const x = sigX(z), y = sigY(sigmoid(z));
    d += (z === -10 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
  }
  sigPath.setAttribute('d', d);
})();

function updateSigmoidWidget(){
  const z = parseFloat(zSlider.value);
  const s = sigmoid(z);
  zVal.textContent = z.toFixed(1);
  zReadout.textContent = z.toFixed(1);
  sigReadout.textContent = s.toFixed(2);
  sigDot.setAttribute('cx', sigX(z));
  sigDot.setAttribute('cy', sigY(s));
}
zSlider.addEventListener('input', updateSigmoidWidget);
updateSigmoidWidget();

// ---------- Single neuron widget ----------
const ids = ['a1','a2','a3','w1','w2','w3','biasN'];
const els = {}; ids.forEach(id => els[id] = document.getElementById(id));
const labelMap = { a1:'a1v', a2:'a2v', a3:'a3v', w1:'w1v', w2:'w2v', w3:'w3v', biasN:'bv' };

function updateNeuron(){
  const a1 = parseFloat(els.a1.value), a2 = parseFloat(els.a2.value), a3 = parseFloat(els.a3.value);
  const w1 = parseFloat(els.w1.value), w2 = parseFloat(els.w2.value), w3 = parseFloat(els.w3.value);
  const b = parseFloat(els.biasN.value);
  const z = w1*a1 + w2*a2 + w3*a3 + b;
  const out = sigmoid(z);

  document.getElementById('a1v').textContent = a1.toFixed(2);
  document.getElementById('a2v').textContent = a2.toFixed(2);
  document.getElementById('a3v').textContent = a3.toFixed(2);
  document.getElementById('w1v').textContent = w1.toFixed(2);
  document.getElementById('w2v').textContent = w2.toFixed(2);
  document.getElementById('w3v').textContent = w3.toFixed(2);
  document.getElementById('bv').textContent = b.toFixed(2);

  document.getElementById('neuronZ').textContent = z.toFixed(2);
  document.getElementById('neuronOut').textContent = out.toFixed(2);
  document.getElementById('neuronBar').style.width = (out*100).toFixed(1) + '%';
}
ids.forEach(id => els[id].addEventListener('input', updateNeuron));
updateNeuron();

// ---------- Gradient descent widget ----------
// C(x) = x^4/4 - x^3/3 - 3x^2 + 5   ;  C'(x) = x^3 - x^2 - 6x
function costC(x){ return x**4/4 - x**3/3 - 3*x**2 + 5; }
function costSlope(x){ return x**3 - x**2 - 6*x; }

const GD_XMIN = -3.6, GD_XMAX = 4.6;
let sampleYs = [];
for(let x=GD_XMIN; x<=GD_XMAX; x+=0.05) sampleYs.push(costC(x));
const GD_YMIN = Math.min(...sampleYs), GD_YMAX = Math.max(...sampleYs);

const gdCurve = document.getElementById('gdCurve');
const gdBall = document.getElementById('gdBall');
const lrSlider = document.getElementById('lrSlider');
const lrVal = document.getElementById('lrVal');

function gdSx(x){ return 15 + (x - GD_XMIN) / (GD_XMAX - GD_XMIN) * 610; }
function gdSy(y){ return 235 - (y - GD_YMIN) / (GD_YMAX - GD_YMIN) * 215; }

(function drawCostCurve(){
  let d = '';
  for(let x=GD_XMIN; x<=GD_XMAX; x+=0.05){
    const sx = gdSx(x), sy = gdSy(costC(x));
    d += (x === GD_XMIN ? 'M' : 'L') + sx.toFixed(1) + ',' + sy.toFixed(1) + ' ';
  }
  gdCurve.setAttribute('d', d);
})();

let gdX = -3.2;

function renderBall(){
  gdBall.setAttribute('cx', gdSx(gdX));
  gdBall.setAttribute('cy', gdSy(costC(gdX)));
  document.getElementById('gdX').textContent = gdX.toFixed(2);
  document.getElementById('gdC').textContent = costC(gdX).toFixed(2);
  document.getElementById('gdSlope').textContent = costSlope(gdX).toFixed(2);
}

function gdStep(){
  const lr = parseFloat(lrSlider.value);
  const slope = costSlope(gdX);
  gdX = gdX - lr * slope * 0.15;
  gdX = Math.max(GD_XMIN, Math.min(GD_XMAX, gdX));
  renderBall();
}

lrSlider.addEventListener('input', () => { lrVal.textContent = parseFloat(lrSlider.value).toFixed(2); });
document.getElementById('stepBtn').addEventListener('click', gdStep);
document.getElementById('stepsBtn').addEventListener('click', () => { for(let i=0;i<10;i++) gdStep(); });
document.getElementById('randomBtn').addEventListener('click', () => {
  gdX = GD_XMIN + Math.random() * (GD_XMAX - GD_XMIN);
  renderBall();
});

renderBall();

// ---------- Activation function comparison widget ----------
const actFns = {
  sigmoid: { f: z => 1/(1+Math.exp(-z)), ymin:0, ymax:1, label:'σ(z) = 1 / (1 + e⁻ᶻ)' },
  tanh:    { f: z => Math.tanh(z), ymin:-1, ymax:1, label:'tanh(z)' },
  relu:    { f: z => Math.max(0,z), ymin:-1, ymax:5, label:'ReLU(z) = max(0, z)' },
  leaky:   { f: z => z>0 ? z : 0.15*z, ymin:-1.2, ymax:5, label:'LeakyReLU(z) = max(0.15z, z)' }
};
const actCurve = document.getElementById('actFnCurve');
const actLabel = document.getElementById('actFnLabel');
const actButtons = document.querySelectorAll('#actFnButtons .btn');

function actSx(z){ return 320 + z * 30; }
function actSy(y, ymin, ymax){ return 230 - (y - ymin) / (ymax - ymin) * 210; }

function drawActFn(name){
  const { f, ymin, ymax, label } = actFns[name];
  let d = '';
  for(let z=-10; z<=10; z+=0.2){
    const y = Math.max(ymin, Math.min(ymax, f(z)));
    const x = actSx(z), sy = actSy(y, ymin, ymax);
    d += (z === -10 ? 'M' : 'L') + x.toFixed(1) + ',' + sy.toFixed(1) + ' ';
  }
  actCurve.setAttribute('d', d);
  actLabel.textContent = label;
  actButtons.forEach(b => b.classList.toggle('secondary', b.dataset.f !== name));
}
actButtons.forEach(b => b.addEventListener('click', () => drawActFn(b.dataset.f)));
drawActFn('sigmoid');
