const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// DRAWING SETTINGS
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";


let drawing = false;

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
});

canvas.addEventListener("mouseup", () => {drawing = false});

canvas.addEventListener("mouseleave", () => {drawing = false});


document.getElementById("clear").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

