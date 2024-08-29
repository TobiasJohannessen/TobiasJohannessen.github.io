const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
var N = 50;
var fps = parseInt(fpsSlider.value);
var generation = 0;
var lonelinessCount = 1;
var reproductionCount = 3;
var overpopulationCount = 4;
const generationCounter = document.getElementById('generationCounter');
const cellSize = canvas.width / N;
let boolGrid = Array.from({ length: N }, () => Array(N).fill());
let isSimulationRunning = false;
let animationFrameId;


function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            ctx.fillStyle = boolGrid[i][j] ? 'black' : 'white';
            ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
            ctx.strokeRect(i * cellSize, j * cellSize, cellSize, cellSize);
        }
    }
}

function evaluateCell(x, y) {
    let count = 0;
    if (boolGrid[x][y]) count -= 1;
    for (let i = x - 1; i <= x + 1; i++) {
        for (let j = y - 1; j <= y + 1; j++) {
            if (i >= 0 && i < N && j >= 0 && j < N && boolGrid[i][j]) count += 1;
        }
    }
    
    if (count <= lonelinessCount && boolGrid[x][y]) return false;
    if (count >= overpopulationCount && boolGrid[x][y]) return false;
    if (count === reproductionCount && !boolGrid[x][y]) return true;
    return boolGrid[x][y];
}

function updateCells() {
    let newGrid = Array.from({ length: N }, () => Array(N).fill(false));
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            newGrid[i][j] = evaluateCell(i, j);
        }
    }
    boolGrid = newGrid;
    drawGrid();
    
}

/*Function to toggle cell state on mouse click*/
canvas.addEventListener('click', (e) => {
    /*if (isSimulationRunning) return;*/

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left - 5) / cellSize);
    const y = Math.floor((e.clientY - rect.top - 5) / cellSize );

    boolGrid[x][y] = !boolGrid[x][y];
    drawGrid();
});

function startSimulation() {
    if (!isSimulationRunning) {
        generation = 0;
        generationCounter.textContent = "Generation: 0";
        isSimulationRunning = true;
        startButton.textContent = 'Stop Simulation';
        runSimulation();
        
    } else {
        isSimulationRunning = false;
        startButton.textContent = 'Start Simulation';
        cancelAnimationFrame(animationFrameId);
    }
}

function runSimulation() {
    
    
    if (isSimulationRunning) {
        updateCells();
        animationFrameId = requestAnimationFrame(() => setTimeout(runSimulation, 1000/fps));
        generation += 1;
        generationCounter.textContent = "Generation: " + generation;
    }
    
}


/* Buttons and sliders*/


/*Slider to change the speed of the simulation*/
fpsSlider.addEventListener('input', () => {
    fps = parseInt(fpsSlider.value);
});


/*Start button, which starts the simulation*/
startButton.addEventListener('click', startSimulation);

/*Clear button, which resets the grid and stops the simulation*/
clearButton.addEventListener('click', () => {
    if (isSimulationRunning){
        isSimulationRunning = false;
        startButton.textContent = 'Start Simulation';
        cancelAnimationFrame(animationFrameId)
    };
    boolGrid = Array.from({ length: N }, () => Array(N).fill(false));
    generation = 0;
    generationCounter.textContent = "Generation: 0";
    drawGrid();

});

randomButton.addEventListener('click', () => {
    if (isSimulationRunning) return;
    boolGrid = Array.from({ length: N }, () => Array(N).fill().map(() => Math.random() > 0.9));
    drawGrid();

});

/*Read the inputs of the three parameters for the game of life rules*/
loneliness.addEventListener('input', () => {
    lonelinessCount = parseInt(loneliness.value);
});
reproduction.addEventListener('input', () => {
    reproductionCount = parseInt(reproduction.value);
});
overpopulation.addEventListener('input', () => {
    overpopulationCount = parseInt(overpopulation.value);
});

refreshButton.addEventListener('click', () => {
    loneliness.value = 1;
    lonelinessCount = 1;
    reproduction.value = 3;
    reproductionCount = 3;
    overpopulation.value = 4; 
    overpopulationCount = 4;
});



drawGrid();
