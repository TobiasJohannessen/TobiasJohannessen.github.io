


function createWeekElement(i) {
    const linkContainer = document.createElement("div");
    linkContainer.className = "linkContainer";

    const link = document.createElement("a");
    link.href = `Weeks/Week${i}.html`;

    const img = document.createElement("img");
    const pngSrc = `Weeks/Figures/Thumbnails/Week${i}.png`;
    const gifSrc = `Weeks/Figures/Thumbnails/Week${i}.gif`;

    // Try PNG first, then GIF, log error if both fail
    fetch(pngSrc, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                img.src = pngSrc;
            } else {
                return fetch(gifSrc, { method: 'HEAD' });
            }
        })
        .then(response => {
            if (response && response.ok) {
                img.src = gifSrc;
            } else if (!img.src) {
                console.error(`Both PNG and GIF not found for Week ${i}`);
            }
        })
        .catch(() => console.error(`Error loading images for Week ${i}`));

    const container = document.querySelector(".container");
    link.appendChild(img);
    container.appendChild(link);

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    const overlayText = document.createElement("div");
    overlayText.className = "text";
    overlayText.textContent = `Week ${i}`;
    overlay.appendChild(overlayText);
    link.appendChild(overlay);

    const bottomText = document.createElement("div");
    bottomText.className = "text";
    bottomText.textContent = `Week ${i}`;
    link.appendChild(bottomText);

    linkContainer.appendChild(link);

    return linkContainer;
}


const Subjects = ["Temperature", "Optimisation", "Clustering", "Molecular Graphs", "Regression", "Phase Transitions", 'Neural Networks I', "Neural Networks II"]

function createSubjectElement(i) {
    const linkContainer = document.createElement("div");
    linkContainer.className = "linkContainer";
    
    const link = document.createElement("a");
    const filename = Subjects[i].replace(' ', '');

    link.href = `Subjects/${filename}.html`;

    const img = document.createElement("img");
    const pngSrc = `Subjects/Figures/Thumbnails/${filename}.png`;
    const gifSrc = `Subjects/Figures/Thumbnails/${filename}.gif`;

    fetch(pngSrc, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                img.src = pngSrc;
            } else {
                return fetch(gifSrc, { method: 'HEAD' });
            }
        })
        .then(response => {
            if (response && response.ok) {
                img.src = gifSrc;
            } else if (!img.src) {
                console.error(`Both PNG and GIF not found for Subject ${filename}`);
            }
        })
        .catch(() => console.error(`Error loading images for Subject ${filename}`));

    const container = document.querySelector(".container");
    link.appendChild(img);
    container.appendChild(link);

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    const overlayText = document.createElement("div");
    overlayText.className = "text";
    overlayText.textContent = ` ${Subjects[i]}`;
    overlay.appendChild(overlayText);
    link.appendChild(overlay);

    const bottomText = document.createElement("div");
    bottomText.className = "text";
    bottomText.textContent = `${Subjects[i]}`;
    link.appendChild(bottomText);

    linkContainer.appendChild(link);

    return linkContainer;
}


function createToDoListElement() {
    const linkContainer = document.createElement("div");
    linkContainer.className = "linkContainer";
    
    const link = document.createElement("a");
    link.href = `todolist/todolist.html`;

    const img = document.createElement("img");
    const pngSrc = `todolist/todolist.png`;
    const gifSrc = `todolist/todolist.gif`;

    fetch(pngSrc, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                img.src = pngSrc;
            } else {
                return fetch(gifSrc, { method: 'HEAD' });
            }
        })
        .then(response => {
            if (response && response.ok) {
                img.src = gifSrc;
            } else if (!img.src) {
                console.error(`Both PNG and GIF not found for To Do List`);
            }
        })
        .catch(() => console.error(`Error loading images for To Do List`));

    const container = document.querySelector(".container");
    link.appendChild(img);
    container.appendChild(link);

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    const overlayText = document.createElement("div");
    overlayText.className = "text";
    overlayText.textContent = ` To Do List`;
    overlay.appendChild(overlayText);
    link.appendChild(overlay);

    const bottomText = document.createElement("div");
    bottomText.className = "text";
    bottomText.textContent = `To Do List`;
    link.appendChild(bottomText);

    linkContainer.appendChild(link);

    return linkContainer;
}

let showingWeeks = true;

// Function to toggle between displaying Weeks and Subjects
function toggleWeekSubjects() {
    // Clear the container first
    container.innerHTML = "";

    container.append(createToDoListElement());
    // Toggle the boolean state
    showingWeeks = !showingWeeks;

    if (showingWeeks) {
        // Add Week elements
        for (let i = 1; i <= 14; i++) {
            container.appendChild(createWeekElement(i));
        }

        document.getElementById("toggleSwitch").textContent = "Show Subjects";
    } else {
        // Add Subject elements
        for (let i = 0; i < Subjects.length; i++) {
            container.appendChild(createSubjectElement(i));
        }
        document.getElementById("toggleSwitch").textContent = "Show Weeks";
    }
}



const content = document.querySelector(".content");
const container = document.querySelector(".container")
const switchContainer = document.querySelector(".switchContainer");

function initialize() {

    // Add the toggle button
    const toggleSwitch = document.createElement("button");

    toggleSwitch.className = ".toggleSwitch";
    toggleSwitch.textContent = "Show Subjects";
    toggleSwitch.id = "toggleSwitch";
    toggleSwitch.onclick = toggleWeekSubjects;
    switchContainer.appendChild(toggleSwitch);

    // Add the initial Week elements
    
    for (let i = 1; i <= 14; i++) {
        container.appendChild(createWeekElement(i));
    }
}



initialize();
 

