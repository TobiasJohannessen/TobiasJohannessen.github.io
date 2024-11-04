var path = window.location.pathname;

// Set the title of the page based on the file name
title =  path.split('/').pop().replace('.html', '');
document.title = title;

//Link a stylesheet to the page
var head = document.getElementsByTagName('head')[0];
var link = document.createElement('link');
link.rel = 'stylesheet';
link.type = 'text/css';
link.href = '../style.css';
head.appendChild(link);



// Extract the file name with extension
var fileName = path.split('/').pop();

// Extract the current number from the file name (assuming it's at the end)
const regex = /Week(\d+).html/; // Match 'Week' followed by a number
const match = fileName.match(regex);
const currentNumber = match ? parseInt(match[1]) : -1;


const relevantWeeksData = {
    'Temperature': [2, 5, 7],
    'Optimisation': [3, 5, 6, 8, 9, 10],
    'Clustering': [4, 7, 11],
    'MolecularGraphs': [4, 12],
    'Regression': [6, 10],
    'PhaseTransitions': [5, 7],
    'NeuralNetworksI': [8, 9, 12, 13],
    'NeuralNetworksII': [10, 11, 13]
};

// Function to add links to relevant weeks
function linkRelevantWeeks(title) {
    const relevantWeeks = relevantWeeksData[title];
    if (relevantWeeks) {
        const weekLinks = document.getElementById('weekLinks');
        for (let week of relevantWeeks) {
            const a = document.createElement('a');
            a.setAttribute('href', `../Weeks/Week${week}.html`);
            a.textContent = `Week ${week}`;
            a.style.margin = '5px'; // Add some spacing between the links
            weekLinks.appendChild(a);
        }
    }
}

// Call the function with the page title to generate relevant links
linkRelevantWeeks(title);
