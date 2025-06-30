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

document.addEventListener('DOMContentLoaded', () => {
    const modal = document.createElement('div');
    modal.classList.add('modal');
    modal.innerHTML = `
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
        <div class="caption" id="caption"></div>
        <span class="nav-button prev">&#10094;</span>
        <span class="nav-button next">&#10095;</span>
        <button id="zoomInButton">Zoom In</button>
        <button id="zoomOutButton">Zoom Out</button>
    `;
    document.body.appendChild(modal);

    const modalImg = document.getElementById('modalImage');
    const captionText = document.getElementById('caption');
    const closeModal = modal.querySelector('.close');
    const prevButton = modal.querySelector('.prev');
    const nextButton = modal.querySelector('.next');
    const zoomInButton = document.getElementById('zoomInButton');
    const zoomOutButton = document.getElementById('zoomOutButton');

    const images = Array.from(document.querySelectorAll('img'));
    let currentIndex = 0;
    let zoomFactor = 1;  // Initial zoom factor

    const showImage = (index) => {
        // Ensure wrapping and update currentIndex
        currentIndex = (index + images.length) % images.length;
        modalImg.src = images[currentIndex].src;
        // Reset zoom factor when showing a new image
        zoomFactor = 1;
        modalImg.style.transform = `scale(${zoomFactor})`;
    };

    images.forEach((img, index) => {
        img.addEventListener('click', () => {
            modal.style.display = 'flex';
            showImage(index);
        });
    });

    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    prevButton.addEventListener('click', (e) => {
        e.stopPropagation();
        currentIndex = (currentIndex - 1 + images.length) % images.length; // Wrap index
        showImage(currentIndex);
    });

    nextButton.addEventListener('click', (e) => {
        e.stopPropagation();
        currentIndex = (currentIndex + 1) % images.length; // Wrap index
        showImage(currentIndex);
    });

    zoomInButton.addEventListener('click', () => {
        zoomFactor += 0.1; // Zoom in
        modalImg.style.transform = `scale(${zoomFactor})`;
    });

    zoomOutButton.addEventListener('click', () => {
        zoomFactor = Math.max(0.1, zoomFactor - 0.1); // Zoom out, but don't allow negative scaling
        modalImg.style.transform = `scale(${zoomFactor})`;
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    window.addEventListener('keydown', (e) => {
        if (modal.style.display === 'flex') {
            if (e.key === 'ArrowLeft') {
                e.preventDefault(); // Prevent default browser behavior
                currentIndex = (currentIndex - 1 + images.length) % images.length; // Wrap index
                showImage(currentIndex);
            }
            if (e.key === 'ArrowRight') {
                e.preventDefault(); // Prevent default browser behavior
                currentIndex = (currentIndex + 1) % images.length; // Wrap index
                showImage(currentIndex);
            }
            if (e.key === 'Escape') {
                modal.style.display = 'none';
            }
        }
    });

    // Mouse wheel zoom logic
    modalImg.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.deltaY < 0) {
            // Zoom in on scroll up
            zoomFactor += 0.1;
        } else {
            // Zoom out on scroll down
            zoomFactor = Math.max(0.1, zoomFactor - 0.1);
        }
        modalImg.style.transform = `scale(${zoomFactor})`;
    });
});

(function generateSlides() {
    // Get the title of the current HTML page
    const pageTitle = document.title;

    // Define the folder path
    const folderPath = `./Figures/${pageTitle}`;

    // Number of slides (Adjust as needed or make dynamic)
    const slideCount = 30; // Example: 10 slides

    // Get the container where slides will be appended
    const container = document.getElementById('slides-container');

    // Create and append <img> elements for each slide
    for (let i = 1; i <= slideCount; i++) {
      const img = document.createElement('img');
      img.src = `${folderPath}/Slide${i}.JPG` // Assumes slides are jpg
        
      img.alt = `Slide ${i}`;
      img.style.display = 'block'; // Optional: ensure each slide is on a new line
      container.appendChild(img);
    }
  })();
