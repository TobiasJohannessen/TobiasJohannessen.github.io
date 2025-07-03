var path = window.location.pathname;

const Folders = ["lineq", "eigenvalue", "leastsquares", "splines", "ODE", "quadrature", "montecarlo", "root_finding", "minimum", "neuralnetwork"]
const Titles = ["Linear Equations", "Eigenvalue Decomposition \n (EVD)", "Ordinary Least Squares (OLS)", "Splines", "Ordinary Differential Equations \n (ODE)", "Quadratures", "Monte Carlo Integration", "Root Finding", "Minimization", "Artificial Neural Networks \n (ANN)"];
// Set the title of the page based on the file name
var title =  path.split('/').pop().replace('.html', '');
// Check if the file name matches any of the folders
for (let i = 0; i < Folders.length; i++) {
    if (title.includes(Folders[i])) {
        title = Titles[i];
        break;
    }
}

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




//Generate a header div with a link to the main page
//<div class="header">
//<div id="header">
//</div>
//</div>

var headerDiv = document.createElement('div');
headerDiv.setAttribute('class', 'header');
document.body.appendChild(headerDiv);

var header = document.createElement('div');
header.setAttribute('id', 'header');
headerDiv.appendChild(header);


var header = document.createElement('a');
header.setAttribute('href', '../homework.html');
header.innerHTML = 'Back to Homework';
header.style.position = 'relative';
document.getElementById('header').appendChild(header);



// Remove display of arrows if the current page is not a week page:

if (currentNumber === -1) {
    leftArrow.style.display = 'none';
    rightArrow.style.display = 'none';
}


// Logic for image integration and modal display
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





