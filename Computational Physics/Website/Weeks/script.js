var path = window.location.pathname;

// Set the title of the page based on the file name
title =  path.split('/').pop().replace('.html', '');
title = title.replace('Week', 'Week ');
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



//Generate a header div with a link to the main page
//<div class="header">
//<div id="header">
//</div>
// <div id="weekLinks"></div>
//</div>
//</div>

var headerDiv = document.createElement('div');
headerDiv.setAttribute('class', 'header');
document.body.appendChild(headerDiv);

var header = document.createElement('div');
header.setAttribute('id', 'header');
headerDiv.appendChild(header);

var weekLinks = document.createElement('div');
weekLinks.setAttribute('id', 'weekLinks');
headerDiv.appendChild(weekLinks);


var header = document.createElement('a');
header.setAttribute('href', '../CompPhy.html');
header.innerHTML = 'Back to Computational Physics';
header.style.position = 'relative';
document.getElementById('header').appendChild(header);

// Generate 14 week links dynamically
for (let i = 0; i <= 15; i++) {
    
    if (i === 0) {
        var leftArrow = document.createElement('a');
        leftArrow.setAttribute('href', `Week${currentNumber - 1}.html`);
        if (currentNumber === 1) {
            leftArrow.style.display = 'none';
        }
        /* Remove underline from leftArrow:*/
        leftArrow.style.textDecoration = 'none';
       
        leftArrow.innerHTML = '<<';
        document.getElementById('weekLinks').appendChild(leftArrow);
    }

    else if (i === 15) {
        var rightArrow = document.createElement('a');
        rightArrow.setAttribute('href', `Week${currentNumber + 1}.html`);
        rightArrow.style.textDecoration = 'none';
        if (currentNumber === 14) {
            rightArrow.style.display = 'none';
        }
    
        rightArrow.innerHTML = '>>';
        document.getElementById('weekLinks').appendChild(rightArrow);

    }
    else {
        var a = document.createElement('a');
    
        a.setAttribute('href', `Week${i}.html`);
        a.innerHTML = i;
        a.style.margin = '5px'; // Add some spacing between the links
    
        // Highlight the current week link
       
    
        // Append the anchor to the #weekLinks container
        if (i === currentNumber) {
    
            a.style.fontWeight = 'bold';
            a.style.color = 'red'; // Customize the style for the current page
        }
        document.getElementById('weekLinks').appendChild(a);
    }
}

// Remove display of arrows if the current page is not a week page:

if (currentNumber === -1) {
    leftArrow.style.display = 'none';
    rightArrow.style.display = 'none';
}






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





