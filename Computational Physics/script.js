var path = window.location.pathname;

// Extract the file name with extension
var fileName = path.split('/').pop();

// Extract the current number from the file name (assuming it's at the end)
const regex = /Week(\d+).html/; // Match 'Week' followed by a number
const match = fileName.match(regex);
const currentNumber = match ? parseInt(match[1]) : -1;

// Function to toggle cross-out effect and save the state in localStorage
function toggleCrossOut(checkbox) {
    const text = checkbox.nextElementSibling;
    const taskId = checkbox.id;

    if (checkbox.checked) {
        text.classList.add('crossed-out');
        localStorage.setItem(taskId, 'checked');
    } else {
        text.classList.remove('crossed-out');
        localStorage.setItem(taskId, 'unchecked');
    }
}

// Function to load the state of checkboxes from localStorage
function loadChecklistState() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        const taskId = checkbox.id;
        const state = localStorage.getItem(taskId);

        if (state === 'checked') {
            checkbox.checked = true;
            checkbox.nextElementSibling.classList.add('crossed-out');
        } else {
            checkbox.checked = false;
            checkbox.nextElementSibling.classList.remove('crossed-out');
        }
    });
}

// Load the checklist state when the page is loaded
document.addEventListener('DOMContentLoaded', loadChecklistState);



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
