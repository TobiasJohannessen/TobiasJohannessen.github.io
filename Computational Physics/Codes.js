// Make a link based on the folder name
var path = window.location.pathname;
var folderName = path.split('/')[3];
folderName = folderName.replace('%20', ' ');
var a = document.createElement('a');
a.innerHTML = 'Back to ' + folderName;
folderName = folderName.replace(' ', '');
a.setAttribute('href', `../../${folderName}.html`);
a.style.margin = '5px';
document.getElementById('backLink').appendChild(a);