const Folders = ["exam", "exercises", "homework"]
const Titles = ["Exam", "Exercises", "Homework"]; // Add this line
function createFolderElement(i) {
    const linkContainer = document.createElement("div");
    linkContainer.className = "linkContainer";
    const link = document.createElement("a");
    const filename = Folders[i].replace(' ', '');
    link.href = `${filename}/${filename}.html`;
    const img = document.createElement("img");
    const pngSrc = `${filename}/thumbnail.png`;
    const gifSrc = `${filename}/thumbnail.gif`;
    const svgSrc = `${filename}/thumbnail.svg`;

    (async () => {
    try {
        const pngResp = await fetch(pngSrc, { method: 'HEAD' });
        if (pngResp.ok) {
            img.src = pngSrc;
            return;
        }

        const gifResp = await fetch(gifSrc, { method: 'HEAD' });
        if (gifResp.ok) {
            img.src = gifSrc;
            return;
        }

        const svgResp = await fetch(svgSrc, { method: 'HEAD' });
        if (svgResp.ok) {
            img.src = svgSrc;
            return;
        }

        img.src = "default.png"; // Fallback
    } catch (error) {
        console.error("Error fetching images:", error);
        img.src = "default.png";
    }
})();

    const container = document.querySelector(".container");
    link.appendChild(img);
    container.appendChild(link);

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    const overlayText = document.createElement("div");
    overlayText.className = "text";
    overlayText.textContent = ` ${Titles[i]}`;
    overlay.appendChild(overlayText);
    link.appendChild(overlay);

    const bottomText = document.createElement("div");
    bottomText.className = "text";
    bottomText.textContent = `${Titles[i]}`; // Use Titles array for bottom text
    link.appendChild(bottomText);

    linkContainer.appendChild(link);

    return linkContainer;
}








const content = document.querySelector(".content");
const container = document.querySelector(".container")

function initialize() {


    // Add the initial Week elements
    
    for (let i = 0; i <= 2; i++) {
        container.appendChild(createFolderElement(i));
    }
}



initialize();
 

