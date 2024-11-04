function createWeekElement(i) {
    const linkContainer = document.createElement("div");
    linkContainer.className = "linkContainer";

    const link = document.createElement("a");
    link.href = `Weeks/Week${i}.html`;

    const img = document.createElement("img");
    const pngSrc = `Weeks/Figures/Thumbnails/Week${i}.png`;
    const gifSrc = `Weeks/Figures/Thumbnails/Week${i}.gif`;

    fetch(pngSrc, { method: 'HEAD' })
        .then(response => {
            img.src = response.ok ? pngSrc : gifSrc;
        })
        .catch(() => {
            img.src = gifSrc;
        });
    
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

// Add elements dynamically to the container
const container = document.querySelector(".container");
for (let i = 1; i <= 14; i++) {
    container.appendChild(createWeekElement(i));
}
