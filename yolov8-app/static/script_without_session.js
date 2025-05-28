const urlParams = new URLSearchParams(window.location.search);
const filename = urlParams.get('filename');

function setActiveMode(modeName) {
    const buttons = document.querySelectorAll(".mode-buttons .btn");
    buttons.forEach(btn => {
        if (btn.innerText === modeName) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
}

function startCameraMode() {
    clearStream();
    setActiveMode("Camera");

    const stream = document.getElementById("video-stream");
    stream.src = "/camera_feed";
    document.getElementById("upload-form").style.display = "none";

    const clearBtn = document.querySelector(".clear-btn");
    clearBtn.classList.add("disabled");

    const startBtn = document.querySelector(".start-btn");
    startBtn.classList.add("disabled");
}

function selectFileMode() {
    clearStream();
    handleUploadStart();
    document.getElementById("upload-form").style.display = "block";
    setActiveMode("File Upload");
}

function handleUploadStart() {
    const clearBtn = document.querySelector(".clear-btn");
    clearBtn.classList.remove("disabled");

    const startBtn = document.querySelector(".start-btn");
    startBtn.classList.add("disabled");

    const saveBtn = document.querySelector(".save-btn");
    saveBtn.classList.add("disabled");

    window.addEventListener('load', () => {
        startBtn.classList.remove("disabled");
        saveBtn.classList.remove("disabled");
    });
}

function saveResults() {
    if (!filename) return alert("Filename not found in URL");

    fetch(`/save_prediction?filename=${filename}`, {
        method: 'POST'
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "ok") {
            alert("Prediction saved successfully!");
            console.log("Saved to:", data.output);
        } else {
            alert("Failed to save prediction: " + data.message);
        }
    })
    .catch(err => {
        console.error("Error saving prediction:", err);
        alert("Failed to process video.");
    });
}

function startDetection() {
    if (!filename) return alert("Filename not found in URL");

    setActiveMode("File Upload");

    const stream = document.getElementById("video-stream");
    const uploadForm = document.getElementById("upload-form");

    fetch(`/check_video_uploaded?filename=${filename}`)
    .then(res => res.json())
    .then(data => {
        if (data.status === "ok") {
            stream.src = `/video_feed?filename=${filename}`;
            uploadForm.style.display = "none";
        } else {
            alert("Please upload a video first!");
        }
    })
    .catch(error => {
        console.error("Error checking upload status:", error);
        alert("Failed to check video status. Please try again.");
    });
}

function clearStream() {
    handleUploadStart();

    const stream = document.getElementById("video-stream");
    stream.src = "";
    document.getElementById("upload-form").style.display = "block";
    setActiveMode("");

    fetch('/clear_uploaded_video', {
        method: 'POST'
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "deleted") {
            console.log("Uploaded file deleted.");
        } else {
            console.log("No file deleted.");
        }
    })
    .catch(err => {
        console.error("Failed to delete file:", err);
    });
}

setInterval(() => {
    const stream = document.getElementById("video-stream");
    const isStreamInactive = !stream.src || stream.src === window.location.origin + "/";
    const fpsBox = document.getElementById("fps-box");
    const resultContainer = document.getElementById("result-box");

    if (isStreamInactive) {
        if (fpsBox) fpsBox.innerText = "FPS: --";
        if (resultContainer) {
            resultContainer.innerHTML = "";
            const item = document.createElement("div");
            item.className = "result-item";
            item.innerText = "";
            resultContainer.appendChild(item);
        }
        return;
    }

    fetch('/get_fps')
        .then(res => res.json())
        .then(data => {
            if (fpsBox) {
                fpsBox.innerText = `FPS: ${data.fps}`;
            }
        });

    fetch('/latest_detection')
        .then(res => res.json())
        .then(data => {
            if (resultContainer) {
                resultContainer.innerHTML = "";
                if (data.detections.length === 0) {
                    const item = document.createElement("div");
                    item.className = "result-item";
                    item.innerText = "No objects detected.";
                    resultContainer.appendChild(item);
                } else {
                    data.detections.forEach(det => {
                        const item = document.createElement("div");
                        item.className = "result-item";
                        item.innerText = det;
                        resultContainer.appendChild(item);
                    });
                }
            }
        });

}, 500);
