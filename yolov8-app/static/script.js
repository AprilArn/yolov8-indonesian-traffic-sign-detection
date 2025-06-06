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
    document.getElementById("upload-form").style.display = "none";    // Sembunyikan form upload

    // Nonaktifkan tombol Clear
    const clearBtn = document.querySelector(".clear-btn");
    clearBtn.classList.add("disabled");

    // Nonaktifkan tombol Start
    const startBtn = document.querySelector(".start-btn");
    startBtn.classList.add("disabled");

}

function selectFileMode() {

    clearStream(); // reset tampilan
    handleUploadStart();
    document.getElementById("upload-form").style.display = "block"; // tampilkan upload form
    setActiveMode("File Upload");

}

function handleUploadStart() {

    //  Aktifkan tombol Clear
    const clearBtn = document.querySelector(".clear-btn");
    clearBtn.classList.remove("disabled");

    // Nonaktifkan tombol Start
    const startBtn = document.querySelector(".start-btn");
    startBtn.classList.add("disabled");

    // Nonaktifkan tombol Save
    const saveBtn = document.querySelector(".save-btn");
    saveBtn.classList.add("disabled");

    // Aktifkan kembali tombol Start dan Save setelah halaman dimuat
    window.addEventListener('load', () => {
        startBtn.classList.remove("disabled");  // Aktifkan tombol Start
        saveBtn.classList.remove("disabled");   // Aktifkan tombol Save
    });

}

function saveResults() {

    const saveBtn = document.querySelector(".save-btn");
    const startBtn = document.querySelector(".start-btn");

    // Nonaktifkan tombol Save dan Start
    saveBtn.classList.add("disabled");
    startBtn.classList.add("disabled");

    fetch('/save_prediction', {
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

    setActiveMode("File Upload");

    const stream = document.getElementById("video-stream");
    const uploadForm = document.getElementById("upload-form");
    const saveBtn = document.querySelector(".save-btn");
    const startBtn = document.querySelector(".start-btn");

    // Nonaktifkan tombol Save dan Start
    saveBtn.classList.add("disabled");
    startBtn.classList.add("disabled");
    
    fetch("/check_video_uploaded")
    .then(res => res.json())
    .then(data => {
        if (data.status === "ok") {
            // Jika file sudah diupload, mulai stream deteksi
            stream.src = "/video_feed";
            uploadForm.style.display = "none"; // sembunyikan tombol upload
        } else {
            // Jika belum, biarkan tombol upload tampil dan beri alert
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

    // Hentikan stream jika kamera aktif
    const videoElement = stream;
    if (videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
    }

    stream.src = "";
    document.getElementById("upload-form").style.display = "block";
    setActiveMode("");

    // Request ke server untuk hapus file yang sudah di-upload
    fetch('/clear_uploaded_video', {
        method: 'POST'
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "deleted") {
            console.log("File upload berhasil dihapus.");
        } else {
            console.log("Tidak ada file yang dihapus.");
        }
    })
    .catch(err => {
        console.error("Gagal menghapus file:", err);
    });

}

setInterval(() => {
    
    const stream = document.getElementById("video-stream");
    const isStreamInactive = !stream.src || stream.src === window.location.origin + "/";
    const fpsBox = document.getElementById("fps-box");
    const resultContainer = document.getElementById("result-box");

    if (isStreamInactive) {
        // Kosongkan FPS dan hasil deteksi
        if (fpsBox) fpsBox.innerText = "FPS: --";
        if (resultContainer) {
            resultContainer.innerHTML = "";
            const item = document.createElement("div");
            item.className = "result-item";
            item.innerText = ""; // Kosongkan isi
            resultContainer.appendChild(item);
        }
        return;
    }

    // Ambil FPS
    fetch('/get_fps')
        .then(res => res.json())
        .then(data => {
            if (fpsBox) {
                fpsBox.innerText = `FPS: ${data.fps}`;
            }
        });

    // Ambil hasil deteksi
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

}, 500); // Setiap 500ms
