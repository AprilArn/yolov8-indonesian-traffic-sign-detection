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

    setActiveMode("Camera");
    clearStream();
    
    const stream = document.getElementById("video-stream");
    stream.src = "/camera_feed";
    document.getElementById("upload-form").style.display = "none";
    document.getElementById('video-stream').src = '/camera_feed'; // ganti dengan stream kamera jika perlu

}

function selectFileMode() {

    clearStream(); // reset tampilan
    document.getElementById("upload-form").style.display = "block"; // tampilkan upload form
    // stream.src = ""; // Hentikan stream video
    setActiveMode("File Upload");

}

function startDetection() {

    setActiveMode("File Upload");

    const stream = document.getElementById("video-stream");
    const uploadForm = document.getElementById("upload-form");
    
    // Tampilkan tombol upload dulu
    // uploadForm.style.display = "block";
    
    // Periksa apakah video sudah diupload
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

    // Jika source kosong, hentikan deteksi dan kosongkan hasil
    if (!stream.src || stream.src === window.location.origin + "/") {
        const container = document.getElementById("result-box");
        container.innerHTML = ""; // Bersihkan hasil sebelumnya

        const item = document.createElement("div");
        item.className = "result-item";
        item.innerText = ""; // Kosongkan isi
        container.appendChild(item);
        return;
    }

    // Lanjutkan deteksi jika stream aktif
    fetch('/latest_detection')
    .then(res => res.json())
    .then(data => {
        const container = document.getElementById("result-box");
        container.innerHTML = ""; // Hapus hasil lama

        if (data.detections.length === 0) {
            const item = document.createElement("div");
            item.className = "result-item";
            item.innerText = "No objects detected.";
            container.appendChild(item);
        } else {
            data.detections.forEach(det => {
                const item = document.createElement("div");
                item.className = "result-item";
                item.innerText = det;
                container.appendChild(item);
            });
        }
    });
        
}, 500);