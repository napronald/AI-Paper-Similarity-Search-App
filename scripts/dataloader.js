import { updateProgressBar, setProgressToComplete } from './progress.js';

export async function loadPapersData(
    embeddingsFile,
    BASE_URL,
    progressBar,
    progressPercentage,
    estimatedTime,
    loadingOverlay
) {
    try {
        loadingOverlay.style.display = "flex";
        updateProgressBar(0, 'Calculating...', progressBar, progressPercentage, estimatedTime);
        let data = await loadFromIndexedDB(embeddingsFile);
        if (!data) {
            const rawUrl = `${BASE_URL}${embeddingsFile}`;
            const jsonText = await fetchWithProgress(rawUrl, (percentage, estTime) => {
                if (percentage !== null && estTime !== null) {
                    updateProgressBar(percentage, estTime, progressBar, progressPercentage, estimatedTime);
                } else {
                    progressBar.style.width = `100%`;
                    progressPercentage.textContent = `Loading...`;
                    estimatedTime.textContent = `Estimated Time: Calculating...`;
                }
            });
            data = JSON.parse(jsonText);
            await saveToIndexedDB(embeddingsFile, data);
        } else {
            setProgressToComplete(progressBar, progressPercentage, estimatedTime);
        }
        return data;
    } finally {
        loadingOverlay.style.display = "none";
    }
}

function saveToIndexedDB(key, data) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("EmbeddingsDB", 2);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains("embeddings")) {
                db.createObjectStore("embeddings", { keyPath: "filename" });
            }
        };
        request.onsuccess = () => {
            const db = request.result;
            const tx = db.transaction("embeddings", "readwrite");
            const store = tx.objectStore("embeddings");
            store.put({ filename: key, data: data });
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        };
        request.onerror = () => reject(request.error);
    });
}

function loadFromIndexedDB(key) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("EmbeddingsDB", 2);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains("embeddings")) {
                db.createObjectStore("embeddings", { keyPath: "filename" });
            }
        };
        request.onsuccess = () => {
            const db = request.result;
            const tx = db.transaction("embeddings", "readonly");
            const store = tx.objectStore("embeddings");
            const getRequest = store.get(key);
            getRequest.onsuccess = () => {
                if (getRequest.result) {
                    resolve(getRequest.result.data);
                } else {
                    resolve(null);
                }
            };
            getRequest.onerror = () => reject(getRequest.error);
        };
        request.onerror = () => reject(request.error);
    });
}

async function fetchWithProgress(url, onProgress) {
    const response = await fetch(url);
    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : null;
    let loaded = 0;
    const reader = response.body.getReader();
    const chunks = [];
    const startTime = Date.now();
    let previousPercentage = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;

        if (total) {
            let percentage = ((loaded / total) * 100).toFixed(2);
            if (percentage > 100) percentage = 100;

            if (Math.abs(percentage - previousPercentage) > 0.5) {
                previousPercentage = percentage;
            } else {
                percentage = previousPercentage;
            }

            const elapsedTime = (Date.now() - startTime) / 1000;
            const speed = loaded / elapsedTime; 
            const remainingBytes = total - loaded;
            const estimatedTimeSec = speed > 0 ? (remainingBytes / speed).toFixed(1) : 'Calculating...';

            onProgress(percentage, estimatedTimeSec);
        }
    }

    if (total) {
        onProgress(100, '0');
    }

    const chunksAll = new Uint8Array(loaded);
    let position = 0;
    for (let chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }

    return new TextDecoder("utf-8").decode(chunksAll);
}