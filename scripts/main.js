import { loadPapersData } from './dataloader.js';
import { performSimilaritySearch, displayResults } from './similarity.js';
import { initializeTokenizer, inferModel, normalize } from './tokenizer.js';
import { initializeProgressBar } from './progress.js';

document.addEventListener("DOMContentLoaded", async () => {
    const resultsContainer = document.getElementById("results");
    const embeddingSelect = document.getElementById("embeddingSelect");
    const resultCountSelect = document.getElementById("resultCountSelect");
    const tokenizeButton = document.getElementById("tokenizeButton");
    const textInput = document.getElementById("textInput");

    const { progressBar, progressPercentage, estimatedTime, loadingOverlay } = initializeProgressBar();

    const BASE_URL = 'https://huggingface.co/datasets/napronald/AI-Paper-Similarity-Search/resolve/main/';

    const tokenizerPromise = initializeTokenizer();
    let currentPapersData = await loadPapersData(
        '10k_papers_embeddings.json',
        BASE_URL,
        progressBar,
        progressPercentage,
        estimatedTime,
        loadingOverlay
    );

    const [tokenizer] = await Promise.all([tokenizerPromise]);

    embeddingSelect.addEventListener("change", async (event) => {
        const selectedEmbeddings = event.target.value;
        currentPapersData = await loadPapersData(
            selectedEmbeddings,
            BASE_URL,
            progressBar,
            progressPercentage,
            estimatedTime,
            loadingOverlay
        );
        resultsContainer.innerHTML = "";
    });

    tokenizeButton.addEventListener("click", async () => {
        const query = textInput.value.trim();
        if (!query) {
            alert("Please enter a query.");
            return;
        }

        const topK = parseInt(resultCountSelect.value);
        tokenizeButton.disabled = true;
        tokenizeButton.textContent = "Processing...";

        try {
            const embedding = await inferModel(tokenizer, query);
            const normalizedEmbedding = normalize(embedding);
            const topPapers = performSimilaritySearch(normalizedEmbedding, currentPapersData, topK);
            displayResults(topPapers);
        } finally {
            tokenizeButton.disabled = false;
            tokenizeButton.textContent = "Find Similar Papers";
        }
    });
});