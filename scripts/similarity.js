export function performSimilaritySearch(inputEmbedding, papersData, topK) {
    const cosineSimilarity = (vecA, vecB) => {
        let dotProduct = 0;
        let magnitudeA = 0;
        let magnitudeB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            magnitudeA += vecA[i] * vecA[i];
            magnitudeB += vecB[i] * vecB[i];
        }
        return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
    };

    const similarities = papersData.map(paper => {
        const similarity = cosineSimilarity(inputEmbedding, paper.embedding);
        return {
            ...paper,
            similarity: similarity
        };
    });

    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topK);
}

export function displayResults(topPapers) {
    const resultsContainer = document.getElementById("results");
    resultsContainer.innerHTML = "";

    if (topPapers.length === 0) {
        resultsContainer.innerHTML = "<p>No similar papers found.</p>";
        return;
    }

    topPapers.forEach((paper, index) => {
        const paperDiv = document.createElement("div");
        paperDiv.className = "similar-paper";

        const title = document.createElement("div");
        title.className = "similar-paper-title";
        title.textContent = `${index + 1}. ${paper.title}`;

        const similarity = document.createElement("div");
        similarity.className = "similarity-score";
        similarity.textContent = `Similarity Score: ${paper.similarity.toFixed(4)}`;

        const abstract = document.createElement("div");
        abstract.className = "abstract";
        const truncatedAbstract = paper.abstract.length > 150 ? paper.abstract.slice(0, 150) + "..." : paper.abstract;
        abstract.textContent = truncatedAbstract;

        const expandButton = document.createElement("button");
        expandButton.textContent = "Read More";
        expandButton.style.marginLeft = "10px";
        expandButton.style.padding = "5px 10px";
        expandButton.style.fontSize = "12px";
        expandButton.style.cursor = "pointer";
        expandButton.style.border = "1px solid #007bff";
        expandButton.style.backgroundColor = "#f1f8ff";
        expandButton.style.color = "#007bff";
        expandButton.style.borderRadius = "4px";

        let isExpanded = false;
        expandButton.addEventListener("click", () => {
            isExpanded = !isExpanded;
            if (isExpanded) {
                abstract.textContent = paper.abstract;
                expandButton.textContent = "Show Less";
            } else {
                abstract.textContent = truncatedAbstract;
                expandButton.textContent = "Read More";
            }
        });

        const link = document.createElement("a");
        link.href = `https://arxiv.org/abs/${paper.id}`;
        link.textContent = "View Paper";
        link.target = "_blank";
        link.style.display = "block";
        link.style.marginTop = "5px";
        link.style.color = "#007bff";
        link.style.textDecoration = "none";

        paperDiv.appendChild(title);
        paperDiv.appendChild(similarity);
        paperDiv.appendChild(abstract);
        paperDiv.appendChild(expandButton);
        paperDiv.appendChild(link);
        resultsContainer.appendChild(paperDiv);
    });
}