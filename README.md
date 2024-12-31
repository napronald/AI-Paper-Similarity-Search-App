# AI Paper Similarity Search
AI Paper Similarity Search is a **client-side web application** that allows users to search for academic papers similar to their query. The application uses machine learning to rank papers based on their relevance and provides direct links to the papers.

**Live Demo**: [AI Paper Similarity Search App](https://napronald.github.io/AI-Paper-Similarity-Search-App/)

**Note:** The app is optimized for smaller subsets to provide a seamless user experience without overwhelming the browser.

## Video Demo
https://github.com/user-attachments/assets/49507189-9110-426b-9adb-9773761f8727

## Usage
1. **Enter a Query:**

    Input a keyword or phrase related to your area of interest (e.g., "computer vision") in the search box.

2. **Select Embeddings Size:**

    Choose the number of paper embeddings to load from the dropdown menu:
    
    - **10K Papers (Default):** Fast loading with a smaller dataset.
    - **25K Papers (Medium):** Balanced option for performance and comprehensiveness.
    - **50K Papers (Large):** More comprehensive search with more paper embeddings.

3. **Choose Number of Results:**

    Select how many top similar papers you want to retrieve (5, 10, 25, 50).

4. **Find Similar Papers:**

    Click the "Find Similar Papers" button to perform the search. The application will display the most relevant papers along with their similarity scores and links to arXiv.

5. **View Abstracts:**

    Click "Read More" to expand and read the full abstract of a paper.

## Project Structure
```plaintext
AI-Paper-Similarity-Search-App/
├── index.html
├── styles/
│   └── style.css
├── scripts/
│   ├── main.js
│   ├── dataloader.js
│   ├── similarity.js
│   ├── tokenizer.js
│   ├── progress.js
├── libs/
│   └── transformers.min.js 
└── README.md
```
- **index.html**: Main entry point  
- **styles/style.css**: Global styling  
- **scripts/main.js:** Initializes the application and handles user interactions.
- **scripts/dataloader.js:** Manages loading and storing paper embeddings using IndexedDB.
- **scripts/similarity.js:** Performs similarity search and displays results.
- **scripts/tokenizer.js:** Initializes the tokenizer and performs model inference.
- **scripts/progress.js:** Manages the progress bar during data loading.
- **libs/transformers.min.js:** External library for Transformers.js.
- **README.md:** Project documentation.

## Tech Stack
- **HTML**, **CSS**, and **JavaScript**  
- **ONNX Runtime:** For running the machine learning model directly in the browser.
- **Transformers.js:** Tokenization and model inference.
- **IndexedDB:** For storing and retrieving paper embeddings locally.
- **arXiv Dataset:** Source of the academic papers.
