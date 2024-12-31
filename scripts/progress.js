export function initializeProgressBar() {
    const loadingOverlay = document.getElementById("loadingOverlay");
    const progressBar = document.getElementById("progressBar");
    const progressPercentage = document.getElementById("progressPercentage");
    const estimatedTime = document.getElementById("estimatedTime");

    return { progressBar, progressPercentage, estimatedTime, loadingOverlay };
}

export function updateProgressBar(percentage, estimatedTimeSec, progressBar, progressPercentage, estimatedTimeText) {
    progressBar.style.width = `${percentage}%`;
    progressPercentage.textContent = `${percentage}%`;
    estimatedTimeText.textContent = `Estimated Time: ${estimatedTimeSec} seconds`;
    progressBar.setAttribute('aria-valuenow', percentage);
}

export function setProgressToComplete(progressBar, progressPercentage, estimatedTimeText) {
    progressBar.style.width = `100%`;
    progressPercentage.textContent = `100%`;
    estimatedTimeText.textContent = `Estimated Time: 0 seconds`;
    progressBar.setAttribute('aria-valuenow', '100');
}