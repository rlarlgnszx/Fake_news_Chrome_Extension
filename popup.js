// function showFake()
document.getElementById('clickme').addEventListener('click', function () {
    chrome.tabs.executeScript(null, { file: "jquery-2.2.js" }, function () {
        chrome.tabs.executeScript(null, { file: "index.js" });
    });
});