// alert("Optimize News... Waiting..")
let all_url = document.querySelectorAll('a[href*="new"]');
var api_url = 'https://summary-h24fed2hha-el.a.run.app';
var metadata =document.createElement('meta');
metadata.httpEquiv='Content-SecurityPolicy';
metadata.content='upgrade-insecure-request';
(document.head||document.documentElement).appendChild(metadata);



function unicodeToChar(text) {
    return text.replace(/\\u[\dA-F]{4}/gi,
        function (match) {
            return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
    });
}

function inputTooltip(obj) {
    var title = document.createElement("span");
    title.textContent = "title : ";
    var doc = document.createElement("div");
    doc.textContent = "summary :" + obj.document;
    title.classList.add("tooltiptext");
    // doc.classList.add('tooltiptext');
    var inp = obj.childNodes[0];
    title.appendChild(doc);
    obj.insertBefore(title, inp);
}

// const spawn = require('child_process').spawn;
async function get_url_document(obj) {
    await fetch(api_url, {
        method: 'POST',
        mode:'cors',
        headers: {
            'Content-Type': 'application/json',
        },
        referrer:'client',
        referrerPolicy:'origin',
        body: JSON.stringify({
            'newsurl': obj.href
        })
    }).then(data=>data.json()).then(data2=>{console.log(data2.title);obj.news_title=data2.title;obj.news_document=data2.document})
};
function get_url_document2(obj) {
    fetch(api_url, {
        method: 'POST',
        mode:'cors',
        headers: {
            'Content-Type': 'application/json',
        },
        referrer:'client',
        referrerPolicy:'origin',
        body: JSON.stringify({
            'newsurl': obj.href
        })
    }).then(data=>data.json()).then(data2=>{console.log(a.title);
        console.log(a.document);
        console.log(data2.title);
        obj.news_title=data2.title;
        obj.news_document=data2.document})
};
async function short_news(obj){
    await fetch('https://8080-a453e8ef-a5f4-481a-ac8c-f0e4364d1764.cs-asia-east1-jnrc.cloudshell.dev/handle_post', {
            method: 'POST',
            mode:'cors',
            headers: {
                'Content-Type': 'application/json',
            },
            referrer:'client',
            referrerPolicy:'origin',
            body: JSON.stringify({
                'document':obj.newsDoc
            })
        }).then(data=>data.json()).then(data2=>{obj.summary=data2.text})
    return obj;
}
// all_url.forEach((a, i) => {
var a = all_url[20];
a.classList.add("tooltip")
// get_url_document2(a);
fetch('https://summary-h24fed2hha-el.a.run.app', {
        method: 'POST',
        mode:'cors',
        headers: {
            'Content-Type': 'application/json',
        },
        referrer:'client',
        referrerPolicy:'origin',
        body: JSON.stringify({
            'newsurl': a.href
        })
}).then(data=>data.json()).then(data2=>{a.newsDoc = data2.document;a.newsTit=data2.title;
    fetch('http://ec2-13-124-98-178.ap-northeast-2.compute.amazonaws.com/handle_post', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',"User-Agent":'Mozilla/5.0',"Accept":"*/*"
            },
            body: JSON.stringify({
                'document':a.newsDoc
            })
        }).then(data3=>data3.json()).then(data4=>{
            console.log(data4);
            a.summary=data4.text
        })
});

// a.newsDoc = 기사제목
// a.newsTit = 기사 본문
// setTimeout(get_url_document,1000,a)
// console.log(NewsData);
short_news(a);

inputTooltip(a);
// })




