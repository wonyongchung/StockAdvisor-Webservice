const bgImages = ['newyork.jpg', 'newyork2.jpg']
const selectedImg = bgImages[Math.floor(Math.random()*bgImages.length)]
const bgImage = document.createElement("img");
const bgImageUrl = `../static/front/images/${selectedImg}`
bgImage.classList.add('hidden')
bgImage.src = bgImageUrl
document.body.style.backgroundImage = `url(${bgImageUrl})`
console.log(`url(${bgImageUrl})`)
document.body.appendChild(bgImage);