// const current = document.querySelector('#current');
const imgs = document.querySelectorAll('.imgs img');

const modal = document.getElementById('myModal');
const modalImg = document.getElementById("img01");
const captionText = document.getElementById("caption");
const opacity = 0.4;

imgs.forEach(img => 
	img.addEventListener('click', imgClick)
);

function imgClick(e){

	imgs.forEach(img => (img.style.opacity = 1));
// 
	// current.src = e.target.src;

	modal.style.display = "block";
  	modalImg.src = e.target.src;
  	captionText.innerHTML = e.target.alt;

	e.target.style.opacity = opacity;
}

var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() { 
  modal.style.display = "none";
}

// When the user scrolls the page, execute myFunction 
window.onscroll = function() {myFunction()};

// Get the navbar
var navbar = document.getElementById("navbar");

// Get the offset position of the navbar
var sticky = navbar.offsetTop;

// Add the sticky class to the navbar when you reach its scroll position. Remove "sticky" when you leave the scroll position
function myFunction() {
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}