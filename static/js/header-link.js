(function() {
  var d, h, hs = document.querySelectorAll(".section.level1 h1, .section.level2 h2, .section.level3 h3, .section.level4 h4");
  for (var i = 0; i < hs.length; i++) {
    h = hs[i];
    d = h.parentNode.id;
    if (d === '') continue;
    h.innerHTML += ' <span class="anchor"><a href="#' + d + '">#</a></span>';
  }
  
})();
(function() {
  var h2, hs2 = document.querySelectorAll('main h1, main h2, main h3, main h4');
  for (var i = 0; i < hs2.length; i++) {
    h2 = hs2[i];
    if (h2.id === '') continue;
    h2.innerHTML += ' <span class="anchor"><a href="#' + h2.id + '">#</a></span>';
  }
})();