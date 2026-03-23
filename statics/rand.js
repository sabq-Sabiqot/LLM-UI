function randomString(length = 6) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({length}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
}

function randomDiv(numClasses = 4, classLength = 6, idLength = 6) {
  // const classes = Array.from({length: numClasses}, () => randomString(classLength)).join(' ');
  const divId = randomString(idLength);

  const div = document.createElement('div');
  // div.className = classes;
  div.id = divId;
  div.textContent = `Random div dengan id=${divId}`;
  return div;
}

// Sisipkan ke halaman
const target = document.getElementById('r_gen');
target.appendChild(randomDiv());
target.appendChild(randomDiv()); // bisa panggil berkali-kali
