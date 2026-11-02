const text = document.querySelector('#text');
const counter = document.querySelector('#counter');
const button = document.querySelector('#detect');
const error = document.querySelector('#error');
const result = document.querySelector('#result');
text.addEventListener('input', () => counter.textContent = `${text.value.length} / 5000`);
button.addEventListener('click', async () => {
  error.textContent = ''; result.classList.add('hidden'); button.disabled = true; button.firstChild.textContent = 'Detecting ';
  try {
    const response = await fetch('/detect', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:text.value})});
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Could not analyze this text.');
    document.querySelector('#language').textContent = data.language;
    document.querySelector('#confidence').textContent = `${Math.round(data.confidence * 100)}%`;
    document.querySelector('#notice').textContent = data.notice;
    document.querySelector('#bars').innerHTML = data.top_languages.map(item => `<div class="bar-row"><div class="bar-label"><span>${item.language}</span><span>${Math.round(item.probability * 100)}%</span></div><div class="track"><div class="fill" style="width:${item.probability * 100}%"></div></div></div>`).join('');
    result.classList.remove('hidden');
  } catch (e) { error.textContent = e.message; } finally { button.disabled = false; button.firstChild.textContent = 'Detect language '; }
});
