async function fetchStatus(){
  const res = await fetch('/status');
  if (!res.ok) return;
  const j = await res.json();
  const list = document.getElementById('filesList');
  list.innerHTML = '';
  if (j.files && j.files.length){
    j.files.forEach(f=>{
      const d = document.createElement('div');
      d.textContent = f.filename;
      list.appendChild(d);
    });
  } else {
    list.textContent = 'No files uploaded yet.';
  }
  const vs = document.getElementById('vsStatus');
  vs.textContent = j.vectorstore ? 'Indexed' : 'Not indexed';
}

document.addEventListener('DOMContentLoaded', ()=>{
  fetchStatus();

  const uploadForm = document.getElementById('uploadForm');
  uploadForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const files = document.getElementById('files').files;
    if (!files.length) return alert('Select at least one PDF');
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true; btn.textContent = 'Uploading...';
    const res = await fetch('/upload', {method:'POST', body: fd});
    const j = await res.json();
    btn.disabled = false; btn.textContent = 'Upload & Index';
    if (!res.ok) return alert('Upload error: ' + (j.error||res.statusText));
    fetchStatus();
    alert('Indexed ' + j.documents + ' documents')
  });

  const chatForm = document.getElementById('chatForm');
  const chatWindow = document.getElementById('chatWindow');
  const questionEl = document.getElementById('question');
  const clearBtn = document.getElementById('clearBtn');

  chatForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const q = questionEl.value.trim();
    if (!q) return;
    appendMessage('user', q);
    questionEl.value = '';
    const thinking = appendMessage('ai', 'Thinking...');
    try{
      const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q, chat_history:[]})});
      const j = await res.json();
      thinking.remove();
      if (!res.ok){ appendMessage('ai', 'Error: ' + (j.error||res.statusText)); return; }
      appendMessage('ai', j.answer || 'No answer');
      if (j.sources && j.sources.length){
        const s = document.createElement('div'); s.className='sources';
        s.textContent = 'Sources: ' + j.sources.map(x=>x.source || x.filename || JSON.stringify(x)).join(', ');
        chatWindow.appendChild(s);
      }
    }catch(err){
      thinking.remove();
      appendMessage('ai', 'Request failed: ' + err.toString());
    }
  });

  clearBtn.addEventListener('click', async ()=>{
    if (!confirm('Clear index and manifest?')) return;
    const res = await fetch('/clear', {method:'POST'});
    const j = await res.json();
    if (!res.ok) return alert('Clear failed: ' + (j.error||res.statusText));
    fetchStatus();
    alert('Cleared')
  });
});

function appendMessage(who, text){
  const chatWindow = document.getElementById('chatWindow');
  const div = document.createElement('div');
  div.className = 'message ' + who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}
