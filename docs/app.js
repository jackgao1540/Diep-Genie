'use strict';
const $ = id => document.getElementById(id);
const labels = {accepted: 'Retained', held: 'Withheld', rejected: 'Rejected'};
let data, selected, images, frame = 0, playing = false, last = 0, generation = 0, filter = 'all';
function text(id, value) { $(id).textContent = value; }
function pause() { playing = false; text('play', '▶ Play'); $('play').setAttribute('aria-label', 'Play clip'); }
function draw() {
  if (!images) return;
  for (const name of ['original', 'processed']) {
    const canvas = $(name), [w, h] = name === 'original' ? selected.source_crop_size : selected.processed_size;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(images[name], (frame % 8) * w, Math.floor(frame / 8) * h, w, h, 0, 0, w, h);
  }
  $('scrub').value = frame;
  $('scrub').setAttribute('aria-valuetext', `Frame ${frame + 1} of 40, ${(frame / 10).toFixed(1)} seconds`);
  text('time', `${(frame / 10).toFixed(1)} / 4.0s`);
}
function animate(now) {
  if (playing && images) {
    const step = 1000 / (selected.fps * Number($('speed').value));
    if (now - last >= step) { const n = Math.floor((now - last) / step); frame = (frame + n) % 40; last += n * step; draw(); }
  }
  requestAnimationFrame(animate);
}
function loadImage(path) { return new Promise((resolve, reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = () => reject(new Error('The footage could not load. Please reload the page.')); im.src = path; }); }
async function choose(clip) {
  const token = ++generation; pause(); selected = clip; images = null; frame = 0;
  for (const id of ['play', 'back', 'next', 'scrub']) $(id).disabled = true;
  for (const id of ['original', 'processed']) $(id).getContext('2d').clearRect(0, 0, $(id).width, $(id).height);
  document.querySelector('.viewer').setAttribute('aria-busy', 'true');
  text('load-status', 'Loading selected frames…'); text('clip-title', clip.title);
  text('badge', labels[clip.status]); $('badge').className = `badge ${clip.status}`;
  text('clip-position', `${String(data.clips.indexOf(clip) + 1).padStart(2, '0')} / 13`);
  text('source-size', clip.source_crop_size.join(' × ')); text('description', clip.description);
  const notes = {retained_proxy: 'Processed view: the exact retained 128px proxy frames.', withheld_proxy: 'Processed view: a preserved proxy from the withheld source. Excluded from the current cohort.', illustrative_resize_not_training_data: 'Processed view: an illustrative 128px resize. This rejected clip is not training data.'};
  text('processing-note', notes[clip.processed_kind]);
  $('tags').replaceChildren(...clip.tags.map(tag => { const el = document.createElement('span'); el.textContent = tag; return el; }));
  $('source-link').href = clip.source_url; text('source-link', `${clip.source_title} ↗`);
  text('source-meta', `${clip.creator} · ${clip.split === 'train' ? 'Training' : 'Validation'} source · ${Math.floor(clip.start / 60)}:${String(Math.floor(clip.start % 60)).padStart(2, '0')}`);
  renderGallery();
  try {
    const [original, processed] = await Promise.all([loadImage(clip.media.original.path), loadImage(clip.media.processed.path)]);
    if (token !== generation) return;
    images = {original, processed}; draw(); text('load-status', '');
    for (const id of ['play', 'back', 'next', 'scrub']) $(id).disabled = false;
  } catch (error) { if (token === generation) text('load-status', error.message); }
  finally { if (token === generation) document.querySelector('.viewer').setAttribute('aria-busy', 'false'); }
}
function renderGallery() {
  const clips = data.clips.filter(c => filter === 'all' || c.status === filter);
  text('sample-count', `${clips.length} example${clips.length === 1 ? '' : 's'}`);
  $('gallery').replaceChildren(...clips.map(clip => {
    const button = document.createElement('button'); button.className = 'sample'; button.setAttribute('aria-pressed', String(clip === selected)); button.setAttribute('aria-label', `${clip.title}, ${labels[clip.status]}`);
    const top = document.createElement('div'); top.className = 'sample-top';
    const image = document.createElement('img'); image.src = clip.media.thumbnail.path; image.alt = ''; image.loading = 'lazy'; image.width = 128; image.height = 128;
    const badge = document.createElement('span'); badge.className = `badge ${clip.status}`; badge.textContent = labels[clip.status];
    const duration = document.createElement('span'); duration.className = 'sample-time'; duration.textContent = '04s'; top.append(image, badge, duration);
    const body = document.createElement('div'); body.className = 'sample-body'; const title = document.createElement('strong'); title.textContent = clip.title; const sub = document.createElement('small'); sub.textContent = `${clip.creator} · ${clip.split === 'train' ? 'Train' : 'Validation'}`; body.append(title, sub); button.append(top, body);
    button.addEventListener('click', () => { choose(clip); document.querySelector('.viewer').scrollIntoView({block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'}); });
    return button;
  }));
}
$('play').addEventListener('click', () => { if (playing) pause(); else { playing = true; last = performance.now(); text('play', 'Ⅱ Pause'); $('play').setAttribute('aria-label', 'Pause clip'); } });
for (const [id, amount] of [['back', -1], ['next', 1]]) $(id).addEventListener('click', () => { pause(); frame = (frame + amount + 40) % 40; draw(); });
$('scrub').addEventListener('input', () => { pause(); frame = Number($('scrub').value); draw(); });
$('speed').addEventListener('change', () => { last = performance.now(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => { filter = button.dataset.filter; document.querySelectorAll('[data-filter]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); renderGallery(); }));
fetch('showcase.json').then(r => { if (!r.ok) throw new Error('Could not load the showcase manifest.'); return r.json(); }).then(value => { data = value; return choose(data.clips[0]); }).catch(error => { text('load-status', `${error.message} Please reload the page.`); text('clip-title', 'Showcase unavailable'); document.querySelector('.viewer').setAttribute('aria-busy', 'false'); });
requestAnimationFrame(animate);
