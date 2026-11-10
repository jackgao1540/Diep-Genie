'use strict';

// The local format accepts embedded raster images only. Imported files never
// supply a fetch URL; local images are decoded directly inside the browser.
(function () {
  const MAX_FILE_BYTES = 200 * 1024 * 1024;
  const MAX_IMAGE_CHARS = 16 * 1024 * 1024;
  const MAX_TOTAL_PIXELS = 128 * 1024 * 1024;
  const KEYS = ['source', 'before', 'after'];
  function boundedText(value, fallback, limit = 500) {
    if (value == null) return fallback;
    if (typeof value !== 'string' || value.length > limit) throw new Error('Comparison text is invalid or too long.');
    return value;
  }
  function integer(value, low, high, label) {
    if (!Number.isInteger(value) || value < low || value > high) throw new Error(`Invalid ${label}.`);
    return value;
  }
  function mediaSource(media, local) {
    if (!media || typeof media !== 'object') throw new Error('A comparison image is missing.');
    if (local) {
      const uri = media.data_url;
      if (typeof uri !== 'string' || uri.length > MAX_IMAGE_CHARS || !/^data:image\/(?:png|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(uri)) {
        throw new Error('Local comparisons must contain embedded PNG or WebP images. External image links are not opened.');
      }
      const payload = uri.slice(uri.indexOf(',') + 1);
      let header;
      try { header = atob(payload.slice(0, 24)); } catch (_) { throw new Error('A local image has invalid encoding.'); }
      const png = uri.startsWith('data:image/png;') && header.startsWith('\x89PNG\r\n\x1a\n');
      const webp = uri.startsWith('data:image/webp;') && header.startsWith('RIFF') && header.slice(8, 12) === 'WEBP';
      if (!png && !webp) throw new Error('A local image is not a PNG or WebP raster.');
      return uri;
    }
    const path = media.path;
    if (typeof path !== 'string' || !/^assets\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+\.(?:png|webp)$/.test(path)) {
      throw new Error('Invalid public comparison image path.');
    }
    return path;
  }
  function safeCredit(url) {
    if (typeof url !== 'string') return null;
    // A credit is navigated only after a click; it is never fetched or embedded.
    return /^https:\/\/www\.youtube\.com\/watch\?v=[A-Za-z0-9_-]{11}(?:&t=\d+s)?$/.test(url) ? url : null;
  }
  function validateManifest(raw, local = false) {
    if (!raw || raw.schema_version !== 1 || raw.kind !== 'diep-genie-reconstructions') throw new Error('This is not a supported reconstruction comparison file.');
    if (!Array.isArray(raw.clips) || raw.clips.length < 1 || raw.clips.length > 128) throw new Error('A comparison needs between 1 and 128 clips.');
    const checkpoints = {};
    for (const key of ['before', 'after']) {
      const checkpoint = raw.checkpoints && raw.checkpoints[key];
      if (!checkpoint || typeof checkpoint.preflight !== 'boolean') throw new Error('Checkpoint information is incomplete.');
      checkpoints[key] = {step: integer(checkpoint.step, 0, 100000000, 'checkpoint step'), preflight: checkpoint.preflight,
        label: boundedText(checkpoint.label, '', 60)};
    }
    const seen = new Set(); let pixels = 0;
    const clips = raw.clips.map(clip => {
      if (!clip || typeof clip !== 'object') throw new Error('Invalid clip.');
      const id = boundedText(clip.id, '', 100);
      if (!id || seen.has(id)) throw new Error('Clip identifiers must be present and unique.');
      seen.add(id);
      if (!['train', 'val'].includes(clip.split)) throw new Error('Only training or validation examples can be opened.');
      const width = integer(clip.width, 8, 512, 'frame width');
      const height = integer(clip.height, 8, 512, 'frame height');
      const columns = integer(clip.columns, 1, 16, 'atlas columns');
      const frames = integer(clip.frames, 2, 600, 'frame count');
      const fps = integer(clip.fps, 1, 60, 'frame rate');
      pixels += width * height * columns * Math.ceil(frames / columns) * 3;
      if (pixels > MAX_TOTAL_PIXELS) throw new Error('This comparison contains too many image pixels.');
      const images = {};
      for (const key of KEYS) images[key] = mediaSource(clip.media && clip.media[key], local);
      return {id, title: boundedText(clip.title, id, 200), description: boundedText(clip.description, '', 2000), split: clip.split,
        source_id: boundedText(clip.source_id, '', 100), source_title: boundedText(clip.source_title, '', 300), creator: boundedText(clip.creator, '', 100),
        source_url: safeCredit(clip.source_url), width, height, columns, frames, fps, images,
        diagnostic_frame: clip.diagnostic_frame == null ? null : integer(clip.diagnostic_frame, 0, frames - 1, 'diagnostic frame')};
    });
    return {title: boundedText(raw.title, local ? 'Local evaluation' : 'Public examples', 200), checkpoints, clips, local};
  }
  function frameRect(clip, frame, center = false) {
    const index = integer(frame, 0, clip.frames - 1, 'frame index');
    const w = center ? clip.width / 2 : clip.width;
    const h = center ? clip.height / 2 : clip.height;
    return {x: (index % clip.columns) * clip.width + (center ? clip.width / 4 : 0),
      y: Math.floor(index / clip.columns) * clip.height + (center ? clip.height / 4 : 0), width: w, height: h};
  }
  function nextFrame(frame, delta, frames) { return ((frame + delta) % frames + frames) % frames; }
  function matchingAtlas(image, clip) {
    return image.naturalWidth === clip.width * clip.columns && image.naturalHeight === clip.height * Math.ceil(clip.frames / clip.columns);
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {validateManifest, mediaSource, safeCredit, frameRect, nextFrame, matchingAtlas, MAX_FILE_BYTES};
  if (typeof document === 'undefined') return;

  const $ = id => document.getElementById(id);
  const text = (id, value) => { $(id).textContent = value; };
  let dataset = null, selected = null, images = null, frame = 0, playing = false, last = 0;
  let imageGeneration = 0, sourceGeneration = 0, publicManifest = null;
  function pause() { playing = false; text('play', '▶ Play'); $('play').setAttribute('aria-label', 'Play comparison'); }
  function setControls(enabled) { for (const id of ['play', 'back', 'next', 'scrub', 'anchor']) $(id).disabled = !enabled; }
  function draw() {
    if (!images || !selected) return;
    const area = frameRect(selected, frame, $('crop').value === 'center');
    for (const key of KEYS) {
      const canvas = $(`${key}-frame`); canvas.width = selected.width; canvas.height = selected.height;
      const context = canvas.getContext('2d'); context.imageSmoothingEnabled = false;
      context.drawImage(images[key], area.x, area.y, area.width, area.height, 0, 0, canvas.width, canvas.height);
    }
    $('scrub').value = frame;
    const time = (frame / selected.fps).toFixed(1);
    $('scrub').setAttribute('aria-valuetext', `Frame ${frame + 1} of ${selected.frames}, ${time} seconds`);
    text('frame-count', `${frame + 1} / ${selected.frames} · ${time}s`);
  }
  function step(delta) { if (!images) return; pause(); frame = nextFrame(frame, delta, selected.frames); draw(); }
  function togglePlayback() {
    if (!images) return;
    if (playing) pause();
    else { playing = true; last = performance.now(); text('play', 'Ⅱ Pause'); $('play').setAttribute('aria-label', 'Pause comparison'); }
  }
  function animate(now) {
    if (playing && images) {
      const period = 1000 / (selected.fps * Number($('speed').value));
      const elapsed = Math.floor((now - last) / period);
      if (elapsed > 0) { frame = nextFrame(frame, elapsed, selected.frames); last += elapsed * period; draw(); }
    }
    requestAnimationFrame(animate);
  }
  function loadImage(src, clip) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => matchingAtlas(image, clip) ? resolve(image) : reject(new Error('The comparison image dimensions do not match its frame description.'));
      image.onerror = () => reject(new Error('A comparison image could not be decoded. Choose another clip or reopen the file.'));
      image.src = src;
    });
  }
  async function choose(clip) {
    const token = ++imageGeneration; pause(); setControls(false); images = null; selected = clip; frame = 0;
    $('reconstruction-viewer').setAttribute('aria-busy', 'true');
    for (const key of KEYS) { const c = $(`${key}-frame`); c.getContext('2d').clearRect(0, 0, c.width, c.height); }
    $('anchor').hidden = clip.diagnostic_frame == null;
    $('clip-select').value = clip.id; $('scrub').max = clip.frames - 1; $('scrub').value = 0;
    text('load-status', 'Loading synchronized frames…'); text('clip-title', clip.title); text('description', clip.description);
    text('frame-count', `— / ${clip.frames}`); text('split-badge', clip.split === 'val' ? 'Held-out validation' : 'Training example');
    text('timing', `${clip.frames} frames · ${clip.fps} FPS · ${(clip.frames / clip.fps).toFixed(1)}s loop · First-to-last timestamp: ${((clip.frames - 1) / clip.fps).toFixed(1)}s${clip.diagnostic_frame == null ? '' : ` · Scored anchor: frame ${clip.diagnostic_frame + 1}`}`);
    $('source-link').hidden = !clip.source_url;
    if (clip.source_url) { $('source-link').href = clip.source_url; text('source-link', `${clip.source_title || clip.source_id} ↗`); }
    else $('source-link').removeAttribute('href');
    text('source-meta', [clip.creator, clip.source_id, clip.split === 'val' ? 'Validation source' : 'Training source'].filter(Boolean).join(' · '));
    try {
      const loaded = await Promise.all(KEYS.map(key => loadImage(clip.images[key], clip)));
      if (token !== imageGeneration) return;
      images = Object.fromEntries(KEYS.map((key, index) => [key, loaded[index]]));
      draw(); setControls(true); text('load-status', '');
    } catch (error) { if (token === imageGeneration) text('load-status', error.message); }
    finally { if (token === imageGeneration) $('reconstruction-viewer').setAttribute('aria-busy', 'false'); }
  }
  function activate(manifest, filename = '') {
    dataset = manifest;
    $('clip-select').replaceChildren(...dataset.clips.map((clip, index) => { const option = document.createElement('option'); option.value = clip.id; option.textContent = `${String(index + 1).padStart(2, '0')} · ${clip.title}`; return option; }));
    $('clip-select').disabled = false;
    text('dataset-kind', dataset.local ? 'LOCAL FILE · NEVER UPLOADED' : 'PUBLIC EXAMPLES');
    text('dataset-title', `${dataset.title} · ${dataset.clips.length} clips${filename ? ` · ${filename}` : ''}`);
    $('public-examples').hidden = !dataset.local;
    for (const key of ['before', 'after']) {
      const checkpoint = dataset.checkpoints[key]; text(`${key}-label`, checkpoint.label || `STEP ${checkpoint.step}`);
      text(`${key}-description`, checkpoint.preflight ? 'Diagnostic tokenizer checkpoint' : 'Tokenizer checkpoint');
      $(`${key}-frame`).setAttribute('aria-label', `${checkpoint.label || `Step ${checkpoint.step}`} reconstruction`);
    }
    choose(dataset.clips[0]);
  }
  async function openPublic() {
    const token = ++sourceGeneration; pause(); text('load-status', 'Loading public examples…');
    try {
      if (!publicManifest) {
        const response = await fetch('reconstructions.json', {credentials: 'omit'});
        if (!response.ok) throw new Error('The public comparison library is unavailable. Please reload the page.');
        const declaredSize = Number(response.headers.get('content-length') || 0);
        if (declaredSize > MAX_FILE_BYTES) throw new Error('The public comparison file is too large.');
        const source = await response.text();
        if (source.length > MAX_FILE_BYTES) throw new Error('The public comparison file is too large.');
        publicManifest = validateManifest(JSON.parse(source), false);
      }
      if (token === sourceGeneration) activate(publicManifest);
    } catch (error) {
      if (token !== sourceGeneration) return;
      text('load-status', error.message); $('reconstruction-viewer').setAttribute('aria-busy', 'false');
      if (!dataset) { text('dataset-title', 'Public library unavailable · You can still open a local file'); text('clip-title', 'Choose a comparison file'); }
    }
  }
  $('local-file').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    const token = ++sourceGeneration; pause();
    text('load-status', 'Reading the local comparison file in this browser…');
    try {
      if (file.size > MAX_FILE_BYTES) throw new Error('The comparison file must be smaller than 200 MiB.');
      const manifest = validateManifest(JSON.parse(await file.text()), true);
      if (token === sourceGeneration) activate(manifest, file.name);
    } catch (error) {
      if (token === sourceGeneration) text('load-status', error instanceof SyntaxError ? 'This file is not valid comparison JSON. Choose the exported reconstruction comparison file.' : error.message);
    } finally { event.target.value = ''; }
  });
  $('public-examples').addEventListener('click', openPublic);
  $('clip-select').addEventListener('change', () => { const clip = dataset.clips.find(item => item.id === $('clip-select').value); if (clip) choose(clip); });
  $('anchor').addEventListener('click', () => { if (!images || selected.diagnostic_frame == null) return; pause(); frame = selected.diagnostic_frame; draw(); });
  $('play').addEventListener('click', togglePlayback); $('back').addEventListener('click', () => step(-1)); $('next').addEventListener('click', () => step(1));
  $('scrub').addEventListener('input', () => { if (!images) return; pause(); frame = Number($('scrub').value); draw(); });
  $('speed').addEventListener('change', () => { last = performance.now(); }); $('crop').addEventListener('change', draw);
  $('reconstruction-viewer').addEventListener('keydown', event => {
    if (event.target.closest('input, select, textarea, button, a') || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); step(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); step(1); }
    if (event.key === ' ') { event.preventDefault(); togglePlayback(); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  openPublic(); requestAnimationFrame(animate);
})();
