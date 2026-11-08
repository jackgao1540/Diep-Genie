'use strict';
const $ = id => document.getElementById(id);
const state = {clips:[],feedback:{},view:'featured',filtered:[],page:0,selected:null,mode:'proxy',frame:0,atlas:null,playing:false,last:0,load:0,dirty:false,token:''};
const descriptions = {
  featured:['Reviewed examples','Start with clips examined for scene and behavior coverage. This is a targeted selection, not a random sample.'],
  accepted:['Retained footage','Browse the current retained training and validation footage. Quality remains sample-backed.'],
  held:['Source hold','160 clips from one source are withheld from the current dataset. Inspect them here without reinstating them.'],
  rejected:['Screening rejects','Four low-motion screening rejects, kept for comparison. Some show active gameplay; one contains added text.'],
  flagged:['Your flags','Clips you flagged for a closer look. Feedback does not change the frozen dataset.'],
  reviewed:['Your reviews','Your saved quality decisions and notes, stored on this computer.']
};
const labels = {good:'Looks good',flag:'Flagged issue',unsure:'Unsure',clear:'Not reviewed'};
const time = seconds => `${Math.floor(seconds/60).toString().padStart(2,'0')}:${Math.floor(seconds%60).toString().padStart(2,'0')}`;
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function notice(message='',error=false){$('notice').textContent=message;$('notice').classList.toggle('error',error);}
function decision(id){const r=state.feedback[id];return r&&r.decision!=='clear'?r:null;}
function stats(){
  const reviewed=Object.values(state.feedback).filter(r=>r.decision!=='clear');
  $('progress').textContent=reviewed.length.toLocaleString();$('count-reviewed').textContent=reviewed.length;
  $('count-flagged').textContent=reviewed.filter(r=>r.decision==='flag').length;
  $('progress-detail').textContent=`${reviewed.filter(r=>r.decision==='flag').length} flagged · saved on this computer`;
}
function applyFilters(reset=true){
  if(reset)state.page=0;
  const query=$('search').value.toLowerCase().trim(),split=$('split').value,category=$('category').value,source=$('source').value;
  state.filtered=state.clips.filter(c=>{
    const view=state.view;
    if(view==='featured'&&(!c.reviewed||c.status!=='accepted'))return false;
    if(['accepted','held','rejected'].includes(view)&&c.status!==view)return false;
    if(view==='flagged'&&decision(c.id)?.decision!=='flag')return false;
    if(view==='reviewed'&&!decision(c.id))return false;
    return (split==='all'||split===c.split)&&(category==='all'||c.tags.includes(category))&&(source==='all'||source===c.source_id)&&(!query||`${c.title} ${c.channel} ${c.source_id}`.toLowerCase().includes(query));
  });
  if(state.view==='held')state.filtered.sort((a,b)=>Number(b.tags.includes('Suspected freeze'))-Number(a.tags.includes('Suspected freeze'))||a.start-b.start);
  const pages=Math.max(1,Math.ceil(state.filtered.length/18));state.page=Math.min(state.page,pages-1);
  $('collection-title').textContent=descriptions[state.view][0];$('collection-description').textContent=descriptions[state.view][1];
  $('results').textContent=`${state.filtered.length.toLocaleString()} clips`;$('page-label').textContent=`${state.page+1} / ${pages}`;
  $('page-prev').disabled=state.page===0;$('page-next').disabled=state.page>=pages-1;$('empty').hidden=!!state.filtered.length;$('shuffle').disabled=!state.filtered.length;
  renderCards();
}
function renderCards(){
  const visible=state.filtered.slice(state.page*18,state.page*18+18);
  $('grid').innerHTML=visible.map(c=>`<button class="clip-card ${c.id===state.selected?.id?'current':''}" data-id="${c.id}" aria-label="Review ${escape(c.title)} at ${time(c.start)}"><div class="clip-thumb"><img loading="lazy" src="/api/image/${c.id}/thumb" alt="Gameplay thumbnail at ${time(c.start)}"></div><span class="card-duration">4 SEC</span><div class="card-text"><strong>${escape(c.channel)}</strong><p>${escape(c.source_id)} · ${time(c.start)}</p><span class="card-status ${c.status}">${escape(decision(c.id)?labels[decision(c.id).decision]:c.status==='accepted'?(c.split==='train'?'Training':'Validation'):c.status==='held'?'On hold':'Rejected')} ${c.tags.includes('Large pentagons')?'· Large pentagons':c.tags.includes('Maze walls')?'· Maze':''}</span></div></button>`).join('');
  $('grid').querySelectorAll('[data-id]').forEach(button=>button.onclick=()=>selectClip(button.dataset.id));
}
function reviewFields(){
  const review=state.feedback[state.selected.id];$('note').value=review?.note||'';state.dirty=false;
  $('saved-state').textContent=review&&review.decision!=='clear'?'Saved · '+labels[review.decision]:'Not reviewed';
  document.querySelectorAll('[data-decision]').forEach(b=>b.classList.toggle('chosen',b.dataset.decision===review?.decision));
}
async function saveReview(value,showNotice=true){
  if(!state.selected)return false;
  const id=state.selected.id,note=value==='clear'?'':$('note').value;
  try{
    const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':state.token},body:JSON.stringify({id,decision:value,note})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Could not save review');
    state.feedback[id]=result;
    if(state.selected.id===id)reviewFields();stats();applyFilters(false);
    if(showNotice)notice(value==='clear'?'Review cleared. Previous history is retained locally.':'Review saved on this computer.');
    return true;
  }catch(error){notice('Your review was not saved: '+error.message,true);return false;}
}
async function flushNote(){return !state.dirty||await saveReview(decision(state.selected?.id)?.decision||'unsure',false);}
async function selectClip(id,continuePlaying=false){
  if(!await flushNote())return;
  const clip=state.clips.find(c=>c.id===id);if(!clip)return;
  state.selected=clip;state.frame=0;state.playing=false;state.atlas?.close();state.atlas=null;$('review-layout').hidden=false;
  state.mode=clip.has_proxy?state.mode:'original';$('proxy-mode').disabled=!clip.has_proxy;
  $('clip-title').textContent=clip.title;$('clip-byline').textContent=`${clip.channel} · ${clip.source_id} · source ${time(clip.start)}–${time(clip.start+4)}`;
  $('split-badge').textContent=clip.split==='train'?'Training':'Validation';$('clip-status').textContent={accepted:'Retained',held:'Source hold',rejected:'Rejected'}[clip.status];$('clip-status').className='pill '+clip.status;
  $('reason').textContent=clip.reason;$('tags').innerHTML=clip.tags.map(t=>`<span class="tag">${escape(t)}</span>`).join('');
  $('evidence').innerHTML=clip.evidence.length?clip.evidence.map(e=>`<p>${escape(e)}</p>`).join(''):'<p>No clip-specific semantic annotation. Retention follows the existing screening and cohort review.</p>';
  $('timestamp').textContent=`${clip.start.toFixed(3)} seconds`;$('block-id').textContent=clip.block_id;
  $('sequence-label').textContent=clip.status==='accepted'?`${clip.sequence_seconds}s checked sequence`:'Not in current sequences';
  $('previous-block').disabled=!clip.previous;$('next-block').disabled=!clip.next;$('follow').disabled=clip.status!=='accepted';
  reviewFields();renderCards();notice();await loadAtlas();if(continuePlaying&&state.selected.id===id&&state.atlas){state.playing=true;state.last=performance.now();updatePlay();}
}
async function loadAtlas(){
  const load=++state.load,clip=state.selected,mode=state.mode;state.playing=false;state.atlas?.close();state.atlas=null;updatePlay();$('loading').hidden=false;$('loading').textContent=mode==='original'?'Loading original crop…':'Loading exact training frames…';
  $('proxy-mode').classList.toggle('selected',mode==='proxy');$('original-mode').classList.toggle('selected',mode==='original');
  const w=mode==='proxy'?128:clip.crop[2],h=mode==='proxy'?128:clip.crop[3];
  $('resolution').textContent=`${w} × ${h} · ${clip.fps} FPS`;
  try{
    const response=await fetch(`/api/image/${clip.id}/${mode}`);
    if(!response.ok){const error=await response.json();throw Error(error.error||'Frame load failed');}
    const bitmap=await createImageBitmap(await response.blob());
    if(load!==state.load){bitmap.close();return;}
    if(bitmap.width!==w*8||bitmap.height!==h*5){bitmap.close();throw Error('Unexpected frame layout');}
    state.atlas=bitmap;$('canvas').width=w;$('canvas').height=h;$('loading').hidden=true;draw();
  }catch(error){if(load!==state.load)return;$('loading').textContent='Unable to load clip';notice(error.message,true);}
}
function draw(){
  if(!state.atlas||!state.selected)return;
  const canvas=$('canvas'),context=canvas.getContext('2d');context.imageSmoothingEnabled=false;
  context.drawImage(state.atlas,(state.frame%8)*canvas.width,Math.floor(state.frame/8)*canvas.height,canvas.width,canvas.height,0,0,canvas.width,canvas.height);
  $('scrub').value=state.frame;$('elapsed').textContent=`${(state.frame/10).toFixed(1)} / 4.0s`;
  $('frame-label').textContent=`FRAME ${String(state.frame+1).padStart(2,'0')} / 40`;$('source-time').textContent=`SOURCE ${time(state.selected.start+state.frame/10)}`;
}
function updatePlay(){$('play').textContent=state.playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',state.playing?'Pause clip':'Play clip');}
function play(){if(!state.atlas)return;state.playing=!state.playing;state.last=performance.now();updatePlay();}
function step(amount){state.playing=false;state.frame=Math.max(0,Math.min(39,state.frame+amount));updatePlay();draw();}
function tick(now){
  if(state.playing&&state.atlas){
    const delay=100/Number($('speed').value),elapsed=now-state.last;
    if(elapsed>=delay){const advance=Math.floor(elapsed/delay);state.last+=advance*delay;
      if(state.frame+advance>=40&&$('follow').checked){state.playing=false;updatePlay();if(state.selected.next)selectClip(state.selected.next,true);else{state.frame=39;draw();notice('Reached the end of this checked sequence. Playback stops at the gap.');}}
      else{state.frame=(state.frame+advance)%40;draw();}
    }
  }requestAnimationFrame(tick);
}
async function changeView(view){if(!await flushNote())return;state.view=view;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));applyFilters();if(state.filtered.length)await selectClip(state.filtered[0].id);else{state.playing=false;updatePlay();$('review-layout').hidden=true;}}
async function context(){
  const clip=state.selected;if(!clip)return;$('context-caption').textContent=`${clip.channel} · ${clip.source_id} · ${time(clip.start+2)}`;
  $('context-image').removeAttribute('src');$('context-image').alt='Loading full source frame…';$('context-dialog').showModal();
  try{const response=await fetch(`/api/image/${clip.id}/context`);if(!response.ok)throw Error('Could not load full source frame');const url=URL.createObjectURL(await response.blob());$('context-image').onload=()=>URL.revokeObjectURL(url);$('context-image').src=url;$('context-image').alt='Full source frame at the middle retained timestamp';}catch(error){$('context-caption').textContent=error.message;}
}
async function boot(){
  try{
    const response=await fetch('/api/catalog');if(!response.ok)throw Error('The local library could not be loaded.');const data=await response.json();state.clips=data.clips;state.feedback=data.feedback;state.token=data.token;
    $('hours').textContent=data.meta.accepted_hours.toFixed(2)+' h';$('sources').textContent=data.meta.accepted_sources;$('visible').textContent=state.clips.filter(c=>c.status==='accepted').length.toLocaleString();$('version').textContent=`${data.meta.cohort} · 10 FPS · 128px`;
    for(const view of ['accepted','held','rejected'])$('count-'+view).textContent=state.clips.filter(c=>c.status===view).length.toLocaleString();$('count-featured').textContent=state.clips.filter(c=>c.status==='accepted'&&c.reviewed).length;
    [...new Set(state.clips.flatMap(c=>c.tags))].sort().forEach(tag=>$('category').add(new Option(tag,tag)));
    const sources=[...new Map(state.clips.map(c=>[c.source_id,c])).values()].sort((a,b)=>a.channel.localeCompare(b.channel)||a.source_id.localeCompare(b.source_id));sources.forEach(c=>$('source').add(new Option(`${c.channel} · ${c.source_id}`,c.source_id)));
    stats();applyFilters();const preferred=state.filtered.find(c=>c.tags.includes('Maze walls')&&c.tags.includes('Visible firing'))||state.filtered[0];if(preferred)await selectClip(preferred.id);notice();
  }catch(error){notice(error.message,true);}
}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>changeView(b.dataset.view));
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>saveReview(b.dataset.decision));
async function filterChanged(){if(!await flushNote())return;applyFilters();if(!state.filtered.length){state.playing=false;updatePlay();$('review-layout').hidden=true;}else if(!state.filtered.some(c=>c.id===state.selected?.id)||$('review-layout').hidden){await selectClip(state.filtered[0].id);}}
for(const id of ['split','category','source'])$(id).onchange=filterChanged;$('search').oninput=filterChanged;
$('play').onclick=play;$('back-frame').onclick=()=>step(-1);$('next-frame').onclick=()=>step(1);$('scrub').oninput=()=>{state.playing=false;state.frame=Number($('scrub').value);updatePlay();draw();};
$('speed').onchange=()=>state.last=performance.now();$('proxy-mode').onclick=()=>{state.mode='proxy';loadAtlas();};$('original-mode').onclick=()=>{state.mode='original';loadAtlas();};
$('note').oninput=()=>{state.dirty=true;$('saved-state').textContent='Unsaved note';};$('save-note').onclick=()=>saveReview(decision(state.selected?.id)?.decision||'unsure');$('clear-review').onclick=()=>saveReview('clear');
$('previous-block').onclick=()=>state.selected.previous&&selectClip(state.selected.previous);$('next-block').onclick=()=>state.selected.next&&selectClip(state.selected.next);
$('shuffle').onclick=()=>{const candidates=state.filtered.filter(c=>c.id!==state.selected?.id);if(candidates.length)selectClip(candidates[Math.floor(Math.random()*candidates.length)].id);};
$('page-prev').onclick=()=>{state.page--;applyFilters(false);};$('page-next').onclick=()=>{state.page++;applyFilters(false);};
$('reset').onclick=()=>{$('search').value='';for(const id of ['split','category','source'])$(id).value='all';filterChanged();};
$('context').onclick=context;$('close-context').onclick=()=>$('context-dialog').close();
$('export').onclick=async()=>{if(!await flushNote())return;try{const response=await fetch('/api/reviews');if(!response.ok)throw Error('Export failed');const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download='genie-footage-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);notice('Review export downloaded.');}catch(error){notice(error.message,true);}};
window.addEventListener('beforeunload',event=>{if(state.dirty){event.preventDefault();event.returnValue='';}});
document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName)||$('context-dialog').open)return;if(event.code==='Space'){event.preventDefault();play();}if(event.code==='ArrowLeft'){event.preventDefault();step(-1);}if(event.code==='ArrowRight'){event.preventDefault();step(1);}});
requestAnimationFrame(tick);boot();
