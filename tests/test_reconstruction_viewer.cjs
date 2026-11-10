const test = require('node:test');
const assert = require('node:assert/strict');
const {validateManifest, mediaSource, safeCredit, frameRect, nextFrame, matchingAtlas} = require('../docs/reconstruction-viewer.js');
function manifest() {
  return {schema_version:1,kind:'diep-genie-reconstructions',title:'Example',checkpoints:{before:{step:219,preflight:false},after:{step:518,preflight:true}},clips:[{id:'sample-01',title:'A clip',split:'val',source_id:'abcdefghijk',fps:10,frames:40,width:128,height:128,columns:8,media:Object.fromEntries(['source','before','after'].map(k=>[k,{path:`assets/example-${k}.webp`,data_url:'data:image/webp;base64,UklGRgAAAABXRUJQ'}]))}]};
}
test('local images cannot trigger requests to remote or relative URLs', () => {
  for (const value of ['https://example.com/image.png','assets/image.png','data:image/svg+xml;base64,PHN2Zz4=']) assert.throws(()=>mediaSource({data_url:value},true));
  assert.throws(()=>mediaSource({path:'https://example.com/image.webp'},true));
  assert.equal(validateManifest(manifest(),true).clips[0].images.source,'data:image/webp;base64,UklGRgAAAABXRUJQ');
  assert.throws(()=>mediaSource({data_url:'data:image/png;base64,PHN2Zz4='},true));
});
test('public atlas paths cannot escape the explicit asset directory', () => {
  for (const path of ['../private.webp','assets/../private.webp','//example.com/a.webp','assets/%2e%2e/a.webp','https://example.com/a.webp']) assert.throws(()=>mediaSource({path},false));
  assert.equal(mediaSource({path:'assets/recon-sample-01-after.webp'},false),'assets/recon-sample-01-after.webp');
});
test('reject reserved-test inputs, duplicate IDs and excessive dimensions', () => {
  for (const mutate of [m=>m.clips[0].split='test',m=>m.clips.push(m.clips[0]),m=>m.clips[0].frames=10000,m=>m.clips[0].width=50000,m=>m.clips[0].diagnostic_frame=40]) {
    const m=manifest(); mutate(m); assert.throws(()=>validateManifest(m,false));
  }
});
test('all synchronized views use the same frame rectangle, including atlas row transitions and center crop', () => {
  const clip=validateManifest(manifest()).clips[0];
  assert.deepEqual(frameRect(clip,7),{x:896,y:0,width:128,height:128});
  assert.deepEqual(frameRect(clip,8),{x:0,y:128,width:128,height:128});
  assert.deepEqual(frameRect(clip,39,true),{x:928,y:544,width:64,height:64});
  assert.equal(nextFrame(39,1,40),0); assert.equal(nextFrame(0,-1,40),39);
  assert.throws(()=>frameRect(clip,40));
});
test('atlas dimensions must match declared frames, including partial rows', () => {
  const clip=validateManifest(manifest()).clips[0];
  assert.equal(matchingAtlas({naturalWidth:1024,naturalHeight:640},clip),true);
  assert.equal(matchingAtlas({naturalWidth:128,naturalHeight:128},clip),false);
  assert.equal(matchingAtlas({naturalWidth:1024,naturalHeight:384},{...clip,frames:17}),true);
});
test('source credits accept only explicit YouTube watch links', () => {
  assert.equal(safeCredit('javascript:alert(1)'),null);
  assert.equal(safeCredit('https://example.com/'),null);
  assert.equal(safeCredit('https://www.youtube.com/watch?v=abcdefghijk&t=12s'),'https://www.youtube.com/watch?v=abcdefghijk&t=12s');
});
test('optional checkpoint labels remain bounded text and old step-only manifests stay valid', () => {
  const old=validateManifest(manifest());
  assert.equal(old.checkpoints.before.step,219); assert.equal(old.checkpoints.after.step,518);
  assert.equal(old.checkpoints.before.label,''); assert.equal(old.checkpoints.after.label,'');
  const paired=manifest(); paired.checkpoints.before.label='Baseline'; paired.checkpoints.after.label='Detail loss';
  const result=validateManifest(paired,true);
  assert.equal(result.checkpoints.before.label,'Baseline'); assert.equal(result.checkpoints.after.label,'Detail loss');
  paired.checkpoints.before.label='<img src=x onerror=alert(1)>';
  assert.equal(validateManifest(paired,true).checkpoints.before.label,'<img src=x onerror=alert(1)>');
  // Labels use the existing textContent renderer, so markup remains literal text.
  for (const value of [123,{},'x'.repeat(61)]) {
    const invalid=manifest(); invalid.checkpoints.before.label=value;
    assert.throws(()=>validateManifest(invalid,true),/text is invalid or too long/);
  }
});
