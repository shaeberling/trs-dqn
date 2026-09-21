// Portable diagnostic replay codec and controls; no browser/network required.
// node tests/test_defense_replay.js [results/defense/random-baseline/replay.html]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2] || 'results/defense/random-baseline/replay.html', 'utf8');
const script = html.match(/<script>([\s\S]+?)<\/script>/)[1];
const elements = new Map();
let draws = 0;
const context = vm.createContext({
  Uint8Array, DataView,
  atob: s => Buffer.from(s, 'base64').toString('binary'),
  Image: class { complete = true; },
  requestAnimationFrame: () => {},
  document: { getElementById: id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'speed' ? '1' : '0',
      getContext: () => ({ drawImage: () => { draws++; } }),
    });
    return elements.get(id);
  } },
});
vm.runInContext(script, context);
const read = expression => vm.runInContext(expression, context);
assert.equal(read('frames.length'), read('metadata.result.steps + 1'));
assert.equal(read('actions.length'), read('metadata.verified_actions'));
assert(read('actions.every(a => Number.isInteger(a) && a >= 0 && a < metadata.action_names.length)'));
assert(read('frames.every(f => f.length === 1024)'));
assert(read('metadata.result.terminated'));
assert.equal(typeof read('metadata.trained_model'), 'boolean');
if (read('metadata.trained_model')) {
  assert(html.includes('Trained screen-only neural policy'));
  assert.match(read('metadata.checkpoint_sha256'), /^[0-9a-f]{64}$/);
  if (read('metadata.evaluation_only')) {
    assert(html.includes('Evaluation-only sampling probe'));
    assert(html.includes(`temperature ${read('metadata.temperature')}`));
    assert.equal(read('metadata.promotion_eligible'), false);
  }
} else {
  assert(html.includes('not a trained agent'));
}
const final = Buffer.from(read('frames.at(-1)')).toString('latin1');
assert.equal(Number(final.slice(0, 16).trim()), read('metadata.result.score'));
assert(final.slice(448, 512).includes('GAME OVER PLAYER 1'));
read('atlas.onload()');
assert.equal(draws, 1024);
elements.get('play').onclick();
read('tick(100); tick(200)');
assert(read('index > 0'));
elements.get('play').onclick();
assert.equal(read('playing'), false);
elements.get('seek').value = '42';
elements.get('seek').oninput();
assert.equal(read('index'), 42);
elements.get('speed').value = '4';
elements.get('play').onclick();
read('tick(300); tick(400)');
assert(read('index > 43'));
read('index = frames.length - 1; playing = false');
elements.get('play').onclick();
assert.equal(read('index'), 0);
assert.equal(read('playing'), true);
console.log(`Defense replay verified: ${read('frames.length')} frames, score ${read('metadata.result.score')}; controls pass.`);
