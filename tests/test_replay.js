// Replay codec and control-logic check; no browser or network required.
// Usage: node tests/test_replay.js [results/replay.html]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(process.argv[2] || 'results/replay.html', 'utf8');
const script = html.match(/<script>([\s\S]+?)<\/script>/)[1];
const elements = new Map();
let draws = 0;
const context = vm.createContext({
  Uint8Array,
  atob: s => Buffer.from(s, 'base64').toString('binary'),
  Image: class {},
  requestAnimationFrame: () => {},
  document: { querySelector: id => {
    if (!elements.has(id)) elements.set(id, {
      value: id === '#speed' ? '1' : '0',
      getContext: () => ({ drawImage: () => { draws++; } }),
    });
    return elements.get(id);
  } },
});
vm.runInContext(script, context);
const read = expression => vm.runInContext(expression, context);
assert.equal(read('frames.length'), read('metadata.frames'));
assert.equal(read('actions.length'), read('frames.length - 1'));
assert.equal(read('actions.length'), read('metadata.steps'));
assert(read('actions.every(a => Number.isInteger(a) && a >= 0 && a < 6)'));
assert(read('frames.every(f => f.length === 1024)'));
assert(read('metadata.terminated'));
const final = Buffer.from(read('frames[frames.length - 1]')).toString('latin1')
  .replace(/[^\x20-\x7e]/g, ' ');
assert.equal(Number(final.slice(6, 11)), read('metadata.score'));
assert(final.slice(640, 704).includes('GAME OVER'));
assert(read('frames.some(f => Number(String.fromCharCode(...f.slice(59, 64))) >= 2)'));

read('atlas.onload()');
assert.equal(draws, 1024);
read('tick(100); tick(200)');
assert(read('current > 0'));
elements.get('#play').onclick();
assert.equal(read('playing'), false);
elements.get('#seek').value = '42';
elements.get('#seek').oninput();
assert.equal(read('current'), 42);
elements.get('#restart').onclick();
assert.equal(read('current'), 0);
assert.equal(read('playing'), true);
elements.get('#speed').value = '4';
read('tick(300)');
assert(read('current > 1'));
console.log(`Replay verified: ${read('frames.length')} frames; score ${read('metadata.score')}; play/pause/seek/restart/speed pass.`);
