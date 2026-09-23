// Exercise generated page navigation, persistence and quoted CSV without a browser dependency.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const nodes=new Map();let exported;
function node(tag){return {tag,children:[],value:'',textContent:'',append(...xs){this.children.push(...xs)},replaceChildren(){this.children=[]},click(){},play(){},pause(){}}}
const context={document:{getElementById(id){if(!nodes.has(id))nodes.set(id,node(id));return nodes.get(id)},createElement:node},localStorage:{values:{},getItem(k){return this.values[k]},setItem(k,v){this.values[k]=v}},Blob:class {constructor(parts){exported=parts.join('')}},URL:{createObjectURL(){return 'blob:test'},revokeObjectURL(){}},console};
vm.createContext(context);vm.runInContext(script,context);
vm.runInContext(`
if(items.length<1)throw Error('Need generated review clips');
const first=items[0].clips[0].blind_id;
save(first,'initial_attribute','CORRECT');
save(first,'notes','red, "jacket"');
move(1);move(-1);download();
`,context);
assert(exported.startsWith('\ufeff"blind_id"'));
assert(exported.includes('"red, ""jacket"""'));
assert(exported.includes('"CORRECT"'));
assert(!html.includes('initial_color_proxy'));
console.log('Review navigation, label persistence, CSV quoting and hidden scores passed.');
