const fs=require('fs');
const calls={arc:0,lineTo:0};
const ctx=new Proxy({}, {get:(t,k)=>{ if(k in t) return t[k]; return (...a)=>{ if(k in calls) calls[k]++; for(const x of a) if(typeof x==='number'&&!isFinite(x)) throw new Error('nonfinite arg to '+k); }; }, set:(t,k,v)=>{t[k]=v;return true;}});
const mk=(v)=>({value:v,textContent:'',disabled:false,listeners:{},addEventListener(e,f){this.listeners[e]=f;},style:{}});
const els={'w1-sampler':mk('ode'),'w1-nu':mk('0.7'),'w1-steps':mk('12'),'w1-cond':mk('-1'),'w1-w':mk('1'),'w1-reseed':mk(),'w1-play':mk(),'w1-out':mk(),'w1-nu-val':mk(),'w1-steps-val':mk(),'w1-w-val':mk()};
const canvas={getContext:()=>ctx,style:{},width:0,height:0};
const root={clientWidth:360,querySelector:(s)=> s==='canvas'?canvas:els[s.slice(1)]};
global.document={getElementById:()=>root,documentElement:{}};
global.getComputedStyle=()=>({getPropertyValue:()=>'#123456'});
global.window={matchMedia:()=>({matches:false,addEventListener(){}}),devicePixelRatio:2,addEventListener(){}};
global.ResizeObserver=class{constructor(f){this.f=f}observe(){this.f()}};
global.performance={now:()=>0};global.requestAnimationFrame=()=>1;global.cancelAnimationFrame=()=>{};
eval(fs.readFileSync('w1.js','utf8'));
console.log(canvas.width,canvas.height,calls, els['w1-out'].textContent);
for (const [k,v] of [['w1-sampler','sde'],['w1-steps','1'],['w1-steps','4'],['w1-cond','2'],['w1-w','4'],['w1-sampler','ode'],['w1-steps','1']]) { els[k].value=v; els[k].listeners.input(); console.log(k,v,'->',els['w1-out'].textContent); }
