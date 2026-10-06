import { Flare } from '@sauravpanda/flare';
const assert = (value, message) => { if (!value) throw new Error(message); };
const equal = (a,b,message) => assert(JSON.stringify(a)===JSON.stringify(b),message);
const rejects = async (promise,code) => { try { await promise; } catch(e) { assert(e.code===code, `${code}: ${e}`); return; } throw new Error(`Expected ${code}`); };
export async function decisionCI(real=false) {
  const reference=await (await fetch(`/decision/${real?'reference':'fixture-reference'}.json`)).json();
  const config={modelUrl: real?'/real.gguf':'/decision-fixture.gguf', tokenizerUrl:real?'/original-tokenizer.json':'/decision/tokenizer-reduced.json',backend:'cpu',cache:false};
  let flare;
  const report={backend:'browser WASM CPU',records:[],maxLogitError:0,maxScoreError:0};
  try {
    const start=performance.now();flare=await Flare.init(config);report.loadMs=performance.now()-start;
    if (real) globalThis.decisionLoaded = true;
    for(const record of reference.records) {
      const start=performance.now();
      let result;
      try { result=await flare.decide(record.request); }
      catch (error) {
        report.records.push({id:record.id,order:record.order,error:String(error),elapsedMs:performance.now()-start});
        if (!real) throw error;
        continue;
      }
      {
        report.records.push({id:record.id,order:record.order,result,elapsedMs:performance.now()-start});
        equal(result.prompt,record.prompt,'Independent template rendering');
        equal(result.promptIds,record.promptIds,'Independent prompt IDs');
        equal(result.scores.map(s=>s.tokenId),record.labelIds,'Independent label IDs');
        equal(result.scores.map(s=>s.choice),record.request.choices,'Choice order');
        assert(result.choice===record.request.choices[result.index],'Choice mapping');
        const max=Math.max(...record.logits), exps=record.logits.map(x=>Math.exp(x-max)),sum=exps.reduce((a,b)=>a+b,0);
        result.scores.forEach((score,i)=>{
          const logitError=Math.abs(score.logit-record.logits[i]),scoreError=Math.abs(score.score-exps[i]/sum);
          report.maxLogitError=Math.max(report.maxLogitError,logitError);report.maxScoreError=Math.max(report.maxScoreError,scoreError);
          assert(logitError<=reference.logitTolerance.absolute+reference.logitTolerance.relative*Math.abs(record.logits[i]),`${record.id}: reference candidate logit ${i}: ${logitError}`);
          assert(scoreError<=reference.scoreAbsoluteTolerance,`${record.id}: reference score ${i}: ${scoreError}`);
        });
      }
    }
    const successful = report.records.findIndex(record => record.result);
    if (successful < 0) { report.lifecycle = 'not exercised: no successful decision'; return report; }
    const request=reference.records[successful].request,first=report.records[successful].result;
    const checkRepeat=async()=>{ const r=await flare.decide(request);equal(r.scores,first.scores,'Deterministic repeated scores'); };
    await flare.reset();await checkRepeat();
    const chat=await flare.chat({message:'Hello',maxTokens:1});
    await checkRepeat();
    equal((await flare.chat({message:'Hello',maxTokens:1})).tokenIds,chat.tokenIds,'Decision to chat isolation');
    await checkRepeat();
    const controller=new AbortController();
    const pending=flare.decide({...request,signal:controller.signal});
    await rejects(flare.decide(request),'BUSY');await rejects(flare.reset(),'BUSY');
    // Let synchronous prefill start before terminating the worker.
    setTimeout(()=>controller.abort(), real?100:1);
    await rejects(pending,'ABORTED');await checkRepeat();
    await rejects(flare.decide({...request,signal:AbortSignal.abort()}),'ABORTED');
    const disposed=flare.decide(request);flare.dispose();await rejects(disposed,'DISPOSED');
    await rejects(flare.decide(request),'DISPOSED');
    flare=await Flare.init(config);
    await rejects(flare.decide({...request,state:'word '.repeat(2000)}),'DECIDE');
    await checkRepeat();
    await rejects(flare.decide({...request,choices:['same','same']}),'DECIDE');
    await checkRepeat();
    report.lifecycle='reset, repeat, decision/chat isolation, BUSY, in-flight abort/reload, pre-abort, dispose, overflow/error recovery passed';
    return report;
  } finally { flare?.dispose(); }
}
