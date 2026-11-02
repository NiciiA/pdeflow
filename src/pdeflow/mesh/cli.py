"""Thin commands for the same local/Colab model family, lifecycle and comparison."""
from pathlib import Path
import argparse,json
from pdeflow.runtime import read_json,write_json
from .contracts import MeshConfig,CANDIDATES,preset
from .session import MeshSession,plan
from .suite import Suite,suite_plan


def architecture(value):
    aliases={f'M{i+1}':name for i,name in enumerate(CANDIDATES)}
    result=aliases.get(value.upper(),value)
    if result not in CANDIDATES:raise argparse.ArgumentTypeError('Choose M1–M5 or an exact mesh architecture ID')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='PDEFlow observation-informed physical mesh candidates; no AE latent targets');sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('status');a.add_argument('--run',required=True);a.add_argument('--watch',action='store_true');a.add_argument('--refresh',type=float,default=5.)
    a=sub.add_parser('health');a.add_argument('--checkpoint',required=True)
    a=sub.add_parser('replay');a.add_argument('--failure',required=True);a.add_argument('--device',default='cpu');a.add_argument('--output',required=True)
    a=sub.add_parser('corrected-continuation');a.add_argument('--parent-run',required=True);a.add_argument('--local-root',required=True);a.add_argument('--persistent-root');a.add_argument('--device')
    a=sub.add_parser('plan');a.add_argument('--candidate',type=architecture,default='mesh_m1_v1');a.add_argument('--capacity',default='pipeline-check');a.add_argument('--updates',type=int);a.add_argument('--device',default='cpu');a.add_argument('--precision',default='auto');a.add_argument('--output')
    for command in ('train','preflight'):
        a=sub.add_parser(command);a.add_argument('--config',required=True);a.add_argument('--output',required=True);a.add_argument('--persistent');a.add_argument('--stop-after',type=int)
    a=sub.add_parser('resume');a.add_argument('--run',required=True);a.add_argument('--persistent');a.add_argument('--device');a.add_argument('--additional-updates',type=int);a.add_argument('--stage')
    a=sub.add_parser('stop');a.add_argument('--run',required=True)
    a=sub.add_parser('export');a.add_argument('--run',required=True);a.add_argument('--kind',choices=['inference','diagnostic','resumable'],default='inference')
    a=sub.add_parser('restore');a.add_argument('--persistent-root',required=True);a.add_argument('--run-id',required=True);a.add_argument('--local-root',required=True);a.add_argument('--snapshot');a.add_argument('--device',default='cpu')
    a=sub.add_parser('import');a.add_argument('--checkpoint',required=True);a.add_argument('--workspace')
    a=sub.add_parser('reconstruct');a.add_argument('--checkpoint',required=True);a.add_argument('--problem',required=True);a.add_argument('--observations',required=True);a.add_argument('--output',required=True);a.add_argument('--queries');a.add_argument('--particles',type=int,default=4);a.add_argument('--steps',type=int,default=16);a.add_argument('--seed',type=int,default=91);a.add_argument('--device',default='cpu')
    a=sub.add_parser('cases');a.add_argument('--config',required=True);a.add_argument('--output',required=True);a.add_argument('--split',choices=['validation','test'],default='validation');a.add_argument('--count',type=int,default=8)
    a=sub.add_parser('compare');a.add_argument('--checkpoint',action='append',required=True,help='LABEL=PATH; repeat for candidates/legacy');a.add_argument('--cases',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu');a.add_argument('--particles',type=int,default=4);a.add_argument('--steps',type=int,default=8)
    a=sub.add_parser('suite-plan');a.add_argument('--capacity',default='medium');a.add_argument('--candidate',action='append',type=architecture);a.add_argument('--device',default='cuda');a.add_argument('--updates',type=int);a.add_argument('--include-legacy',action='store_true');a.add_argument('--output',required=True)
    a=sub.add_parser('suite-new');a.add_argument('--plan',required=True);a.add_argument('--local-root',required=True);a.add_argument('--persistent-root')
    for command in ('suite-run','suite-preflight','suite-stop','suite-inspect'):
        a=sub.add_parser(command);a.add_argument('--root',required=True);a.add_argument('--persistent');a.add_argument('--candidate',action='append')
    a=sub.add_parser('suite-restore');a.add_argument('--persistent-root',required=True);a.add_argument('--run-id',required=True);a.add_argument('--local-root',required=True);a.add_argument('--snapshot')
    args=p.parse_args(argv);command=args.command
    if command=='status':
        from .monitoring import status,text_summary
        if args.refresh<1:raise ValueError('CLI refresh must be at least1 second')
        while True:
            result=status(args.run)
            if not args.watch:break
            print(text_summary(result),flush=True)
            try:__import__('time').sleep(args.refresh)
            except KeyboardInterrupt:return
        print(json.dumps(result,indent=2,allow_nan=False));return
    elif command=='health':
        from .diagnostics import checkpoint_health
        result=checkpoint_health(args.checkpoint)
    elif command=='replay':
        from .diagnostics import replay_failure
        result=replay_failure(args.failure,args.device);write_json(args.output,result)
    elif command=='corrected-continuation':
        from .recovery import corrected_continuation
        session=corrected_continuation(args.parent_run,args.local_root,args.persistent_root,args.device);result={'new_root':str(session.root),'persistent':str(session.persistent),'parent_preserved':True,'not_exact_resume':True,'next':'Preflight the new source, then use resume on this NEW run root'}
    elif command=='plan':
        overrides={'budgets':{s:args.updates for s in MeshConfig().budgets}} if args.updates else None;c=preset(args.capacity,args.candidate,args.device,args.precision,overrides);result=plan(c)
        if args.output:write_json(args.output,c.to_dict())
    elif command in ('train','preflight'):
        session=MeshSession(read_json(args.config),args.output,args.persistent);result=session.preflight() if command=='preflight' else session.run(stop_after=args.stop_after)
    elif command=='resume':
        session=MeshSession.reopen(args.run,args.persistent,args.device);result=session.extend(args.stage or session.c.kind['stages'][-1],args.additional_updates) if args.additional_updates else session.run()
    elif command=='stop':MeshSession.reopen(args.run).stop();result={'state':'stop requested at next safe boundary'}
    elif command=='export':
        from .artifacts import archive
        session=MeshSession.reopen(args.run);session.finalize();result={'archive':str(archive(session.root,args.kind))}
    elif command=='restore':result={'root':str(MeshSession.from_store(args.persistent_root,args.run_id,args.local_root,args.snapshot,args.device).root)}
    elif command=='import':
        from .registry import register_checkpoint
        result=register_checkpoint(args.checkpoint,args.workspace)
    elif command=='reconstruct':
        from .inference import reconstruct
        result=reconstruct(args.checkpoint,read_json(args.problem),read_json(args.observations),args.output,read_json(args.queries) if args.queries else None,args.steps,args.particles,args.seed,device=args.device)
    elif command=='cases':
        from .evaluation import create_cases
        result=create_cases(read_json(args.config),args.output,args.count,args.split)
    elif command=='compare':
        from .evaluation import compare
        checkpoints={}
        for entry in args.checkpoint:
            name,path=entry.split('=',1)
            if not name.replace('-','').replace('_','').isalnum():raise ValueError('Plain comparison label required')
            checkpoints[name]=path
        compare(checkpoints,args.cases,args.output,args.device,args.particles,args.steps);result={'report':str(Path(args.output)/'comparison.json'),'accuracy_claim':False}
    elif command=='suite-plan':
        overrides={'budgets':{s:args.updates for s in MeshConfig().budgets}} if args.updates else None;result=suite_plan(args.capacity,args.device,candidates=args.candidate,overrides=overrides,include_legacy=args.include_legacy);write_json(args.output,result)
    elif command=='suite-new':result={'root':str(Suite.new(read_json(args.plan),args.local_root,args.persistent_root).root)}
    elif command=='suite-restore':result={'root':str(Suite.from_store(args.persistent_root,args.run_id,args.local_root,args.snapshot).root)}
    else:
        suite=Suite(args.root,args.persistent)
        if command=='suite-run':result=suite.run(args.candidate)
        elif command=='suite-preflight':result=suite.preflight(candidate=args.candidate[0] if args.candidate else None)
        elif command=='suite-stop':suite.stop();result={'state':'safe stop requested'}
        else:result=suite.inspect()
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
