"""Package entry points; notebooks only configure and launch these operations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pdeflow.core import ProblemSpec, ReconstructionResult
from pdeflow.specialists import get_specialist
from pdeflow.data import generate
from pdeflow.evaluation import export_result, metrics
from pdeflow.evaluation.baseline import inverse_baseline
from pdeflow.observations import ObservationSet
from pdeflow.runtime import Config, RuntimeSpec, environment, read_json, write_json
from pdeflow.sampling import sample
from pdeflow.training import _read_checkpoint, load_inference, train


def read_reference(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as a:
        return torch.from_numpy(a["values"].copy())


def reconstruct(config, checkpoint, problem_path, observations_path, output, reference_path=None, fine_checkpoint=None, mode=None):
    output = Path(output)
    if (output / "comparison.json").exists():
        raise FileExistsError(f"Reconstruction exists at {output}; choose a fresh directory")
    problem = ProblemSpec.from_dict(read_json(problem_path))
    obs = ObservationSet.from_dict(read_json(observations_path), problem.grid)
    specialist = get_specialist(config.specialist)
    specialist.validate(problem, config.formulation, config.inference.stepper, config.inference.schedule)
    bundle = load_inference(checkpoint, runtime=config.runtime, expected=config)
    reports = {}
    fine_bundle=load_inference(fine_checkpoint,runtime=config.runtime) if fine_checkpoint else None
    # Complete every inference before opening evaluation truth or plotting.
    results = {}
    modes=[mode] if mode else ["unguided","observation","observation_pde"]+(["importance","fk"] if config.inference.fk is not None else [])
    for mode in modes:
        results[mode] = sample(bundle, problem, obs, config.inference, mode, fine_bundle=fine_bundle)
    specialist.validate_observations(problem, obs)
    if len(modes)>1:results.update(specialist.baseline(problem, obs, config.inference))
    reference = read_reference(reference_path)
    for mode, result in results.items():
        final_problem=ProblemSpec.from_dict(result.provenance['final_problem']) if 'final_problem' in result.provenance else problem
        final_obs=ObservationSet.from_dict(result.provenance['final_observations'],final_problem.grid) if 'final_observations' in result.provenance else obs
        # Coarse truth must never be interpolated and called fine ground truth.
        final_reference=reference if final_problem.grid==problem.grid else None
        reports[mode] = export_result(output / mode, result, final_problem, final_obs, final_reference)
    write_json(output / "comparison.json", reports)
    write_json(output / "resolved_config.json", config.to_dict())
    write_json(output / "environment.json", environment(config.runtime))
    return {k: {name: value for name, value in report.items() if name in ("observation_rmse", "relative_l2_unobserved_mean", "pde_fd_rmse")}
            for k, report in reports.items()}


def evaluate(path, reference_path=None):
    path = Path(path)
    problem = ProblemSpec.from_dict(read_json(path / "problem.json"))
    obs = ObservationSet.from_dict(read_json(path / "observations.json"), problem.grid)
    previous = read_json(path / "metrics.json")
    with np.load(path / "result.npz", allow_pickle=False) as a:
        samples = torch.from_numpy(a["samples"].copy())
        truth = torch.from_numpy(a["reference"].copy()) if "reference" in a else None
        population_weights=torch.from_numpy(a["weights"].copy()) if "weights" in a else None
    if reference_path:
        truth = read_reference(reference_path)
    result = ReconstructionResult(samples, previous["cost"], previous["provenance"], previous["stop_reason"], previous["corrected"],weights=population_weights)
    report = metrics(result, problem, obs, truth)
    write_json(path / "metrics.json", report)
    return report


def smoke(config, output):
    output = Path(output)
    if config.provider:
        from pdeflow.data.providers import prepare
        evaluation=output/'evaluation';prepare(config,evaluation)
        data_path=output/'data' if config.provider.mode=='offline' else None
        if data_path:prepare(config,data_path,True)
        inputs=evaluation/'test/0'
    else:
        evaluation=None;data_path=output/'data';generate(config,data_path);inputs=data_path
    train(config,data_path,output/'training',stop_after=max(1,config.training.steps//2),evaluation_path=evaluation)
    train(config,data_path,output/'training',resume=output/'training/checkpoint.pt',evaluation_path=evaluation)
    report=reconstruct(config,output/'training/inference.pt',inputs/('problem.json' if config.provider else 'test_problem.json'),inputs/('observations.json' if config.provider else 'test_observations.json'),output/'reconstruction',inputs/('reference.npz' if config.provider else 'test_reference.npz'))
    for mode in report:
        evaluate(output / "reconstruction" / mode)
    return report


def main():
    parser = argparse.ArgumentParser(description="PDEFlow local training and reconstruction", epilog="Everyday: pdeflow setup → pdeflow train --preset heat-2d --device mps → pdeflow resume latest → pdeflow evaluate --run latest. Start API: pdeflow serve; frontend: cd frontend && npm run dev. Advanced --config commands remain available.")
    sub = parser.add_subparsers(dest="command", required=True)
    integrated = sub.add_parser('training', help='Integrated Train model workflow; legacy training commands remain available')
    actions = integrated.add_subparsers(dest='training_action', required=True)
    for action in ('start', 'list', 'status', 'resume', 'stop', 'use', 'extend'):
        p = actions.add_parser(action)
        p.add_argument('--workspace')
        if action in ('status', 'resume', 'stop', 'use', 'extend'):
            p.add_argument('selector', nargs='?', default='latest')
        if action in ('start', 'resume', 'extend'):
            p.add_argument('--detach', action='store_true', help='Return after launch; worker survives terminal/API disconnects')
            p.add_argument('--allow-poor-representation', action='store_true', help='Explicitly continue despite the validation baseline gate; recorded in run provenance')
        if action == 'start':
            p.add_argument('--problem', default='heat-wave-2d', choices=['heat-2d','wave-2d','heat-wave-2d'])
            p.add_argument('--model', default='mesh_m1_v1', choices=[*__import__('pdeflow.mesh.contracts',fromlist=['CANDIDATES']).CANDIDATES,'resolution-flexible'])
            p.add_argument('--budget', default='pipeline-check', choices=['pipeline-check','development','extended'])
            p.add_argument('--device', default='auto', choices=['auto','cpu','mps','cuda'])
            p.add_argument('--plan', action='store_true', help='Inspect resolved settings without launching')
            p.add_argument('--updates', type=int, help='Fixed-grid optimizer updates')
            p.add_argument('--evolution-updates',type=int)
            p.add_argument('--representation-updates', type=int)
            p.add_argument('--flow-updates', type=int)
            p.add_argument('--field-loss',choices=['mse','region_balanced_mse'],help='New-run field loss (default Standard MSE); does not change B or display metrics')
            p.add_argument('--field-alpha',type=float,help='Experimental region-balanced blend, default0.5; only with --field-loss region_balanced_mse')
        if action == 'extend':p.add_argument('--additional-updates', type=int, required=True)
        if action in ('status', 'list'):
            p.add_argument('--json', action='store_true')
    for name in ("generate", "train", "reconstruct", "smoke", "validate-config"):
        p = sub.add_parser(name)
        if name=='train':
            source=p.add_mutually_exclusive_group(required=True)
            source.add_argument('--config');source.add_argument('--preset')
        else:p.add_argument("--config", required=True)
        p.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
        if name != "validate-config":
            p.add_argument("--output", required=name!='train')
        if name == "train":
            p.add_argument("--data")
            p.add_argument("--evaluation", help="Fixed evaluation manifest from prepare-evaluation")
            p.add_argument("--resume")
            p.add_argument("--stop-after", type=int)
            p.add_argument('--workspace');p.add_argument('--name')
            p.add_argument('--steps',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--seed',type=int)
            p.add_argument('--data-mode',choices=['online','offline','hybrid'])
            p.add_argument('--dry-run','--plan',action='store_true');p.add_argument('--plain',action='store_true',help='Plain interval progress (also the default)')
        if name == "reconstruct":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--mode", choices=("unguided","observation","observation_pde","importance","fk"))
            p.add_argument("--fine-checkpoint", help="Optional compatible nested heat export for controller refinement")
            p.add_argument("--problem", required=True)
            p.add_argument("--observations", required=True)
            p.add_argument("--reference", help="Optional truth for evaluation only")
    for action in ('prepare-evaluation','prepare-training'):
        p=sub.add_parser(action);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p = sub.add_parser("evaluate")
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--result');selection.add_argument('--run')
    p.add_argument("--reference")
    p.add_argument('--workspace');p.add_argument('--specialist',choices=['heat','darcy','chemistry'])
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--full',action='store_true')
    p.add_argument('--device',choices=['cpu','mps','cuda','auto']);p.add_argument('--export',choices=['latest','best'],default='latest')
    p = sub.add_parser("inspect-checkpoint")
    p.add_argument("--checkpoint", required=True)
    sub.add_parser("hardware")
    p = sub.add_parser("estimate-data", help="Bounded reference generation/compression probe; no full dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--output")
    p = sub.add_parser("preflight", help="Bounded real training/resume/guidance/export at the configured layout")
    p.add_argument("--config", required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    p.add_argument("--data", help="Optional existing compatible dataset; otherwise generate a 12-instance preflight dataset")
    p.add_argument("--output", required=True)
    for name in ['setup','doctor','serve','presets','runs','status','resume','models']:
        p=sub.add_parser(name)
        p.add_argument('--workspace')
        if name!='serve':p.add_argument('--json',action='store_true')
        if name=='setup':p.add_argument('--headless',action='store_true',help='Skip Node/npm and frontend installation')
        if name=='serve':
            p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000);p.add_argument('--reload',action='store_true')
        if name in ['status','resume']:
            p.add_argument('selector',nargs='?',default='latest');p.add_argument('--specialist',choices=['heat','darcy','chemistry'])
        if name=='resume':
            p.add_argument('--device',choices=['cpu','mps','cuda','auto']);p.add_argument('--additional-steps',type=int,help='Add to the recorded total budget; constant Adam learning rate is retained')
            p.add_argument('--stop-after',type=int);p.add_argument('--plain',action='store_true')
        if name=='models':p.add_argument('--import',dest='import_path',help='Link existing exported checkpoints without moving them')
    p=sub.add_parser('capacity-probe');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('reference-example');p.add_argument('--preset',default='heat-2d');p.add_argument('--output',required=True)
    p=sub.add_parser('reference')
    select=p.add_mutually_exclusive_group(required=True);select.add_argument('--problem');select.add_argument('--preset')
    p.add_argument('--solver',choices=['auto','analytical','numerical','manufactured']);p.add_argument('--output',required=True)
    p=sub.add_parser('observe');p.add_argument('--reference',required=True);p.add_argument('--sampling',required=True);p.add_argument('--output',required=True);p.add_argument('--workspace')
    p=sub.add_parser('experiment')
    select=p.add_mutually_exclusive_group(required=True);select.add_argument('--problem');select.add_argument('--preset');select.add_argument('--reference')
    p.add_argument('--sampling');p.add_argument('--model',default='auto');p.add_argument('--output',required=True);p.add_argument('--workspace');p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p.add_argument('--inference');p.add_argument('--mode',default='observation_pde',choices=['unguided','observation','observation_pde','importance','fk']);p.add_argument('--stop-after',choices=['reference','observations'])
    p=sub.add_parser('compare');p.add_argument('--result',required=True);p.add_argument('--reference',required=True);p.add_argument('--policy',choices=['native','restrict_reference'],default='native');p.add_argument('--shared-times-only',action='store_true');p.add_argument('--metrics',default='mse,rmse,mae,relative_l2');p.add_argument('--output',required=True);p.add_argument('--workspace')
    p=sub.add_parser('train-all');p.add_argument('--suite',default='full',choices=['full','all','local-pilot','quickstart','capacity-local']);p.add_argument('--device',default='auto',choices=['cpu','mps','cuda','auto']);p.add_argument('--workspace');p.add_argument('--include',default='');p.add_argument('--exclude',default='');p.add_argument('--seed',type=int,default=37);p.add_argument('--budget',action='append',default=[]);p.add_argument('--plan',action='store_true');p.add_argument('--json',action='store_true');p.add_argument('--missing',action='store_true');p.add_argument('--reuse',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--continue-on-error',action='store_true');p.add_argument('--stop-after',type=int)
    for name in ['resume-suite','suite-status','suites']:
        p=sub.add_parser(name);p.add_argument('--workspace');p.add_argument('--json',action='store_true')
        if name!='suites':p.add_argument('selector',nargs='?',default='latest')
        if name=='resume-suite':p.add_argument('--retry-failed',action='store_true');p.add_argument('--stop-after',type=int)
    for name in ['catalogue','coverage']:
        p=sub.add_parser(name);p.add_argument('--workspace');p.add_argument('--json',action='store_true')
    p=sub.add_parser('storage');p.add_argument('--root');p.add_argument('--refresh',action='store_true');p.add_argument('--json',action='store_true')
    p=sub.add_parser('cleanup');p.add_argument('--root');p.add_argument('--register-cache');p.add_argument('--apply');p.add_argument('--confirm');p.add_argument('--preview',action='store_true')
    p=sub.add_parser('pin');p.add_argument('path');p.add_argument('--root');p.add_argument('--unpin',action='store_true')
    p=sub.add_parser('diagnose-learning');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda']);p.add_argument('--updates',type=int,default=100);p.add_argument('--cases',type=int,default=2)
    p=sub.add_parser('evolution-diagnose',help='Fixed held-out A2 versus same encoded initial state held constant; no training')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--earlier',action='append',default=[]);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda']);p.add_argument('--per-family',type=int,default=1);p.add_argument('--resolution',type=int,default=16)
    p=sub.add_parser('coordinate-diagnostic-case',help='Capture immutable observations, reference and explicit noise for staged diagnosis')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--job');p.add_argument('--reference');p.add_argument('--observations');p.add_argument('--problem');p.add_argument('--validation-index',type=int);p.add_argument('--training-observations',action='store_true');p.add_argument('--seed',type=int,default=91);p.add_argument('--steps',type=int,default=8);p.add_argument('--particles',type=int,default=4);p.add_argument('--resolution',type=int,default=32)
    p=sub.add_parser('coordinate-diagnose',help='A1/oracle-A2/B and exact-sensor evaluation, never hidden-reference conditioning')
    p.add_argument('--checkpoint',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--evolution');p.add_argument('--representation');p.add_argument('--stages',default='a1,a2,b');p.add_argument('--ablations',action='store_true');p.add_argument('--integration-steps',default='');p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('coordinate-diagnostic-baseline',help='Bounded deterministic observation-to-initial-latent diagnostic head')
    p.add_argument('--checkpoint',required=True);p.add_argument('--evolution');p.add_argument('--representation');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--train-cases',type=int,default=4);p.add_argument('--validation-cases',type=int,default=4);p.add_argument('--spatial',action='store_true',help='New direct-sensor diagnostic head; frozen parent unchanged');p.add_argument('--seed',type=int,default=194);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('coordinate-diagnostic-correct',help='Explicit bounded observation-only initial-latent optimization with prior penalty')
    p.add_argument('--checkpoint',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--learning-rate',type=float,default=.02);p.add_argument('--prior',type=float,default=.01);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('coordinate-train');p.add_argument('--config',required=True);p.add_argument('--stage',choices=['autoencoder','evolution','flow'],required=True);p.add_argument('--output',required=True);p.add_argument('--autoencoder','--dependency',dest='autoencoder');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--extend',action='store_true')
    p=sub.add_parser('coordinate-evaluate-ae');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto']);p.add_argument('--split',default='validation',choices=['validation','test'])
    p=sub.add_parser('coordinate-import');p.add_argument('--checkpoint',required=True);p.add_argument('--workspace')
    p=sub.add_parser('coordinate-runs');p.add_argument('--workspace')
    p=sub.add_parser('coordinate-reconstruct');p.add_argument('--checkpoint',required=True);p.add_argument('--problem',required=True);p.add_argument('--observations',required=True);p.add_argument('--queries');p.add_argument('--query-shape');p.add_argument('--query-times');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=12);p.add_argument('--particles',type=int,default=2);p.add_argument('--seed',type=int,default=91);p.add_argument('--chunk-size',type=int,default=1024);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('coordinate-compare');p.add_argument('--coordinate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--counts',default='4,12');p.add_argument('--noise',default='0,.03');p.add_argument('--resolutions',default='8,12');p.add_argument('--seeds',default='91,92');p.add_argument('--cases',type=int,default=2);p.add_argument('--particles',type=int,default=3);p.add_argument('--steps',type=int,default=8);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto'])
    p=sub.add_parser('reference-export');p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('reference-import');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('mesh',help='New observation-informed physical mesh model family');p.add_argument('mesh_args',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command=='mesh':
        from pdeflow.mesh.cli import main as mesh_main
        return mesh_main(args.mesh_args)
    try:
        from pdeflow import local
        if args.command in ('train','reconstruct','smoke','resume','train-all','resume-suite','experiment','compare','preflight','capacity-probe','coordinate-compare'):
            raise ValueError('Legacy generative architecture is retired. Use training start --model resolution-flexible or coordinate-train/reconstruct with a joint checkpoint. Existing artifacts remain untouched.')
        if args.command=='coordinate-diagnostic-case':
            from pdeflow.coordinate.diagnostic_cases import capture
            result=capture(args.checkpoint,args.output,args.job,args.reference,args.validation_index,args.observations,args.problem,args.seed,args.steps,args.particles,args.resolution,args.training_observations)
            print(json.dumps({'case_identity':result['case_identity'],'output':str(Path(args.output).resolve())},indent=2));return
        if args.command=='coordinate-diagnose':
            from pdeflow.coordinate.diagnostics import diagnose
            result=diagnose(args.checkpoint,args.case,args.output,args.device,tuple(args.stages.split(',')),args.evolution,args.representation,args.ablations,tuple(map(int,args.integration_steps.split(','))) if args.integration_steps else ())
            print(json.dumps({'output':str(Path(args.output).resolve()),'seconds':result['seconds'],'stages':{k:v.get('status','evaluated') for k,v in result['stages'].items()}},indent=2));return
        if args.command=='evolution-diagnose':
            from pdeflow.coordinate.evolution_audit import diagnose
            r=diagnose(args.checkpoint,args.output,args.device,args.earlier,args.per_family,args.resolution)
            print(json.dumps({'output':args.output,'successful_updates':r['successful_updates'],'comparisons':[{k:v for k,v in row.items() if k!='records'} for row in r['comparisons']]},indent=2));return
        if args.command=='coordinate-diagnostic-correct':
            from pdeflow.coordinate.diagnostic_correction import correct
            result=correct(args.checkpoint,args.case,args.output,args.device,args.steps,args.learning_rate,args.prior)
            print(json.dumps({'output':str(Path(args.output).resolve()),'correction_steps':result['steps'],'seconds':result['seconds']},indent=2));return
        if args.command=='coordinate-diagnostic-baseline':
            from pdeflow.coordinate.diagnostic_baseline import train_baseline
            result=train_baseline(args.checkpoint,args.output,args.evolution,args.representation,args.device,args.steps,args.train_cases,args.validation_cases,args.seed,spatial=args.spatial)
            print(json.dumps({'output':str(Path(args.output).resolve()),'successful_updates':result['successful_updates'],'seconds':result['seconds']},indent=2));return
        if args.command == 'training':
            from pdeflow import training_workflow as workflow
            if args.training_action == 'start':
                advanced = {}
                if args.field_alpha is not None and args.field_loss!='region_balanced_mse':raise ValueError('--field-alpha requires --field-loss region_balanced_mse')
                if args.field_loss is not None:advanced['field_loss']={'type':args.field_loss,**({'alpha':args.field_alpha} if args.field_alpha is not None else {})}
                if args.model in __import__('pdeflow.mesh.contracts',fromlist=['CANDIDATES']).CANDIDATES:
                    if any(x is not None for x in (args.representation_updates,args.evolution_updates,args.flow_updates)):raise ValueError('Mesh candidates use declared stages; use --updates or mesh config budgets, not legacy A1/A2/B flags')
                    if args.updates is not None:advanced['budgets']={stage:args.updates for stage in __import__('pdeflow.mesh.contracts',fromlist=['MeshConfig']).MeshConfig().budgets}
                elif args.model == 'resolution-flexible':
                    if args.updates is not None:raise ValueError('Choose separate --representation-updates and --flow-updates for this model')
                    if args.representation_updates is not None:advanced['ae_steps'] = args.representation_updates
                    if args.evolution_updates is not None:advanced['evolution_steps']=args.evolution_updates
                    if args.flow_updates is not None:advanced['flow_steps'] = args.flow_updates
                else:
                    if args.representation_updates is not None or args.flow_updates is not None:raise ValueError('Fixed-grid training has one --updates budget')
                    if args.updates is not None:advanced['training'] = {'steps': args.updates}
                request = {'problem':args.problem, 'model':args.model, 'budget':args.budget, 'device':args.device,
                           'advanced':advanced, 'allow_poor_representation':args.allow_poor_representation}
                if args.plan:
                    print(json.dumps(workflow.plan(request), indent=2));return
                report = workflow.start(request, args.workspace)
            elif args.training_action == 'resume':
                report = workflow.resume(args.selector, args.workspace, args.allow_poor_representation)
            elif args.training_action == 'extend':
                report = workflow.extend(args.selector, args.additional_updates, args.workspace)
            elif args.training_action == 'stop':
                report = workflow.stop(args.selector, args.workspace)
            elif args.training_action == 'use':
                report = workflow.select_model(args.selector, args.workspace)
                print(f"Selected {report['model']} model {report['id']}. {report['meaning']}");return
            elif args.training_action == 'list':
                rows = workflow.runs(args.workspace)
                if args.json:print(json.dumps(rows, indent=2))
                else:
                    for row in rows:print(f"{row['id']} · {row['plan']['problem']} · {row['plan']['model_label']} · {row['state']} · {row['stage_label']} {row['progress']}/{row['stage_budget']} · {row['meaning']}")
                return
            else:
                report = workflow.status(args.selector, args.workspace)
                if args.json:print(json.dumps(report, indent=2));return
            def display(row):
                print(f"{row['id']} · {row['state']} · {row['stage_label']} {row['progress']}/{row['stage_budget']} · {row['plan']['resolved_device']} · elapsed {row['elapsed_seconds']:.1f}s", flush=True)
            display(report)
            if args.training_action in ('start', 'resume', 'extend') and not args.detach:
                import time
                previous = None
                try:
                    while report['state'] in workflow.ACTIVE:
                        time.sleep(.5);report = workflow.status(report['id'], args.workspace)
                        marker = (report['state'], report['stage'], report['progress'])
                        if marker != previous:display(report);previous = marker
                except KeyboardInterrupt:
                    workflow.stop(report['id'], args.workspace)
                    print('Safe stop requested. The worker preserves its checkpoint at the next safe boundary.');return
            if report.get('error'):print(report['error'])
            if report.get('warning'):print(report['warning'])
            print(f"Inspect: pdeflow training status {report['id']}\nUse: pdeflow training use {report['id']}")
            if report['state'] == 'failed':raise ValueError('Training stage failed; inspect status/logs before resuming')
            return
        if args.command.startswith('coordinate-'):
            if args.command=='coordinate-train':
                from pdeflow.coordinate.training import train as coordinate_train
                report=coordinate_train(read_json(args.config),args.output,args.stage,args.autoencoder,args.resume,args.stop_after,extend=args.extend)
            elif args.command=='coordinate-evaluate-ae':
                from pdeflow.coordinate.autoencoder_evaluation import evaluate as ae_evaluate
                report=ae_evaluate(args.checkpoint,args.output,args.device,args.split)
            elif args.command=='coordinate-import':
                from pdeflow.coordinate.registry import register_checkpoint
                report=register_checkpoint(args.checkpoint,args.workspace)
            elif args.command=='coordinate-runs':
                from pdeflow.coordinate.registry import runs
                report=runs(args.workspace)
            elif args.command=='coordinate-reconstruct':
                from pdeflow.coordinate.inference import reconstruct as coordinate_reconstruct
                if args.queries and (args.query_shape or args.query_times):raise ValueError('Use scattered --queries or explicit regular query shape/times, not both')
                problem=read_json(args.problem)
                if args.query_shape:problem['grid']['shape']=list(map(int,args.query_shape.split(',')))
                if args.query_times:problem['grid']['times']=list(map(float,args.query_times.split(',')))
                report=coordinate_reconstruct(args.checkpoint,problem,read_json(args.observations),args.output,read_json(args.queries) if args.queries else None,args.steps,args.particles,args.seed,args.chunk_size,args.device)
            else:
                from pdeflow.coordinate.comparison import compare as coordinate_compare
                report=coordinate_compare(args.coordinate,args.baseline,args.output,tuple(map(int,args.counts.split(','))),tuple(map(float,args.noise.split(','))),tuple(map(int,args.resolutions.split(','))),tuple(map(int,args.seeds.split(','))),args.cases,args.particles,args.steps,args.device)
            if args.command in ('coordinate-runs','coordinate-import'):print(json.dumps(report,indent=2,allow_nan=False))
            elif args.command=='coordinate-train':print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2,allow_nan=False))
            else:print(json.dumps({'output':args.output,'kind':args.command,'records':len(report.get('records',[])),'seconds':report.get('seconds'),'meaning':report.get('meaning','Detailed metrics saved in the output artifact')},indent=2))
            return
        if args.command=='reference-export':
            from pdeflow.reference.imports import export_reference
            if Path(args.output).exists():raise FileExistsError('Export path exists; choose a fresh destination')
            write_json(args.output,export_reference(args.reference));print(args.output);return
        if args.command=='reference-import':
            from pdeflow.reference.imports import import_reference
            print(json.dumps(import_reference(read_json(args.input),args.output,__import__('uuid').uuid4().hex),indent=2));return
        if args.command=='diagnose-learning':
            from pdeflow.learning_diagnostics import run
            report=run(args.checkpoint,args.output,args.device,args.updates,args.cases)
            print(json.dumps({'output':args.output,'overfit':report['overfit'],'seconds':report['seconds']},indent=2));return
        if args.command in ('storage','cleanup','pin'):
            from pdeflow import storage
            if args.command=='cleanup' and args.apply and args.preview:raise ValueError('Choose --preview or --apply, not both')
            if args.command=='cleanup' and args.register_cache:
                if args.apply or args.preview:raise ValueError('Cache registration is separate from preview/apply')
                print(json.dumps(storage.register_cache(args.root,args.register_cache,'explicit_user'),indent=2));return
            report=storage.inventory(args.root,args.refresh) if args.command=='storage' else storage.pin(args.root,args.path,not args.unpin) if args.command=='pin' else storage.apply(args.root,args.apply,args.confirm) if args.apply else storage.preview(args.root)
            if args.command=='storage' and not args.json:
                print(f"Storage {report['root']} · {report['files']} files · {report['total_bytes']/1024**2:.2f} MiB · scanned {report['scanned_at']}")
                for kind,size in sorted(report['bytes_by_type'].items()):print(f"{kind}: {size/1024**2:.2f} MiB")
                print('Largest public artifacts (logical file sizes):')
                for row in sorted(report['entries'],key=lambda r:r['bytes'],reverse=True)[:10]:print(f"{row['bytes']/1024**2:.2f} MiB · {row['path']} · protected={row['protected']} pinned={row['pinned']} active={row['active']}")
                print('Use pdeflow storage --json for every path; --refresh explicitly rescans. Quarantine does not free disk space.')
            else:print(json.dumps(report,indent=2))
            return
        if args.command in ('catalogue','coverage'):
            from pdeflow.catalogue import catalogue,coverage
            report=catalogue() if args.command=='catalogue' else coverage(args.workspace)
            if args.json:print(json.dumps(report,indent=2))
            else:
                rows=report if isinstance(report,list) else report['targets']
                print(f"{rows[0]['catalogue_version']} · {len(rows)} finite targets; configuration/coverage is not scientific quality")
                for row in rows:
                    state=row.get('state','configured')
                    print(f"{row['id']}: {row['specialist']} {row['grid']['shape']} · {row['architecture']['parameters']} parameters · {state} {row.get('maximum_updates',0)}/{row['updates']} updates · export={row.get('compatible_discoverable_checkpoint',False)} evaluation={row.get('evaluation_available',False)}")
                print('Use --json for complete capabilities, distributions, resource profiles and resolved configurations.')
            return
        if args.command in ['train-all','resume-suite','suite-status','suites']:
            from pdeflow import suites
            if args.command=='train-all':
                if args.missing and args.force:raise ValueError('--missing and --force express different operations; choose one')
                budgets={}
                for item in args.budget:
                    name,value=item.split('=',1)
                    if name in budgets:raise ValueError('Duplicate preset budget override')
                    budgets[name]=int(value)
                options=dict(suite=args.suite,device=args.device,workspace=args.workspace,include=tuple(filter(None,args.include.split(','))),exclude=tuple(filter(None,args.exclude.split(','))),seed=args.seed,budgets=budgets,missing=args.missing)
                planned=suites.plan(**options)
                if not args.json:
                    print(f"Suite {args.suite}: sequential; no resolution reduction; outputs {planned['output_template']}")
                    for entry in planned['entries']:print(f"{entry['preset']}: {entry['grid']} / times{entry['physical_times']} · batch{entry['batch_size']} · {entry['updates']} updates · {entry['data_mode']} · {entry['device']} · matching runs{entry['existing_exact_matches']} · cost unmeasured")
                report=planned if args.plan else suites.start(**options,reuse=args.reuse,force=args.force,continue_on_error=args.continue_on_error,stop_after=args.stop_after)
                if args.plan:
                    if args.json:print(json.dumps(report,indent=2))
                    return
            elif args.command=='resume-suite':report=suites.resume(args.selector,args.workspace,args.stop_after,args.retry_failed)
            elif args.command=='suite-status':report=suites.status(args.selector,args.workspace)
            else:report=suites.records(args.workspace)
            if args.json:print(json.dumps(report,indent=2))
            elif isinstance(report,dict) and 'id' not in report:
                print(report.get('message','No pending targets'))
            elif isinstance(report,list):
                for row in report:print(f"{row['id']} · {row['state']} · {row['path']}")
            else:
                print(f"Suite {report['id']} · {report['state']} · active child {report['active_child']}\n{report['path']}")
                for entry in report['entries']:print(f"{entry['preset']}: {entry['state']} · update{entry.get('latest_update')} · {entry.get('run_id')} · {entry['run_path']}")
                print(report['resume_command'])
            return
        if args.command in ['reference-example','reference','observe','experiment','compare']:
            from dataclasses import asdict,replace
            from pdeflow.reference import ReferenceProblem,create,load
            from pdeflow.reference.examples import example
            from pdeflow.reference.experiment import reference_path,experiment
            from pdeflow.reference.observations import ObservationSamplingSpec,sample as observe
            if args.command=='reference-example':
                if Path(args.output).exists():raise FileExistsError('Example file exists; choose a fresh path')
                report=asdict(example(args.preset));write_json(args.output,report)
            elif args.command=='reference':
                spec=ReferenceProblem(**read_json(args.problem)) if args.problem else example(args.preset)
                if args.solver:spec=replace(spec,solver=args.solver)
                report=create(spec,args.output)
            elif args.command=='observe':
                report=observe(reference_path(args.reference,args.workspace),ObservationSamplingSpec(**read_json(args.sampling)),args.output)
            elif args.command=='experiment':
                spec=ReferenceProblem(**read_json(args.problem)) if args.problem else example(args.preset) if args.preset else None
                sampling=ObservationSamplingSpec(**read_json(args.sampling)) if args.sampling else ObservationSamplingSpec()
                report=experiment(spec,sampling,args.output,args.model,args.workspace,args.device,read_json(args.inference) if args.inference else None,args.mode,args.stop_after,args.reference)
            else:
                from pdeflow.evaluation.comparison import compare,EvaluationSpec
                directory=Path(args.result);p=ProblemSpec.from_dict(read_json(directory/'problem.json'))
                obs=ObservationSet.from_dict(read_json(directory/'observations.json'),p.grid)
                refproblem,reference,_=load(reference_path(args.reference,args.workspace))
                with np.load(directory/'result.npz',allow_pickle=False) as arrays:report=compare(arrays['samples'],p,obs,reference,refproblem,arrays['weights'] if 'weights' in arrays else None,EvaluationSpec(args.policy,args.shared_times_only,metrics=tuple(args.metrics.split(','))))
                if Path(args.output).exists():raise FileExistsError('Comparison file exists; choose a new evaluation output')
                write_json(args.output,report)
            print(json.dumps(report,indent=2,allow_nan=False))
            return
        managed_train=args.command=='train' and (args.preset or args.workspace or args.name or args.dry_run or args.steps is not None or args.batch_size is not None or args.seed is not None or args.data_mode or not args.output)
        if managed_train and (args.resume or args.data or args.evaluation):raise ValueError('Managed train prepares its own data/evaluation; use pdeflow resume RUN, or the original --config --output --data/--evaluation/--resume workflow')
        if managed_train:
            report=local.start_training(args.preset,args.config,args.workspace,args.output,args.device,args.steps,args.batch_size,args.seed,args.name,args.data_mode,args.dry_run,args.stop_after)
            print(json.dumps(report,indent=2) if args.dry_run else f"Run {report['id']}: {report['state']} at update {report['progress']}/{report['budget']}")
            return
        if args.command in ['setup','doctor','serve','presets','runs','status','resume','models']:
            if args.command=='setup':report=local.setup(args.workspace,not args.headless)
            elif args.command=='doctor':report=local.doctor(args.workspace)
            elif args.command=='serve':return local.serve(args.workspace,args.host,args.port,args.reload)
            elif args.command=='presets':report=local.presets()
            elif args.command=='runs':report=local.run_records(args.workspace)
            elif args.command=='status':report=local.status(args.selector,args.workspace,args.specialist)
            elif args.command=='resume':report=local.resume_training(args.selector,args.workspace,args.specialist,args.device,args.additional_steps,args.stop_after)
            else:report=local.models(args.workspace,args.import_path)
            local.display(args.command,report,args.json)
            if args.command=='doctor' and not report['ok']:raise SystemExit(1)
            return
        config = Config.load(args.config, getattr(args, "device", None), check_device=args.command not in ("generate", "estimate-data", "prepare-evaluation", "prepare-training")) if getattr(args, "config",None) else None
        if args.command in ("prepare-evaluation","prepare-training"):
            from pdeflow.data.providers import prepare
            report=prepare(config,args.output,args.command=="prepare-training")
        elif args.command == "generate":
            report = generate(config, args.output)
        elif args.command == "train":
            if not args.output:raise ValueError('Advanced train requires --output, or use --preset for a managed run')
            report = train(config, args.data, args.output, args.resume, args.stop_after, args.evaluation)
            report.pop("history")
        elif args.command == "reconstruct":
            report = reconstruct(config, args.checkpoint, args.problem, args.observations, args.output, args.reference, args.fine_checkpoint, args.mode)
        elif args.command == "smoke":
            report = smoke(config, args.output)
        elif args.command == "evaluate":
            if args.run:
                if args.reference:raise ValueError('--reference belongs to --result; --run uses its immutable recorded evaluation')
                report=local.evaluate_run(args.run,args.workspace,args.specialist,args.split,args.full,args.device,args.export)
                local.display('evaluate',report)
                return
            report = evaluate(args.result, args.reference)
        elif args.command == "inspect-checkpoint":
            checkpoint, _, _ = _read_checkpoint(args.checkpoint)
            report = {k: v for k, v in checkpoint.items() if k not in ("model", "optimizer", "rng", "history")}
        elif args.command == "hardware":
            report = environment(RuntimeSpec())
        elif args.command == "estimate-data":
            from pdeflow.data.shards import estimate
            report = estimate(config, args.samples)
            if args.output:
                if Path(args.output).exists(): raise FileExistsError("Estimate output exists; choose a fresh path")
                write_json(args.output, report)
        elif args.command == "capacity-probe":
            from pdeflow.capacity import probe
            report=probe(config,args.output)
        elif args.command == "preflight":
            from pdeflow.preflight import preflight
            report = preflight(config, args.output, args.data)
        else:
            report = config.to_dict()
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, FloatingPointError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
