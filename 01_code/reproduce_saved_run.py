"""Replay one recorded manuscript run using its saved seed and settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import Interface
from EngineeringInterface import run_engineering_case
from ExperimentAlgorithms import get_f
from IIGO_Ablations import ABLATION_FUNCTIONS
from IIGO_SensitivityVariants import run_sensitivity_variant

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'02_data'
FILES={
    'main':'01_main_benchmark_NP10/runs_final16.csv',
    'np20':'02_population_and_configuration/runs_NP20.csv',
    'np30':'02_population_and_configuration/runs_NP30.csv',
    'recommended':'02_population_and_configuration/runs_Recommended.csv',
    'd100':'03_high_dimensional_D100/full_results.csv',
    'ablation':'04_ablation/full_results.csv',
    'parameters':'05_parameter_sensitivity/full_results.csv',
    'engineering':'08_engineering/classical_and_ELD_300000FEs/engineering_runs.csv',
    'truss':'08_engineering/truss_recommended_FEs/engineering_runs.csv',
}

def digest(population):
    return hashlib.sha256(np.ascontiguousarray(population).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=FILES,default='main')
    parser.add_argument('--algorithm',default='IIGO')
    parser.add_argument('--function',type=int,default=1)
    parser.add_argument('--problem',default='RSLD')
    parser.add_argument('--condition',default='baseline')
    parser.add_argument('--run',type=int,default=0,help='Run index as stored in the CSV; engineering runs start at 1')
    args=parser.parse_args()
    frame=pd.read_csv(DATA/FILES[args.experiment],float_precision='round_trip')
    run_col='run_number' if 'run_number' in frame else 'run'
    chosen=frame[frame[run_col]==args.run]
    engineering=args.experiment in ('engineering','truss')
    if engineering:
        chosen=chosen[(chosen['problem']==args.problem)&(chosen['paper_label']==args.algorithm)]
    else:
        func_col='paper_function' if 'paper_function' in frame else 'function'
        chosen=chosen[chosen[func_col]==args.function]
        if args.experiment=='parameters':
            chosen=chosen[chosen['condition']==args.condition]
        else:
            algo_col='optimizer' if 'optimizer' in frame else 'algorithm'
            chosen=chosen[chosen[algo_col]==args.algorithm]
    if len(chosen)!=1:
        raise ValueError('Choose an algorithm, problem/function and run matching exactly one CSV row')
    row=chosen.iloc[0]
    seed=int(row['seed']); size=int(row['population_size'])
    dimension=int(row['dimension']) if 'dimension' in row else None
    if engineering:
        import EngineerBenchmark
        problem=str(row['code_problem'])
        dimension=EngineerBenchmark.problem_dimension(problem)
        population=np.random.default_rng(seed).uniform(0,1,(size,dimension))
        assert digest(population)==row['initial_population_sha256']
        result=run_engineering_case(str(row['algorithm']),problem,max_fes=int(row['max_fes']),population_size=size,run_number=int(row[run_col])-1,seed=seed,initial_population=population)
        actual=float(result['optimizer_merit']); expected=float(row['fitness'])
        report={'experiment':args.experiment,'seed':seed,'stored_fitness':expected,'replayed_fitness':actual,'stored_objective':float(row['objective']),'replayed_objective':float(result['objective'])}
    else:
        benchmark=str(row['benchmark'])
        func=int(row['cec_function'] if 'cec_function' in row else row['function'])
        matrices,shift,shuffle=Interface.load_cec_input(benchmark,func,dimension,input_root=str(ROOT/'input'))
        population=np.random.default_rng(seed).uniform(-100,100,(size,dimension))
        assert digest(population)==row['initial_population_sha256']
        budget=int(row['max_fes']) if 'max_fes' in row else int(row['optimization_fes']) if 'optimization_fes' in row else int(row['final_fes']) if 'final_fes' in row else 300000
        arguments=(budget,size,-100.0,100.0,dimension,matrices,shift,func,shuffle,int(benchmark[-4:]),seed,population)
        if args.experiment=='parameters':
            output=run_sensitivity_variant(*arguments,float(row['gr']),float(row['window_fraction']),str(row['modulation']))
        elif args.experiment=='ablation':
            output=ABLATION_FUNCTIONS[str(row['algorithm'])](*arguments)
        else:
            label=str(row['optimizer'] if 'optimizer' in row else row['algorithm'])
            label={'SO':'SNAKE','LSHADE-cnEpSin':'LSHADE_cnEpSin'}.get(label,label)
            output=get_f(label)(*arguments)
        actual=float(output[3]); expected=float(row['best_fitness'])
        report={'experiment':args.experiment,'seed':seed,'stored_fitness':expected,'replayed_fitness':actual,'final_fes':int(output[1][-1])}
    report['absolute_difference']=abs(actual-expected)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
