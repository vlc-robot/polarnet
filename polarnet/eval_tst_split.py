import os
import json
import jsonlines
import argparse
import multiprocessing as mp

from polarnet.config.default import get_config

def work_fn(cmd):
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', required=True)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--num_demos', type=int, default=25)
    parser.add_argument('--microstep_data_dir', type=str, default=None)
    parser.add_argument('--microstep_outname', type=str, default='microsteps')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--record_video', action='store_true', default=False)
    parser.add_argument('--taskvars', type=str, default=None)
    parser.add_argument('--cam_rand_factor', type=float, default=0.0)
    parser.add_argument('--instr_embed_file', type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config)
    if len(config.DATASET.taskvars) == 1 and os.path.exists(config.DATASET.taskvars[0]):
        taskvars = list(json.load(open(config.DATASET.taskvars[0], 'r')).keys())
        taskvars.sort()
    else:
        taskvars = config.DATASET.taskvars

    if args.taskvars is not None:
        taskvars = args.taskvars.split(',')
    exist_taskvars = set()
    pred_file = os.path.join(config.output_dir, 'preds', f'seed{args.seed}', 'results.jsonl')
    if os.path.exists(pred_file):
        with jsonlines.open(pred_file, 'r') as f:
            for x in f:
                 if x['checkpoint'] == args.checkpoint:
                     exist_taskvars.add('%s+%d'%(x['task'], x['variation']))

    cmds = []
    for taskvar in taskvars:
        if taskvar not in exist_taskvars:
            cmd = f'python eval_models.py --exp_config {args.exp_config} --headless --seed {args.seed} --num_demos {args.num_demos} --checkpoint {args.checkpoint} --num_workers 1 --taskvars {taskvar} --cam_rand_factor {args.cam_rand_factor} --instr_embed_file {args.instr_embed_file}'
            cmd = '%s %s' % (cmd, ' '.join(args.opts))
            if args.microstep_data_dir is not None:
                cmd = '%s --microstep_data_dir %s --microstep_outname %s' % (cmd, args.microstep_data_dir, args.microstep_outname)
            if args.record_video:
                cmd = '%s --record_video --not_include_robot_cameras' % cmd 
            cmds.append(cmd)
    print('num_jobs', len(cmds))

    if args.num_workers == 1:
        for cmd in cmds:
            work_fn(cmd)
    else:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(work_fn, cmds)
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()