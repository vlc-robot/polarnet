
import numpy as np
import jsonlines
import collections
import tap


class Arguments(tap.Tap):
    result_file: str


def main(args):
    results = collections.defaultdict(dict)
    exists_taskvars = collections.defaultdict(set)
    with jsonlines.open(args.result_file, 'r') as f:
        for item in f:
            taskvar = '%s+%d'%(item['task'], item['variation'])
            if taskvar in exists_taskvars[item['checkpoint']]:
                continue
            exists_taskvars[item['checkpoint']].add(taskvar)
            results[item['checkpoint']].setdefault(item['task'], [])
            results[item['checkpoint']][item['task']].append((item['sr'] * item['num_demos'], item['num_demos']))
                    #.append((item['task'], item['variation'], item['sr']))
    ckpt_results = collections.defaultdict(list)
    for ckpt, res in results.items():
        for task, v in res.items():
            if np.sum([x[1] for x in v]) != 25:
                print(ckpt, task, np.sum([x[1] for x in v]))
            sr = np.sum([x[0] for x in v]) / np.sum([x[1] for x in v])
            ckpt_results[ckpt].append((task, sr))
   
    print('\nnum_tasks', len(ckpt_results[ckpt]))
    for ckpt, res in ckpt_results.items():
        print()
        print(ckpt)
        res.sort(key=lambda x: x[0])

        print(','.join([x[0] for x in res]))

        print(','.join(['%.2f' % (x[1]*100) for x in res]))

        print('average: %.2f' % (np.mean([x[1] for x in res]) * 100))


if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)
