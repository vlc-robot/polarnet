import json
import jsonlines
import collections
import numpy as np
import tap

from polarnet.utils.utils import get_assets_dir


class Arguments(tap.Tap):
    result_file: str


def main(args):
    task_groups = json.load(open(f"{get_assets_dir()}/74_tasks_per_category.json"))
    task2group = {}
    for group, tasks in task_groups.items():
        for task in tasks:
            task2group[task] = group
    group_orders = [
        "planning",
        "tools",
        "long_term",
        "rotation-invariant",
        "motion-planner",
        "screw",
        "multimodal",
        "precision",
        "visual_occlusion",
    ]

    results = collections.defaultdict(dict)
    with jsonlines.open(args.result_file, "r") as f:
        for item in f:
            results[item["checkpoint"]].setdefault(item["task"], [])
            results[item["checkpoint"]][item["task"]].append(item["sr"])

    ckpt_results = collections.defaultdict(list)
    for ckpt, res in results.items():
        for task, v in res.items():
            ckpt_results[ckpt].append((task, np.mean(v)))
    print("\nnum_tasks", len(ckpt_results[ckpt]))

    for ckpt, res in ckpt_results.items():
        print()
        print(ckpt, "num_tasks", len(res))
        group_res = collections.defaultdict(list)
        for task, sr in res:
            group_res[task2group[task]].append(sr)

        print(",".join(group_orders))
        print(",".join(["%.2f" % (np.mean(group_res[g]) * 100) for g in group_orders]))
        print("avg tasks: %.2f" % (np.mean([x[1] for x in res]) * 100))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
