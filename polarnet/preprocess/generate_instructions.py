"""
Generate instruction embeddings
"""
import os
import json
import jsonlines
from tqdm import tqdm

import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import torch

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.utils import name_to_task_class

import transformers

BROKEN_TASKS = set(
    [
        "empty_container",
        "set_the_table",
    ]
)


def generate_all_instructions(env_file: str, instruction_file: str):
    if os.path.exists(instruction_file):
        exist_tasks = set()
        with jsonlines.open(instruction_file) as f:
            for item in f:
                exist_tasks.add(item["task"])
        print("Exist task", len(exist_tasks))
    else:
        exist_tasks = []

    all_tasks = json.load(open(env_file))

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, headless=True)
    env.launch()

    outfile = jsonlines.open(instruction_file, "a", flush=True)

    for task in tqdm(all_tasks):
        if task in BROKEN_TASKS or task in exist_tasks:
            continue
        print(task)
        outs = {"task": task, "variations": {}}
        task_env = env.get_task(name_to_task_class(task))
        num_variations = task_env.variation_count()
        for v in tqdm(range(num_variations)):
            try:
                task_env.set_variation(v)
                descriptions, obs = task_env.reset()
                outs["variations"][v] = descriptions
            except Exception as e:
                print("Error", task, v, e)
        outfile.write(outs)

    env.shutdown()
    outfile.close()


def load_all_instructions(instruction_file: str):
    data = []
    with jsonlines.open(instruction_file, "r") as f:
        for item in f:
            data.append(item)
    return data


def load_text_encoder(encoder: str):
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model_name = "openai/clip-vit-base-patch32"
        tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name)
        model = transformers.CLIPTextModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unexpected encoder {encoder}")

    return tokenizer, model


def main(args):
    taskvar_instrs = load_all_instructions(args.instruction_file)

    tokenizer, model = load_text_encoder(args.encoder)
    model = model.to(args.device)

    os.makedirs(args.output_file, exist_ok=True)
    lmdb_env = lmdb.open(args.output_file, map_size=int(1024**3))

    for item in tqdm(taskvar_instrs):
        task = item["task"]
        for variation, instructions in item["variations"].items():
            key = "%s+%s" % (task, variation)

            instr_embeds = []
            for instr in instructions:
                tokens = tokenizer(instr, padding=False)["input_ids"]
                if len(tokens) > 77:
                    print("Too long", task, variation, instr)

                tokens = torch.LongTensor(tokens).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    embed = model(tokens).last_hidden_state.squeeze(0)
                instr_embeds.append(embed.data.cpu().numpy())

            txn = lmdb_env.begin(write=True)
            txn.put(key.encode("ascii"), msgpack.packb(instr_embeds))
            txn.commit()

    lmdb_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["bert", "clip"], default="clip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--generate_all_instructions", action="store_true", default=False
    )
    parser.add_argument("--env_file", default="assets/all_tasks.json")
    parser.add_argument(
        "--instruction_file", default="assets/taskvar_instructions.jsonl"
    )
    args = parser.parse_args()
    if args.generate_all_instructions:
        generate_all_instructions(args.env_file, args.instruction_file)
    main(args)
