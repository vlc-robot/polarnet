import argparse
import jsonlines

from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser("Generate real instructions", add_help=False)
    parser.add_argument("--output-file", default="assets/real_robot_instructions.jsonl", type=str)
    return parser


def push_buttons_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        first_button = info[0]
        rtn0 = 'push the %s button' % first_button
        rtn1 = 'press the %s button' % first_button
        rtn2 = 'push down the button with the %s base' % first_button
        for next_button in info[1:]:
            rtn0 += ', then push the %s button' % next_button
            rtn1 += ', then press the %s button' % next_button
            rtn2 += ', then the %s one' % next_button
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
        instruction[f"{var}"].append(rtn2)
    return instruction


def stack_cup_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        first_cup = info[0]
        second_cup = info[1]
        rtn0 = f"stack the {first_cup} cup on top of the {second_cup} cup"
        rtn1 = f"place the {first_cup} cup onto the {second_cup} cup"
        rtn2 = f"put the {first_cup} cup on top of the {second_cup} one"
        rtn3 = f"pick up and set the {first_cup} cup down into the {second_cup} cup"
        rtn4 = f"create a stack of cups with the {second_cup} cup as its base and the {first_cup} on top of it"
        rtn5 = f"keeping the {second_cup} cup on the table, stack the {first_cup} one onto it"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
        instruction[f"{var}"].append(rtn2)
        instruction[f"{var}"].append(rtn3)
        instruction[f"{var}"].append(rtn4)
        instruction[f"{var}"].append(rtn5)
    return instruction


def put_plate_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        color = info[0]
        rtn0 = f"take the {color} plate to the target place"
        rtn1 = f"place the {color} plate on the target"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
    return instruction

def put_item_in_drawer_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        obj = info[0]
        part = info[1]
        rtn0 = f"put the {obj} in the {part} part of the drawer"
        rtn1 = f"take the {obj} and put it in the {part} compartiment of the drawer"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
    return instruction


def open_drawer_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        part = info[0]
        rtn0 = f"open {part} drawer"
        rtn1 = f"grip the {part} handle and pull the {part} drawer open"
        rtn2 = f"slide the {part} drawer open"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
        instruction[f"{var}"].append(rtn2)
    return instruction

def put_item_in_cabinet_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        obj = info[0]
        part = info[1]
        rtn0 = f"put the {obj} in the {part} part of the cabinet"
        rtn1 = f"take the {obj} and put it in the {part} compartiment of the cabinet"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
    return instruction


def put_fruit_in_box_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        obj = info[0]
        rtn0 = f"put the {obj} in the box"
        rtn1 = f"take the {obj} and put it inside the box"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
    return instruction


def hang_mug_template(variations):
    instruction = {}
    for var, info in variations.items():
        instruction[f"{var}"] = []
        color = info[0]
        part = info[1]
        rtn0 = f"take the {color} mug and put it on the {part} part of the hanger"
        rtn1 = f"put the {color} mug on the {part} part of the hanger"
        instruction[f"{var}"].append(rtn0)
        instruction[f"{var}"].append(rtn1)
    return instruction


def main(args):

    tasks = ["real_push_buttons", "real_put_plate", "real_stack_cup", "real_put_item_in_drawer", "real_open_drawer", "real_put_item_in_cabinet", "real_put_fruit_in_box", "real_hang_mug"]
    instructions = []
    for task in tasks:
        if task == "real_push_buttons":
            vars = {0: ["red", "green", "yellow"], 1: ["white", "yellow", "black"], 2: ["blue", "black", "red"], 3: ["orange", "pink", "white"], 4: ["green", "cyan"]}
            var_inst = push_buttons_template(vars)
        elif task == "real_stack_cup":
            vars = {0: ["yellow", "pink"], 1: ["navy", "yellow"], 2: ["pink", "cyan"], 3: ["cyan", "navy"], 4: ["pink", "yellow"]}
            var_inst = stack_cup_template(vars)
        elif task == "real_put_plate":
            vars = {0: ["white"], 1: ["blue"], 2: ["red"]}
            var_inst = put_plate_template(vars)
        elif task == "real_put_item_in_drawer":
            vars = {0: ["peach", "top"], 1: ["orange", "middle"], 2: ["strawberry", "top"]}
            var_inst = put_item_in_drawer_template(vars)
        elif task == "real_open_drawer":
            vars = {0: ["top"], 1: ["middle"]}
            var_inst = open_drawer_template(vars)
        elif task == "real_put_item_in_cabinet":
            vars = {0: ["apple", "top"], 1: ["strawberry", "bottom"], 2: ["lemon", "top"]}
            var_inst = put_item_in_cabinet_template(vars)
        elif task == "real_put_fruit_in_box":
            vars = {0: ["strawberry"], 1: ["peach"], 2: ["banana"], 3: ["lemon"]}
            var_inst = put_fruit_in_box_template(vars)
        elif task == "real_hang_mug":
            vars = {0: ["green", "middle"], 1: ["pink", "middle"],  2: ["blue", "top"]}
            var_inst = hang_mug_template(vars)
        else:
            continue
        instructions.append({"task": task, "variations": var_inst})

    print(instructions)
    output_file = jsonlines.open(args.output_file, 'a', flush=True)
    output_file.write_all(instructions)
    print("Instructions succesfully written in", args.output_file)

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate real instructions", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)