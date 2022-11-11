import json


def save_results(path, results, model):
    try:
        with open(path, 'r') as _:
            pass
    except FileNotFoundError:
        with open(path, 'w') as _:
            pass
    try:
        with open(path, "r") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        with open(path, 'w') as fp:
            json.dump({type(model).__name__: results}, fp)
        return

    data[type(model).__name__] = results

    with open(path, "w") as fp:
        json.dump(data, fp)
