from jinja2 import Template
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
import os
import time


if __name__ == "__main__":

    layers = [
        {
            "name": "input layer\n(48 neurons)",
            "nodes_count": 4
        },
        {
            "name": "hidden layer 1\n(256 neurons)",
            "nodes_count": 5
        },
        {
            "name": "hidden layer 2\n(256 neurons)",
            "nodes_count": 5
        },
        {
            "name": "hidden layer 3\n(256 neurons)",
            "nodes_count": 5
        },
        {
            "name": "output layer\n(48 neurons)",
            "nodes_count": 4
        }
    ]

    i = 1
    for layer in layers:
        if i == 1:
            layer["nodes"] = ["x%s" % (j) for j in range(1, layer.get("nodes_count")+1)]
        elif i == len(layers):
            layer["nodes"] = ["O%s" % (j) for j in range(1, layer.get("nodes_count")+1)]
        else:
            layer["nodes"] = ["a%s%s" % (j, i) for j in range(1, layer.get("nodes_count")+1)]
        i += 1

    output_file_path = "neural_net.dot"

    with open(output_file_path, mode="w+") as f:

        file_loader = FileSystemLoader('templates')
        env = Environment(
            loader=file_loader,
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('neural_net.dot.jinja2')
        template.stream(layers=layers).dump(f)

        print("You can generate the graph via the following command:\n  dot -Tpng -O neural_net.dot")
