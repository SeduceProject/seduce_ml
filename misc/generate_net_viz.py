from jinja2 import Template
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
import os
import time


if __name__ == "__main__":

    layers = [
        {
            "name": "input layer",
            "nodes_count": 4
        },
        {
            "name": "hidden layer 1",
            "nodes_count": 6
        },
        {
            "name": "hidden layer 2",
            "nodes_count": 6
        },
        {
            "name": "output layer",
            "nodes_count": 4
        }
    ]

    for layer in layers:
        layer["nodes"] = ["%s_%s" % (layer.get("name").replace(" ", "_"), i) for i in range(1, layer.get("nodes_count")+1)]

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
