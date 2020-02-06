from flask import Flask, escape, request
import flask

app = Flask(__name__)


@app.route('/')
def index():
    return flask.render_template("index.html.jinja2")


if __name__ == '__main__':
    debug = True
    app.run(debug=debug, port=9000, host="0.0.0.0")
