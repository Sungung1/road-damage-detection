from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from .predict import predict_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_FOLDER = PROJECT_ROOT / "data" / "uploads"
RESULT_FOLDER = PROJECT_ROOT / "data" / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload_file():
        file = request.files.get("file")
        if file is None or file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = UPLOAD_FOLDER / filename
        output_path = RESULT_FOLDER / "output.jpg"
        file.save(input_path)

        predict_image(
            image_path=input_path,
            output_path=output_path,
            model_path=MODEL_PATH if MODEL_PATH.exists() else None,
        )

        return redirect(url_for("results"))

    @app.route("/results")
    def results():
        results_image = os.path.join("..", "data", "results", "output.jpg")
        return render_template("results.html", results_image=results_image)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
