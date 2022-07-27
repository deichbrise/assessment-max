from pathlib import Path
import connexion  # type: ignore
from jsonschema import draft4_format_checker  # type: ignore
from datetime import datetime
from rossmann.server.model.predictor import StorePredictor  # type: ignore


@draft4_format_checker.checks("date")
def is_custom_date(val: str) -> bool:
    """jsonschema validator to check if a string is a valid date

    Required due to https://github.com/spec-first/connexion/issues/476\
    Args:
        val (str): string tp be validated

    Returns:
        bool: is valid date
    """

    try:
        _ = datetime.strptime(val, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def create_app():

    # create the application instance
    app = connexion.App(__name__, specification_dir="./")
    app.app.config.from_pyfile("config/default.py")

    #  Create the REST API
    app.add_api("openapi.yaml", validate_responses=True)

    app.app.config["predictor"] = StorePredictor(
        Path(app.app.config["MODEL_PATH"])
    )

    return app.app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
