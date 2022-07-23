import connexion  # type: ignore


def create_app():

    # create the application instance
    app = connexion.App(__name__, specification_dir="./")

    #  Create the REST API
    app.add_api("openapi.yaml", validate_responses=True)

    return app.app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
