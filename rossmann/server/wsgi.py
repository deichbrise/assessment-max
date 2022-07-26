from rossmann.server.app import create_app

# needed for the deserialization of the model
from rossmann.model.pipeline.ridge import to_binary_holidays  # noqa: F401

app = create_app()
