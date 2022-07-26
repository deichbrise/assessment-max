import pytest
from rossmann.server.app import create_app


@pytest.fixture(scope="session")
def app():
    from rossmann.model.pipeline.ridge import to_binary_holidays  # noqa: F401

    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Configures the application for testing."""

    with app.test_client() as client:
        with app.app_context():
            yield client
