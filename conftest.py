import pytest
from rossmann.server.server import create_app


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Configures the application for testing."""

    with app.test_client() as client:
        with app.app_context():
            yield client
