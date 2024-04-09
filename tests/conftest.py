import os
import shutil
import tempfile

os.environ["BAMBOOST_MPI"] = "0"
from bamboost import index
from bamboost._config import paths


def pytest_sessionstart(session):
    """Setup tmp config directory."""
    tempdir = tempfile.mkdtemp()
    paths["CONFIG_DIR"] = tempdir
    paths["CONFIG_FILE"] = os.path.join(tempdir, "config.toml")
    paths["LOCAL_DIR"] = os.path.join(tempdir, "local")
    paths["DATABASE_FILE"] = os.path.join(tempdir, "local", "bamboost.db")

    # Create config files if they don't exist
    os.makedirs(paths["CONFIG_DIR"], exist_ok=True)
    os.makedirs(paths["LOCAL_DIR"], exist_ok=True)

    # change path of sqlite database
    # index.IndexAPI(_file=).__init__()
    print(index.IndexAPI().file)


def pytest_sessionfinish(session, exitstatus):
    """Remove tmp config directory again."""
    shutil.rmtree(paths["CONFIG_DIR"])
