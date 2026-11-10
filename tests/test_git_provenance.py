"""Training provenance distinguishes committed code from modified working trees."""
import subprocess

from quantstream.common import git_provenance


def test_git_revision_and_dirty_state(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    assert git_provenance(tmp_path)["status"] == "unavailable"
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    sample = tmp_path / "sample.txt"
    sample.write_text("original")
    git("add", "sample.txt")
    git("commit", "-m", "test fixture")
    clean = git_provenance(tmp_path)
    assert clean["commit"] == git("rev-parse", "HEAD")
    assert clean["branch"] == "main" and clean["dirty"] is False
    sample.write_text("changed")
    assert git_provenance(tmp_path)["dirty"] is True
