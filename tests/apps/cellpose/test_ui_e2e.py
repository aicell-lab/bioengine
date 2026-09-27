import os
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# Every test here drives a browser at the deployed app and injects HYPHA_TOKEN
# into its localStorage. The `page` fixture is not in LIVE_FIXTURES, so the
# marker is what gates them.
pytestmark = pytest.mark.live

SERVER_URL = "https://hypha.aicell.io"
APP_WORKSPACE = os.environ.get("HYPHA_TEST_WORKSPACE", "ri-scale")
APP_URL = f"{SERVER_URL}/{APP_WORKSPACE}/view/cellpose-finetuning"
RUN_LIVE_REGRESSION_TESTS = os.environ.get("RUN_LIVE_REGRESSION_TESTS") == "1"


def open_and_connect(page: Page) -> None:
    token = os.environ.get("HYPHA_TOKEN")
    if token:
        page.add_init_script(
            f"window.localStorage.setItem('hypha_token', {token!r});"
        )
    response = page.goto(APP_URL, timeout=60000)
    assert response and response.ok, "Failed to load app"
    expect(page).to_have_title(re.compile(r"Cellpose Fine-Tuning"), timeout=15000)
    expect(page.locator("text=Connected").first).to_be_visible(timeout=30000)


def capture_dialog(page: Page):
    captured = {"message": None}

    def _handle(dialog):
        captured["message"] = dialog.message
        dialog.accept()

    page.on("dialog", _handle)
    return captured

def test_app_loads(page: Page):
    """App loads and connects to Hypha."""
    open_and_connect(page)

def test_navigation(page: Page):
    """Basic tab navigation works."""
    open_and_connect(page)
    
    # Default is Dashboard
    expect(page.locator("h2:has-text('Dashboard')")).to_be_visible()
    
    # Click New Training
    page.click("text=New Training")
    expect(page.locator("h2:has-text('New Training Session')")).to_be_visible()
    
    # Click Inference
    page.click("text=Inference")
    expect(page.locator("h2:has-text('Live Inference')")).to_be_visible()

def test_file_browser(page: Page):
    """File browser opens and lists artifact content."""
    open_and_connect(page)
    
    # Go to Training
    page.click("text=New Training")
    
    artifact_id = "ri-scale/cellpose-finetuning"
    page.fill("input[placeholder='workspace/dataset-alias']", artifact_id)
    
    # Open Browser
    page.click("button[title='Browse Files']")
    
    # Wait for modal
    expect(page.locator("text=Artifact Explorer")).to_be_visible()
    
    # Wait for loading to finish
    expect(page.locator("text=Loading files...")).not_to_be_visible()
    
    # Should see "index.html"
    expect(page.locator("text=index.html")).to_be_visible()
    
    page.locator("div:has(h3:has-text('Artifact Explorer')) button").last.click()
    
def test_start_training_button_and_result(page: Page):
    """Clicks Start Training and validates the returned UI result."""
    open_and_connect(page)

    page.click("text=New Training")
    page.fill("input[placeholder='workspace/dataset-alias']", "ri-scale/zarr-demo")
    page.fill("input[placeholder='e.g. images/*/*.tif']", "images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif")
    page.fill("input[placeholder='e.g. annotations/*/*_mask.ome.tif']", "annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif")
    page.fill("input[placeholder='Optional']", "")
    page.locator("input[placeholder='Optional']").nth(1).fill("")
    page.fill("input[placeholder='Pretrained ID or Session ID']", "cpsam")

    dialogs = capture_dialog(page)
    page.click("button:has-text('Start Training')")

    deadline = time.time() + 60
    while dialogs["message"] is None and time.time() < deadline:
        page.wait_for_timeout(250)
    assert dialogs["message"] is not None, "No Start Training dialog received"
    msg = dialogs["message"] or ""
    assert "Start Training Failed" not in msg
    assert "Training started successfully" in msg
    expect(page.locator("h2:has-text('Session Analysis')")).to_be_visible(timeout=20000)


def test_infer_button_and_result(page: Page):
    """Clicks Run Segmentation with a real file and validates output/alert."""
    open_and_connect(page)
    page.click("text=Inference")

    image_path = Path("/Users/hugokallander/github-repos/bioengine-worker/tests/cellpose_legacy_scripts/t0000.ome.tif")
    assert image_path.exists(), f"Inference image not found: {image_path}"

    infer_input = page.locator("input[type='file']").first
    infer_input.set_input_files(str(image_path))
    expect(page.locator("text=t0000.ome.tif")).to_be_visible(timeout=10000)
    expect(page.locator("img[src^='blob:'], img[src^='data:image/png;base64']").first).to_be_visible(timeout=10000)

    dialogs = capture_dialog(page)
    page.click("button:has-text('Run Segmentation')")

    page.wait_for_timeout(12000)
    if dialogs["message"]:
        assert "Inference fail" not in dialogs["message"]

    infer_panel = page.locator("text=Found").first
    expect(infer_panel).to_be_visible(timeout=30000)


def test_no_raw_template_markers(page: Page):
    """The rendered UI should not leak template raw markers."""
    if not RUN_LIVE_REGRESSION_TESTS:
        pytest.skip("Set RUN_LIVE_REGRESSION_TESTS=1 to run live template-marker regression")

    open_and_connect(page)
    html = page.content()
    assert "{% raw %}" not in html
    assert "{% endraw %}" not in html


def test_stop_training_persists_after_reload(page: Page):
    """Start then stop a session, and ensure status does not revert to running after refresh/reload."""
    if not RUN_LIVE_REGRESSION_TESTS:
        pytest.skip("Set RUN_LIVE_REGRESSION_TESTS=1 to run live stop-persistence regression")

    open_and_connect(page)

    page.click("text=New Training")
    page.fill("input[placeholder='workspace/dataset-alias']", "ri-scale/zarr-demo")
    page.fill("input[placeholder='e.g. images/*/*.tif']", "images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif")
    page.fill("input[placeholder='e.g. annotations/*/*_mask.ome.tif']", "annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif")
    page.fill("input[placeholder='Optional']", "")
    page.locator("input[placeholder='Optional']").nth(1).fill("")
    page.fill("input[placeholder='Pretrained ID or Session ID']", "cpsam")

    start_dialogs = capture_dialog(page)
    page.click("button:has-text('Start Training')")

    deadline = time.time() + 60
    while start_dialogs["message"] is None and time.time() < deadline:
        page.wait_for_timeout(250)
    assert start_dialogs["message"] is not None, "No Start Training dialog received"

    match = re.search(r"Session ID:\s*([a-zA-Z0-9-]+)", start_dialogs["message"] or "")
    assert match is not None, f"Could not parse session id from: {start_dialogs['message']}"
    session_id = match.group(1)

    page.click("text=Dashboard")
    row = page.locator("tr", has=page.locator(f"text={session_id[:8]}"))
    expect(row.first).to_be_visible(timeout=20000)

    stop_dialogs = capture_dialog(page)
    row.first.locator("button:has-text('Stop')").click()

    deadline = time.time() + 30
    while stop_dialogs["message"] is None and time.time() < deadline:
        page.wait_for_timeout(250)
    assert stop_dialogs["message"] == "Stop this training session?"

    for _ in range(8):
        page.click("button:has-text('Refresh')")
        page.wait_for_timeout(1000)
        status_text = row.first.locator("td").nth(1).inner_text().strip().lower()
        if "running" not in status_text:
            break
    else:
        raise AssertionError("Session still running after stop + refresh attempts")

    page.reload(timeout=60000)
    expect(page.locator("text=Connected").first).to_be_visible(timeout=30000)
    page.click("text=Dashboard")
    row_after_reload = page.locator("tr", has=page.locator(f"text={session_id[:8]}"))
    expect(row_after_reload.first).to_be_visible(timeout=20000)
    status_after_reload = row_after_reload.first.locator("td").nth(1).inner_text().strip().lower()
    assert "running" not in status_after_reload, f"Status reverted to running: {status_after_reload}"

