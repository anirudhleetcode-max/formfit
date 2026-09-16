"""FormFit end-to-end journey (Playwright, headless Chromium).

Needs backend on :8003 and frontend on :5175 (run_e2e.sh starts both) and a fake-webcam file
(FAKE_CAM, a .y4m made from samples/squat_demo.webm by run_e2e.sh).

Journey: register -> Train (squat) -> fake camera -> landmarks + skeleton -> reps counted ->
end session -> session detail with rep chart -> Upload mode on the same clip -> save ->
History -> Progress. Screenshots go to docs/screenshots/.
"""
import json
import os
import time
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("E2E_BASE", "http://127.0.0.1:5175")
FAKE_CAM = os.environ.get("FAKE_CAM", str(ROOT / "e2e" / ".cache" / "squat.y4m"))
SAMPLE = ROOT / "samples" / "squat_demo.webm"
HERE = Path(__file__).resolve().parent
SHOTS = ROOT / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def browser():
    assert Path(FAKE_CAM).exists(), f"fake camera file missing: {FAKE_CAM}"
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={FAKE_CAM}",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--autoplay-policy=no-user-gesture-required",
        ])
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera"])
    pg = ctx.new_page()
    pg.on("console", lambda m: m.type in ("error",) and print("[console]", m.text))
    yield pg
    ctx.close()


def stats(page: Page) -> dict:
    return page.evaluate("() => window.__formfit || {frames: 0, poseFrames: 0, reps: 0}")


def wait_for(page: Page, cond, timeout_s: float, what: str):
    end = time.time() + timeout_s
    while time.time() < end:
        s = stats(page)
        if cond(s):
            return s
        page.wait_for_timeout(500)
    raise AssertionError(f"timed out waiting for {what}; last stats={stats(page)}")


def register(page: Page) -> str:
    email = f"e2e_{uuid.uuid4().hex[:8]}@example.com"
    page.goto(f"{BASE}/login")
    expect(page.get_by_role("heading", name="Sign in")).to_be_visible()
    page.get_by_role("button", name="New here? Create an account").click()
    page.get_by_label("Name").fill("E2E Lifter")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password").fill("secret123")
    page.get_by_role("button", name="Create account").click()
    expect(page.get_by_role("heading", name="Train")).to_be_visible()
    return email


def test_full_journey(page: Page):
    register(page)
    # CPU delegate: faster than GPU-on-SwiftShader in headless CI (GPU->CPU fallback is automatic in real use)
    page.goto(f"{BASE}/?delegate=cpu")

    # ---- live session with the fake webcam ----
    page.get_by_test_id("ex-squat").click()
    page.get_by_test_id("start-camera").click()
    s = wait_for(page, lambda s: s["poseFrames"] >= 10, 90, "landmarks from the fake camera")
    print("camera stats", s)
    expect(page.get_by_test_id("tracking")).to_have_text("Tracking", timeout=30_000)
    # the overlay canvas has skeleton pixels on it
    painted = page.evaluate("""() => {
        const c = document.querySelector('[data-testid=overlay]');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
        return n;
    }""")
    assert painted > 500, f"skeleton not drawn ({painted} px)"

    page.get_by_test_id("start-session").click()
    s = wait_for(page, lambda s: s["reps"] >= 2, 150, "at least 2 squat reps")
    print("live stats", s)
    assert int(page.get_by_test_id("total-reps").inner_text()) >= 2
    # HUD shows a numeric tracking confidence while the lifter is tracked
    expect(page.get_by_test_id("confidence")).to_contain_text("%")
    # screenshot while the lifter is mid-rep and the skeleton is on screen
    page.wait_for_timeout(1200)
    page.screenshot(path=str(SHOTS / "01-live-session.png"))

    page.get_by_test_id("finish").click()
    page.wait_for_url("**/history/*", timeout=20_000)
    expect(page.get_by_test_id("detail-reps")).to_be_visible()
    live_reps = int(page.get_by_test_id("detail-reps").inner_text())
    assert live_reps >= 2
    expect(page.locator(".recharts-surface").first).to_be_visible()
    expect(page.get_by_test_id("fatigue-note")).to_be_visible()
    expect(page.get_by_test_id("tracking-quality")).to_be_visible()
    page.wait_for_timeout(500)
    page.screenshot(path=str(SHOTS / "02-session-detail.png"), full_page=False)

    # ---- upload mode on the same clip ----
    page.evaluate("() => { window.__formfitTrace = true }")  # keep raw landmarks for the parity check
    page.get_by_label("Main").get_by_role("link", name="Upload").click()
    page.get_by_test_id("ex-squat").click()
    page.get_by_test_id("video-input").set_input_files(str(SAMPLE))
    page.get_by_test_id("analyse").click()
    expect(page.get_by_test_id("upload-result")).to_be_visible(timeout=240_000)
    s = stats(page)
    trace = page.evaluate("() => window.__formfit.trace")
    (HERE / ".cache").mkdir(exist_ok=True)
    (HERE / ".cache" / "upload_trace.json").write_text(json.dumps({"frames": trace, "width": 640, "height": 360}))
    print("upload stats", {k: v for k, v in s.items() if k != "trace"})
    assert s["poseFrames"] / max(1, s["frames"]) > 0.8, "pose should be found in most frames"
    # the clip contains 4 squats (hand-counted, data/real_clips/manifest.json). Upload analysis uses a
    # fresh landmarker with video-time timestamps, so the count does not depend on the live session above.
    assert s["reps"] == 4, {k: v for k, v in s.items() if k != "trace"}
    page.get_by_test_id("save-upload").click()
    page.wait_for_url("**/history/*", timeout=20_000)
    assert int(page.get_by_test_id("detail-reps").inner_text()) == s["reps"]

    # ---- history + progress ----
    page.get_by_label("Main").get_by_role("link", name="History").click()
    expect(page.locator("table.sessions tbody tr")).to_have_count(2)
    page.get_by_label("Main").get_by_role("link", name="Progress").click()
    expect(page.get_by_test_id("kpi-sessions")).to_have_text("2")
    expect(page.get_by_role("heading", name="Recurring faults")).to_be_visible()


def test_demo_account_dashboard(page: Page):
    """Seeded demo data: login via the 'Use demo account' link and capture Progress."""
    page.goto(f"{BASE}/login")
    page.get_by_role("button", name="Use demo account").click()
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_role("heading", name="Train")).to_be_visible()
    page.get_by_label("Main").get_by_role("link", name="Progress").click()
    expect(page.locator(".recharts-surface").first).to_be_visible(timeout=15_000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(SHOTS / "03-progress.png"))
    page.get_by_label("Main").get_by_role("link", name="History").click()
    expect(page.locator("table.sessions tbody tr").first).to_be_visible()
    page.screenshot(path=str(SHOTS / "04-history.png"))
    # mobile layout sanity check (no horizontal scroll)
    page.set_viewport_size({"width": 375, "height": 800})
    page.get_by_label("Main").get_by_role("link", name="Progress").click()
    expect(page.get_by_test_id("kpi-sessions")).to_be_visible()
    overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1, f"horizontal overflow {overflow}px at 375px"
    page.wait_for_timeout(600)
    page.screenshot(path=str(SHOTS / "06-mobile-progress.png"))
    # About / model card panel: synthetic label and disclaimer are visible
    page.set_viewport_size({"width": 1440, "height": 900})
    page.get_by_label("Main").get_by_role("link", name="About").click()
    expect(page.get_by_test_id("model-panel")).to_contain_text("synthetic", timeout=15_000)
    expect(page.get_by_test_id("disclaimer")).to_contain_text("not medical advice")
    page.wait_for_timeout(400)
    page.screenshot(path=str(SHOTS / "05-about-model.png"))
