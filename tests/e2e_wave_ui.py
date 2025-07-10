"""
Quantonium OS - Wave UI End-to-End Tests

End-to-end tests for the waveform visualization UI using Playwright.
These tests validate the UI functionality and interaction with the API.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

# Configuration
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:5000")
WAVE_UI_PATH = "/static/wave_ui/index.html"
STREAM_TIMEOUT = 2000  # ms
EXPECTED_EVENTS = 10


@pytest.mark.asyncio
async def test_wave_ui_loads():
    """Test that the wave UI loads successfully"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the wave UI
        await page.goto(f"{BASE_URL}{WAVE_UI_PATH}")

        # Check for expected elements
        await expect(page.locator("h1")).to_contain_text("Quantonium OS")
        await expect(page.locator("h2")).to_contain_text("Live Resonance Visualization")

        # Verify both mode buttons are present
        await expect(page.locator("#encrypt-mode-btn")).to_be_visible()
        await expect(page.locator("#stream-mode-btn")).to_be_visible()

        # Verify canvas is present
        await expect(page.locator("#waveform-canvas")).to_be_visible()

        await browser.close()


@pytest.mark.asyncio
async def test_encrypt_mode():
    """Test the encrypt mode functionality"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the wave UI
        await page.goto(f"{BASE_URL}{WAVE_UI_PATH}")

        # Ensure encrypt mode is active
        await page.click("#encrypt-mode-btn")

        # Fill in the encryption form
        await page.fill("#plaintext", "Test encryption message")
        await page.fill("#key", "test-key")

        # Set up a listener for API requests
        api_request_received = asyncio.Future()

        async def handle_request(request):
            if request.url.endswith("/api/encrypt") and request.method == "POST":
                api_request_received.set_result(True)

        page.on("request", handle_request)

        # Click the encrypt button
        await page.click("#encrypt-btn")

        # Wait for API request or timeout
        try:
            await asyncio.wait_for(api_request_received, timeout=5.0)
            api_called = True
        except asyncio.TimeoutError:
            api_called = False

        # Check that API was called
        assert api_called, "Encrypt API should be called"

        # Check for result display
        await expect(page.locator("#ciphertext-display")).not_to_be_empty()

        await browser.close()


@pytest.mark.asyncio
async def test_stream_mode():
    """Test the stream mode functionality"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the wave UI
        await page.goto(f"{BASE_URL}{WAVE_UI_PATH}")

        # Switch to stream mode
        await page.click("#stream-mode-btn")

        # Verify stream panel is visible
        await expect(page.locator("#stream-panel")).to_be_visible()

        # Store initial events count
        initial_events = await page.locator("#events-count").text_content()

        # Set up a listener for EventSource requests
        events_source_created = asyncio.Future()

        async def handle_request(request):
            if "/api/stream/wave" in request.url:
                events_source_created.set_result(True)

        page.on("request", handle_request)

        # Click start stream button
        await page.click("#stream-start-btn")

        # Wait for EventSource request or timeout
        try:
            await asyncio.wait_for(events_source_created, timeout=5.0)
            stream_started = True
        except asyncio.TimeoutError:
            stream_started = False

        # Check that stream was started
        assert stream_started, "EventSource should be created for streaming"

        # Wait for events to be received
        await page.wait_for_timeout(STREAM_TIMEOUT)

        # Check that events were received
        events_count_text = await page.locator("#events-count").text_content()
        events_count = int(events_count_text)

        # Should receive multiple events in the timeout period
        assert events_count > 0, "Should receive at least one event"

        # Latency should be displayed
        await expect(page.locator("#avg-latency")).not_to_have_text("0ms")

        # Stop the stream
        await page.click("#stream-stop-btn")

        # Verify stream buttons state
        await expect(page.locator("#stream-start-btn")).not_to_be_disabled()
        await expect(page.locator("#stream-stop-btn")).to_be_disabled()

        await browser.close()


@pytest.mark.asyncio
async def test_mode_switching():
    """Test switching between encrypt and stream modes"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the wave UI
        await page.goto(f"{BASE_URL}{WAVE_UI_PATH}")

        # Check initial mode
        await expect(page.locator("#encrypt-panel")).to_be_visible()
        await expect(page.locator("#stream-panel")).to_be_hidden()

        # Switch to stream mode
        await page.click("#stream-mode-btn")

        # Check stream mode is active
        await expect(page.locator("#encrypt-panel")).to_be_hidden()
        await expect(page.locator("#stream-panel")).to_be_visible()

        # Switch back to encrypt mode
        await page.click("#encrypt-mode-btn")

        # Check encrypt mode is active again
        await expect(page.locator("#encrypt-panel")).to_be_visible()
        await expect(page.locator("#stream-panel")).to_be_hidden()

        await browser.close()


@pytest.mark.asyncio
async def test_canvas_rendering():
    """Test that the canvas is rendering properly"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate to the wave UI
        await page.goto(f"{BASE_URL}{WAVE_UI_PATH}")

        # Check canvas is rendered
        canvas_el = page.locator("#waveform-canvas")
        await expect(canvas_el).to_be_visible()

        # Take a screenshot of the canvas
        screenshot_path = "wave_ui_screenshot.png"
        await canvas_el.screenshot(path=screenshot_path)

        # Verify screenshot was created
        assert Path(screenshot_path).exists(), "Screenshot should be created"

        # Clean up
        Path(screenshot_path).unlink(missing_ok=True)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_wave_ui_loads())
    asyncio.run(test_encrypt_mode())
    asyncio.run(test_stream_mode())
    asyncio.run(test_mode_switching())
    asyncio.run(test_canvas_rendering())
    print("All E2E tests passed!")
