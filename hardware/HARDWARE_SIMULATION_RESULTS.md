# Hardware Simulation Results (Icarus)

A full Icarus/`vvp` simulation of the RFT middleware testbench produced a VCD file `quantoniumos_unified.vcd` (in this `hardware/` folder) and completed the built-in test suite. This document summarizes the run and lists next steps to produce waveform screenshots for the paper submission.

Observed outputs (example):
- VCD produced: `hardware/quantoniumos_unified.vcd`
- Tests executed: IMPULSE, NULL, DC, NYQUIST, RAMP, STEP, TRIANGLE, HEX SEQUENCE, SINGLE HIGH VALUE, TWO PEAKS
- Final message printed by testbench: `ALL TESTS COMPLETED!` and `$finish` called

Notes:
- The simulation prints frequency-domain tables (Amplitude, Phase, Energy%) for each test input.
- The VCD is suitable for inspection in GTKWave or Makerchip wave viewers.

Recommended next steps (pick one):

1) Produce a waveform PNG locally and commit it:

```bash
cd hardware
# Re-run the sim if you need a fresh VCD
iverilog -g2005-sv -o rft_tb *.sv && vvp rft_tb

# If headless, use Xvfb to provide a virtual display
# Install Xvfb if needed: sudo apt update && sudo apt install -y xvfb
xvfb-run --auto-servernum --server-args='-screen 0 1920x1200x24' gtkwave quantoniumos_unified.vcd &
# Then export a PNG from the GTKWave GUI or use a GTKWave script to save a screenshot.
```

2) Upload `quantoniumos_unified.vcd` and an exported PNG to `submission/figures/` (or `figures/`) and I will:
- Add the PNG to the paper figures
- Update `PAPER_VALIDATION_PLAN.md` and the submission manifest to reference the waveform PNG

3) Ask me to try capturing the waveform here by replying: "Please capture waveforms here". I will then attempt a headless `gtkwave` capture (requires `xvfb`) and integrate the resulting PNG into `figures/` if successful.

If you prefer I do the integration after you upload the PNG/VCD, just upload the files and tell me where you put them.
