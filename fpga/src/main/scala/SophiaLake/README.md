# SophiaLake FA26 bring-up

Configs and harness binders for the FA26 chip on the SophiaLake FPGA board
(xc7a100t). The chip link is a 32-bit decoupled serial-TL at 1.8 V (LVCMOS18).
Pin maps are in `HarnessBinders.scala` (`SophiaLakePins.fa26ChipLink`) and
follow the "TL pinout" tab of the FA26 SophiaLake pin spreadsheet.

## Configs

| Config | What it builds |
|---|---|
| `SophiaLakeFA26Config` | FA26 chip link at 50 MHz |
| `SophiaLakeFA26Link10MHzConfig` | FA26 chip link at 10 MHz (divided from the 50 MHz reference) |
| `SophiaLakeFA26PulldownConfig` | 50 MHz link + internal pull-downs on B22/J17 |
| `SophiaLakeFA26Link10MHzPulldownConfig` | 10 MHz link + internal pull-downs on B22/J17 |
| `SophiaLakeClockProgConfig` | Clock-chip I2C programmer, 50 MHz controller clock |
| `SophiaLakeClockProg10MHzConfig` | Clock-chip I2C programmer, 10 MHz controller clock |

The link frequency must divide 50 MHz evenly (50, 25, 10, 5, ...). Going
above 50 MHz would need an MMCM/PLL.

## Building

From `fpga/`:

```bash
make bitstream SUB_PROJECT=sophialake CONFIG=<Config> generated_src_name=<build-folder>
```

The bitstream ends up at
`<build-folder>/chipyard.fpga.sophialake.SophiaLakeHarness.<Config>/obj/SophiaLakeHarness.bit`.

To check a build:

```bash
B=<build-folder>/*/obj
grep -m1 -A6 "Design Timing Summary" $B/report/timing.txt | tail -1   # all slack values positive
grep ser_tl_clock $B/report/clocks.txt                                # 20 ns = 50 MHz, 100 ns = 10 MHz
grep -E "PULLDOWN|PULLUP" $B/report/io.txt                            # pull settings per pin
```

## Bring-up flow

1. Load a clock-programmer bitstream and program the clock chip from the
   `SophiaLakeRegProg` notebook (UART-TSI on the FTDI port, 921600 baud).
2. Load an FA26 bitstream and run the serial-TL tests.

The clock chip keeps its settings when the FPGA is reprogrammed, but may
reset on a board power cycle. If so, repeat step 1.

## Clock-chip I2C

- SDA = D22, SCL = D19, LVCMOS18 (bank 16), internal pull-ups on.
- Controller base address `0x40000000`.
- Notebook `CONTROLLER_CLOCK_HZ`: `50_000_000` for `SophiaLakeClockProgConfig`,
  `10_000_000` for `SophiaLakeClockProg10MHzConfig`.
- Device writes are disabled in the notebook by default (`ALLOW_DEVICE_WRITES = False`).
- Before first use, check that SDA/SCL idle at about 1.8 V, not 3.3 V.

## R1/R6 pull-downs

Chip-board resistors R1 and R6 sit too close to the gold fingers and bend the
SODIMM socket pins, shorting them to GND. This likely explains why different
boards show different bad serial-TL pins. They are being removed.

R1/R6 were pull-downs on chip nets TL_IN_VLD and TL_OUT_RDY, which are FPGA pins
B22 (`out.valid`) and J17 (`in.ready`). The `*Pulldown*` configs add internal
pull-downs (~10 kΩ) on those two pins.

Caveat: B22 and J17 are FPGA outputs. The internal pull-down only applies after
configuration, when the FPGA is already driving them, so it does not cover the
pre-configuration window R1/R6 handled. Keep the chip in reset until the FPGA
is programmed.

## Bitstreams

Bitstreams are not stored in git. They are in the team Google Drive:
https://drive.google.com/drive/folders/1LGwTiMY8dXCNY129IZ1y5f6GlEnjyf1V?usp=drive_link

Built from the code in this folder on branch `fa26-bringup`:

| File | Config |
|---|---|
| `puffin-i2c-50mhz-pullup.bit` | `SophiaLakeClockProgConfig` |
| `puffin-i2c-10mhz-pullup.bit` | `SophiaLakeClockProg10MHzConfig` |
| `puffin-fa26-50mhz.bit` | `SophiaLakeFA26Config` |
| `puffin-fa26-10mhz.bit` | `SophiaLakeFA26Link10MHzConfig` |
| `puffin-fa26-50mhz-pd.bit` | `SophiaLakeFA26PulldownConfig` |
| `puffin-fa26-10mhz-pd.bit` | `SophiaLakeFA26Link10MHzPulldownConfig` |
