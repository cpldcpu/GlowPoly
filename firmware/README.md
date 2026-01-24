# Firmware

CH32V003 firmware for the GlowPoly LED driver board.

## Overview

The firmware drives LED filaments on polyhedra by controlling GPIO pins connected to H-bridge motor drivers. Each channel can be set to Anode (high), Cathode (low), or High-Z (disabled).

## Hardware Requirements

- CH32V003 microcontroller
- GlowPoly driver board (see `../hardware/`)
- WCH-LinkE programmer

## Pin Configuration

| Pin | Function |
|-----|----------|
| PC0 | Channel 0 |
| PC1 | Channel 1 |
| PC2 | Channel 2 |
| PC3 | Channel 3 |
| PC6 | Channel 4 |
| PC7 | Channel 5 |

All pins are configured as 10MHz push-pull outputs.

## Building

Requires the [ch32fun](https://github.com/cnlohr/ch32fun) framework (included as submodule).

```bash
make
```

This produces:
- `main.bin` - Raw binary
- `main.hex` - Intel HEX format
- `main.elf` - ELF with debug symbols

## Flashing

```bash
make flash
```

Or use the WCH-LinkE utility directly.
