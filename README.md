# Microver

This repository contains the source code for the rover participating in the [Swiss Rover Challenge](https://swissroverchallenge.com), a national competition in which students design, build, and operate Mars rover prototypes.

<img src="docs/rover.jpeg" width="256"/>

## Installation

On the Raspberry Pi, the `picamera2` package must be installed as a **system package**.

Create a Python virtual environment with access to the system site packages, then install the `microver` package in editable mode.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the following command to start the application:

```bash
microver
```

## License

This project is licensed under the [MIT LICENSE](LICENSE).
