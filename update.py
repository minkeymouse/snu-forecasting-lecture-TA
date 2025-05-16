#!/usr/bin/env python3
import shutil, subprocess, sys
from pathlib import Path

ENV = "forecasting_lecture"
YAML = Path(__file__).with_name("environment.yml")
DISPLAY = "Python (forecasting)"

def sh(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    if not YAML.exists():
        sys.exit("❌  environment.yml not found")

    solver = "mamba" if shutil.which("mamba") else "conda"
    if solver == "conda":
        print("⚠  mamba not found – installs will be slower")

    # 1 · remove any old env (ignore if it never existed)
    try:
        sh([solver, "env", "remove", "-n", ENV, "-y"])          # conda-doc ref :contentReference[oaicite:2]{index=2}
    except subprocess.CalledProcessError:
        print("ℹ  environment didn’t exist, continuing")

    # 2 · create the env
    sh([solver, "env", "create", "-n", ENV, "-f", str(YAML)])

    # 3 · (optional) clear an old kernelspec, then register a fresh one
    try:
        sh(["jupyter", "kernelspec", "uninstall", "-f", ENV])
    except subprocess.CalledProcessError:
        pass                                                    # nothing to remove
    sh([
        "conda", "run", "-n", ENV,
        "python", "-m", "ipykernel", "install",                 # ipykernel install syntax :contentReference[oaicite:3]{index=3}
        "--user",
        "--name", ENV,
        "--display-name", DISPLAY
    ])

    print(f"\n✅  Done!  Select kernel “{DISPLAY}” in Jupyter.")

if __name__ == "__main__":
    main()
