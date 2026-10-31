import os
import subprocess
import sys

def run_headless_cooja():
    cooja_dir = "/home/user/contiki-ng/tools/cooja"
    csc_path = os.path.abspath("define_sim.csc")
    
    gradle_cmd = [
        "./gradlew", "run",
        f"--args=--no-gui {csc_path}"
    ]
    
    print(f"Starting headless Cooja simulation with: {' '.join(gradle_cmd)}")
    result = subprocess.run(gradle_cmd, cwd=cooja_dir, capture_output=True, text=True)
    
    print("Cooja Execution Output:")
    print(result.stdout)
    if result.stderr:
        print("Cooja Errors:", result.stderr)

if __name__ == "__main__":
    run_headless_cooja()