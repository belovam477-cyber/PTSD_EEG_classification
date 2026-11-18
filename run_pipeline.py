# run_pipeline.py
import subprocess
import sys

def run_step(script_path):
    print(f"\n{'='*50}\nRunning: {script_path}\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {script_path} failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)

if __name__ == "__main__":
    print("Starting EEG ML Pipeline...")
    
    # Шаг 1: Предобработка
    run_step("src/preprocess.py")
    
    # Шаг 2: Извлечение признаков
    run_step("src/features.py")
    
    # Шаг 3: Оценка моделей
    run_step("src/evaluate.py")
    
    print("\nPipeline finished successfully!")