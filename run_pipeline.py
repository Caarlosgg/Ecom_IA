import os
import subprocess
import time

def run_command(command, step_name):
    print(f"\n🚀 INICIANDO: {step_name}...")
    start = time.time()
    result = subprocess.run(command, shell=True)
    end = time.time()
    
    if result.returncode == 0:
        print(f"✅ {step_name} COMPLETADO en {end - start:.2f} segundos.")
    else:
        print(f"❌ ERROR en {step_name}.")
        exit(1)

if __name__ == "__main__":
    print("="*50)
    print("💎 ECOM-IA: AUTO-DEPLOY PIPELINE")
    print("="*50)

    # 1. Procesamiento de Datos (ETL)
    # Asumiendo que tienes data_processor.py
    # run_command("python src/data_processor.py", "ETL & Limpieza de Datos")

    # 2. Entrenamiento del Modelo (Tu nuevo script PRO)
    run_command("python src/model_trainer.py", "Entrenamiento del Modelo AI")

    # 3. Tests (Opcional pero recomendado)
    # run_command("pytest tests/", "Test de Integridad")

    print("\n🎉 TODO LISTO. Sistema actualizado.")
    print("   -> Para ver el dashboard: streamlit run dashboard.py")
    print("   -> Para lanzar la API:    uvicorn src.api:app --reload")