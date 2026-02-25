import os
import sys
import time
import psutil
import subprocess
import webbrowser
import threading

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("==================================================")
    print("      Document Authenticator Server Manager       ")
    print("                By Abdullah Saad                  ")
    print("==================================================")
    
    # Auto-update from GitHub
    print("\nStatus: Checking for updates from GitHub...")
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True
        )
        if "Already up to date." in result.stdout:
            print("System is already up to date.")
        else:
            print("Updates successfully applied!")
    except FileNotFoundError:
        print("Notice: Git is not installed. Skipping auto-update.")
    except subprocess.CalledProcessError:
        print("Notice: Failed to fetch updates (Are you offline?). Starting anyway...")
    except Exception:
        print("Notice: Auto-update skipped. Starting anyway...")
        
    print("\nStatus: Starting Engine...")
    print("Please wait...\n")
    
    # Startup Streamlit as background process
    env = os.environ.copy()
    python_exe = sys.executable
    app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
    
    server_process = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", app_path, "--server.headless", "true"],
        cwd=os.path.join(os.path.dirname(__file__), "app"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(4)
    print("Status: Running on http://localhost:8501")
    print("\nOpening web browser...")
    webbrowser.open("http://localhost:8501")
    
    print("\nPress Ctrl+C to stop the server and exit.")
    
    try:
        while True:
            if server_process.poll() is not None:
                print("Server process died unexpectedly.")
                break
                
            try:
                proc = psutil.Process(server_process.pid)
                mem_bytes = proc.memory_info().rss
                cpu_percent = proc.cpu_percent()
                
                for child in proc.children(recursive=True):
                    mem_bytes += child.memory_info().rss
                    cpu_percent += child.cpu_percent()
                    
                ram_mb = mem_bytes / (1024 * 1024)
                
                # Overwrite the line
                sys.stdout.write(f"\rStatus: CPU {cpu_percent:5.1f}% | RAM {ram_mb:7.1f} MB  (Press Ctrl+C to Quit)")
                sys.stdout.flush()
                
            except Exception:
                pass
                
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        try:
            parent = psutil.Process(server_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
