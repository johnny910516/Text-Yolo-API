import os
import sys
import subprocess
import time
import platform

def is_port_in_use(port):
    """檢查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_service(service_name, directory, script_name, port):
    """啟動服務"""
    print(f"Starting {service_name}...")
    
    # 檢查端口是否已被占用
    if is_port_in_use(port):
        print(f"Port {port} is already in use. {service_name} may already be running.")
        return True
    
    # 構建命令
    cmd = [sys.executable, script_name]
    
    # 在新的進程中啟動服務
    try:
        if platform.system() == 'Windows':
            # 在Windows上使用CREATE_NEW_CONSOLE標誌啟動新的控制台窗口
            from subprocess import CREATE_NEW_CONSOLE
            process = subprocess.Popen(cmd, cwd=directory, creationflags=CREATE_NEW_CONSOLE)
        else:
            # 在其他平台上使用nohup啟動後台進程
            cmd = ['nohup'] + cmd + ['&']
            process = subprocess.Popen(' '.join(cmd), cwd=directory, shell=True)
        
        print(f"{service_name} started with PID: {process.pid}")
        return True
    except Exception as e:
        print(f"Failed to start {service_name}: {str(e)}")
        return False

def main():
    """主函數"""
    # 獲取當前目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 啟動Split API
    split_api_dir = os.path.join(current_dir, 'Split_API')
    split_api_success = start_service("Split API", split_api_dir, "app.py", 5003)
    
    # 等待Split API啟動
    if split_api_success:
        print("Waiting for Split API to start...")
        time.sleep(5)
    
    # 啟動HTR API
    htr_api_dir = os.path.join(current_dir, 'Recognize_API')
    htr_api_success = start_service("Recognize API", htr_api_dir, "app.py", 5004)
    
    # 等待HTR API啟動
    if htr_api_success:
        print("Waiting for HTR API to start...")
        time.sleep(5)
        
    # 顯示服務狀態
    if split_api_success and htr_api_success:
        print("\nAll services started successfully!")
        print("Split API running at: http://localhost:5003")
        print("Recognize API running at: http://localhost:5004")
    else:
        print("\nSome services failed to start. Please check the error messages above.")

if __name__ == "__main__":
    main()
