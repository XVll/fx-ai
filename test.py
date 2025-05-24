# test_logger.py - Test script for the logging system
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import initialize_logger, log_info, log_warning, log_error, log_debug, log_critical


def test_logging_system():
    """Test the centralized logging system"""
    print("Testing Centralized Logging System...")
    print("=" * 50)

    # Initialize logger
    log_file = "./test_logs/test_output.log"
    os.makedirs("./test_logs", exist_ok=True)

    logger_manager = initialize_logger(
        app_name="test-fx-ai",
        log_file=log_file,
        max_dashboard_logs=100
    )

    print(f"Log file will be written to: {log_file}")
    print("Testing different log levels...")
    print("")

    # Test different log levels with emoji and without
    log_info("[START] Starting FX-AI Training System", "main")
    log_info("[DIR] Output directory: ./test_output", "main")
    log_info("[FILE] Log file: ./test_output/training.log", "main")
    log_info("[SAVE] Configuration saved", "main")

    log_warning("[WARN] This is a warning message", "test")
    log_error("[ERROR] This is an error message", "test")
    log_debug("[DEBUG] This is a debug message (may not show)", "test")
    log_critical("[CRITICAL] This is a critical message", "test")

    # Test with original emoji (should be converted on Windows)
    log_info("🚀 Starting FX-AI Training System", "emoji_test")
    log_info("📁 Output directory: ./test_output", "emoji_test")
    log_info("📄 Log file: ./test_output/training.log", "emoji_test")
    log_info("💾 Configuration saved", "emoji_test")

    # Test multiple modules
    log_info("[CONFIG] Model configuration loaded", "model")
    log_info("[BUILD] Environment created", "env")
    log_info("[BRAIN] Neural network initialized", "ai")
    log_info("[RUN] Training started", "ppo")

    # Test recent logs
    print("\n" + "=" * 50)
    print("Recent logs from logger:")
    recent_logs = logger_manager.get_recent_logs(10)
    for entry in recent_logs:
        print(f"{entry.timestamp.strftime('%H:%M:%S')} | {entry.level.value:>8} | {entry.module:>10} | {entry.message}")

    print(f"\nTotal log entries: {len(logger_manager.get_recent_logs())}")
    print(f"Log file created: {os.path.exists(log_file)}")

    if os.path.exists(log_file):
        print(f"Log file size: {os.path.getsize(log_file)} bytes")
        print("\nLast few lines of log file:")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print("  " + line.strip())

    print("\n" + "=" * 50)
    print("Logging test completed successfully!")
    print("Press Ctrl+C to test interrupt handling...")

    # Test interrupt handling
    try:
        for i in range(10):
            log_info(f"[TEST] Iteration {i + 1}/10 - Press Ctrl+C to interrupt", "test")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupt caught successfully!")
        log_warning("[INTERRUPT] Test interrupted by user", "test")

    print("Test finished!")


if __name__ == "__main__":
    test_logging_system()