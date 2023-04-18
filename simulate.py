from concurrent.futures import ThreadPoolExecutor, as_completed
from simulate_worker import register_and_watch
import sys

def run_worker(worker_id):
    try:
        register_and_watch(worker_id)
    except Exception as e:
        return f"worker {worker_id} encountered an exception: {e}"
    return f"worker {worker_id} completed successfully"

def main():
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_worker, worker_id) for worker_id in range(10)]

        for future in as_completed(futures):
            result = future.result()
            print(result)
            if "exception" in result:
                print("Terminating program due to exception in a worker thread")
                executor.shutdown(wait=False)
                sys.exit(1)

if __name__ == "__main__":
    main()
