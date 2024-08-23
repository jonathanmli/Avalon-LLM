import argparse
import re
from utils import filter_log_entries

def main():
    parser = argparse.ArgumentParser(description="Filter and print log entries based on logger name.")
    parser.add_argument("log_file_path", help="Path to the log file.")
    parser.add_argument("logger_name", help="Logger name to filter by.")
    
    args = parser.parse_args()

    filter_log_entries(args.log_file_path, args.logger_name)

if __name__ == "__main__":
    main()
