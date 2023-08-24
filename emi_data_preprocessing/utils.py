import os, sys

def add_to_path(file_path):
    # Get the current directory of this script
    current_dir = os.path.dirname(os.path.abspath(file_path))

    # Get the parent directory of the current directory and add it to the Python path
    parent_dir = os.path.dirname(current_dir)
    print ("Adding directory to path: {}".format(parent_dir))
    sys.path.append(parent_dir)