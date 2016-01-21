import os

file_path = os.path.split(__file__)[0]

# root of the project
project_dir = os.path.abspath(os.path.join(file_path, *([os.pardir]*3)))

# resources
resources_dir = os.path.join(project_dir, "resources")

# data
data_dir = os.path.join(resources_dir, "data")

# data directories
test_data_dir = os.path.join(data_dir, "test")

