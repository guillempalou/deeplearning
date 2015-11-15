import sys
import os
import pytest
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
current_dir = os.path.split(os.path.abspath(__file__))[0]
source_dir = os.path.join(os.path.split(current_dir)[0], "src")

print(source_dir)

sys.path.append(source_dir)

# run the tests
pytest.main(["-s", "-x", current_dir])