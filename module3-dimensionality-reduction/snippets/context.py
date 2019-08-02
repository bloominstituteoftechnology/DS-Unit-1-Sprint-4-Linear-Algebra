# Generate context manager class & instance
import os

class ChDir(object):
    """
    Step into a directory temporarily.
        Arguments - path as str.  May be relative or absolute path
        Example Usage: 
            with ChDir(path+'/'):
                products = pd.read_csv('products.csv', usecols=['product_id', 'product_name'])
    """
    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)