import pickle
import numpy as np

def load_mnist_data(file_name, offset=0, batch_max=10):
    """
        Takes a text file as input, with offset (in examples) and max batch size. 
        Returns: 2d numpy array of shape (batch_max, 784)
    """
    objects = [] # examples
    try:
        with (open(file_name, "r")) as openfile:
            openfile.seek(offset * 840)
            batch = 0
            end = False
            while True:
                try:
                    obj = np.array([])
                    for _ in range(28):
                        raw_line = openfile.readline()
                        processed_line = [(x=='#')-(x==" ")+0.0 for x in raw_line if x!='\n']
                        obj = np.concatenate((obj, processed_line))
                    objects.append(obj)
                    batch += 1 # count items in batch
                    if batch == batch_max:
                        break
                except EOFError:
                    end=True
                    break
            ptr = openfile.tell()/840 # record position in file
        return {'data': np.vstack(objects)[:,::1], 'offset':ptr, 'end':end}
    except IOError:
        raise IOError("Unable to open file %s." % file_name)

def load_mnist_labels(file_name, offset=0, batch_max=10):
    """
        Takes a text file as input, with offset (in examples) and max batch size.
        Returns: 2d numpy array of shape (batch_max, 1)
    """
    objects = []
    try:
        with (open(file_name, "r")) as openfile:
            openfile.seek(offset * 3)
            end = False
            try:
                for _ in range(batch_max):
                    objects.append(np.array(int(openfile.readline())))
                ptr = openfile.tell()/3
            except EOFError:
                end = True
        return {'data': np.vstack(objects)[:,::1], 'offset':ptr, 'end':end}
    except IOError:
        raise IOError("Unable to open file %s." % file_name)
