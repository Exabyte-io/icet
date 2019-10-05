import pickle


def _write_pickle(fname, data):
    """ Write data to pickle file with filename fname """
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(fname):
    """ Read pickle file and return content """
    with open(fname, 'rb') as handle:
        return pickle.load(handle)
