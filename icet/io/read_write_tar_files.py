"""
Helper functions for reading and writing objects to tar files
"""
import pickle
import tempfile
import ase.io as aseIO


def add_ase_atoms_to_tarfile(tar_file, atoms, arcname, format='json'):
    """ Adds an ase.Atoms object to tar_file """
    temp_file = tempfile.NamedTemporaryFile()
    aseIO.write(temp_file.name, atoms, format=format)
    temp_file.seek(0)
    tar_info = tar_file.gettarinfo(arcname=arcname, fileobj=temp_file)
    tar_file.addfile(tar_info, temp_file)


def add_items_to_tarfile(tar_file, items, arcname):
    """ Add items dict to tar_file by pickle """
    temp_file = tempfile.TemporaryFile()
    pickle.dump(items, temp_file)
    temp_file.seek(0)
    tar_info = tar_file.gettarinfo(arcname=arcname, fileobj=temp_file)
    tar_file.addfile(tar_info, temp_file)
    temp_file.close()


def read_ase_atoms_from_tarfile(tar_file, arcname, format='json'):
    """ Reads ase.Atoms from tar file """
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.write(tar_file.extractfile(arcname).read())
    temp_file.seek(0)
    atoms = aseIO.read(temp_file.name, format=format)
    return atoms


def read_items_from_tarfile(tar_file, arcname):
    """ Reads items stored as pickle from tar_file """
    items = pickle.load(tar_file.extractfile(arcname))
    return items
