from icet.core.lattice_site import LatticeSite
import time


def generate_list_of_lattice_site(amount):
    """
    Generates a list of lattice sites.

    amount : int
        size of list of lattice sites
    """
    sites = []
    indices = range(amount)
    offsets = [[x, y, z]
               for x, y, z in zip(range(amount), range(amount), range(amount))]
    for index, offset in zip(indices, offsets):
        sites.append(LatticeSite(index, offset))
    return sites


def time_sorting(sites, iterations=1000):
    """ Times sorting by alternating between reverse = True and False. """
    t = time.process_time()
    for i in range(iterations):
        sites.sort()
        sites.sort(reverse=True)
    total_time = time.process_time() - t
    return total_time / (2 * iterations)


def time_hash_function(sites, iterations=1000):
    """ Times the hash function. """
    t = time.process_time()
    for i in range(iterations):
        for lattice_site in sites:
            hash = lattice_site.__hash__()  # noqa: F841
    total_time = time.process_time() - t
    return total_time / (iterations)


def time_index_lookup(sites, iterations=1000):
    """ Times looping through sites. """
    t = time.process_time()
    for i in range(iterations):
        for lattice_site in sites:
            index = lattice_site.index  # noqa: F841
    total_time = time.process_time() - t
    return total_time / (iterations)


def time_offset_lookup(sites, iterations=1000):
    """ Times looping through sites. """
    t = time.process_time()
    for i in range(iterations):
        for lattice_site in sites:
            offset = lattice_site.unitcell_offset  # noqa: F841
    total_time = time.process_time() - t
    return total_time / (iterations)


if __name__ == '__main__':
    amount = 100
    sites = generate_list_of_lattice_site(amount)

    print('Timing for sorting: {:.6f} musec'.format(
        1e6 * time_sorting(sites)))
    print('Timing for hash: {:.6f} musec'.format(
        1e6 * time_hash_function(sites)))
    print('Timing for index lookup: {:.6f} musec'.format(
        1e6 * time_index_lookup(sites)))
    print('Timing for offset lookup: {:.6f} musec'.format(
        1e6 * time_offset_lookup(sites)))
