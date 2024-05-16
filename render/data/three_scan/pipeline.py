import sys
import os

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib
import tempfile
import re
from render.data.three_scan.align import align_mesh

BASE_URL = 'http://campar.in.tum.de/files/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/files/3RScan/3RScanTOU.pdf'
TEST_FILETYPES = ['mesh.refined.v2.obj']
# We only provide semantic annotations for the train and validation scans as well as the for the
# reference scans in the test set.
FILETYPES = TEST_FILETYPES + ['labels.instances.annotated.v2.ply']

RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'

RELEASE_SIZE = '~94GB'
id_reg = re.compile(
    r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")


def get_scans(scan_file):
    scan_lines = urllib.urlopen(scan_file)
    scans = []
    for scan_line in scan_lines:
        scan_line = scan_line.decode('utf8').rstrip('\n')
        match = id_reg.search(scan_line)
        if match:
            scan_id = match.group()
            scans.append(scan_id)
    return scans


def download_file(url, out_file):
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_scan(scan_id):
    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES
    out_dir = os.path.join("render", "data", "three_scan", "scans", scan_id)
    os.makedirs(out_dir, exist_ok=True)

    if scan_id in release_scans:
        for ft in file_types:
            url = DATA_URL + '/' + scan_id + '/' + ft
            out_file = out_dir + '/' + ft
            download_file(url, out_file)
    elif scan_id in test_scans:
        for ft in file_types_test:
            url = DATA_URL + '/' + scan_id + '/' + ft
            out_file = out_dir + '/' + ft
            download_file(url, out_file)
    else:
        raise KeyError('ERROR: Invalid scan id: ' + scan_id)

    print('Downloaded scan ' + scan_id)


def download_scans():
    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES

    for scan in release_scans:
        for ft in file_types:
            out_dir = os.path.join(
                "render", "data", "three_scan", "scans", scan)
            os.makedirs(out_dir, exist_ok=True)
            url = DATA_URL + '/' + scan + '/' + ft
            out_file = out_dir + '/' + ft
            download_file(url, out_file)
        align_mesh(scan)
        print('Downloaded scan ' + scan)
    for scan in test_scans:
        for ft in file_types_test:
            out_dir = os.path.join(
                "render", "data", "three_scan", "scans", scan)
            os.makedirs(out_dir, exist_ok=True)
            url = DATA_URL + '/' + scan + '/' + ft
            out_file = out_dir + '/' + ft
            download_file(url, out_file)
        align_mesh(scan)
        print('Downloaded scan ' + scan)


if __name__ == "__main__":
    download_scans()
