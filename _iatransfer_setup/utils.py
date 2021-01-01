import distutils.dir_util
import glob
import os
import sys
import tarfile
from pathlib import Path
from typing import List

from setuptools import setup, find_packages

from iatransfer.utils.dot_dict import DotDict
from iatransfer.utils.file_utils import read_json, read_contents

PROPERTIES = 'setup_configuration.json'
DISTRIBUTIONS = 'DISTRIBUTIONS'


def forbid_publish() -> None:
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            values = {'command': command}
            print('Command "%(command)s" has been blacklisted, exiting...' %
                  values)
            sys.exit(2)


def mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def cleanup(path: str) -> None:
    try:
        distutils.dir_util.remove_tree(path)
    except FileNotFoundError:
        pass


def requirements(filename: str) -> List[str]:
    return [i.strip() for i in open(filename).readlines()]


def do_setup(configuration: DotDict) -> None:
    setup(
        name=configuration.NAME,
        version=configuration.VERSION,
        author=configuration.AUTHOR,
        author_email=configuration.EMAIL,
        url=configuration.URL,
        description=configuration.DESCRIPTION,
        long_description=read_contents(configuration.README_PATH),
        long_description_content_type=configuration.LONG_DESCRIPTION_CONTENT_TYPE,
        license=configuration.LICENSE,
        packages=find_packages(include=configuration.PACKAGES),
        entry_points=configuration.ENTRYPOINTS,
        scripts=[x for string in configuration.SCRIPTS for x in glob.glob(string)],
        classifiers=configuration.CLASSIFIERS,
        keywords=configuration.KEYWORDS,
        install_requires=requirements(configuration.REQUIREMENTS_PATH),
        # data_files=[('.', [PROPERTIES, configuration.REQUIREMENTS_PATH])],
        zip_safe=True
    )


def read_configuration(name: str) -> DotDict:
    properties = read_json(PROPERTIES)
    properties = {**properties, **properties[DISTRIBUTIONS][name]}
    del properties[DISTRIBUTIONS]
    return DotDict(properties)


def setup_configuration(name: str) -> None:
    # TODO remove once we can publish
    forbid_publish()

    configuration = read_configuration(name)
    cleanup(configuration.BUILD_DIR_PATH)

    do_setup(configuration)


def replace_setuppy(configuration: DotDict) -> None:
    mkdir(configuration.DIST_DIR_PATH)

    name = glob.glob(f'{configuration.DIST_DIR_PATH}/iatransfer_research-*.tar.gz')[0]
    dest_name = f'{configuration.DIST_DIR_PATH}/iatransfer_research-tmp.tar.gz'

    tar_in = tarfile.open(name, 'r:gz')
    tar_out = tarfile.open(dest_name, 'w:gz')

    for item in tar_in.getmembers():
        name_split = item.name.split('/')
        if len(name_split) > 1:
            if name_split[1] != 'setup.py':
                tar_out.addfile(item, tar_in.extractfile(item))
            if name_split[1] == 'setup_research.py':
                item.name = f'{name_split[0]}/setup.py'
                tar_out.addfile(item, tar_in.extractfile(item))

    tar_in.close()
    tar_out.close()

    os.rename(dest_name, name)
