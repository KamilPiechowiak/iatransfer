import distutils.dir_util
import glob
import sys
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
        entry_points=configuration.ENTRY_POINTS,
        scripts=[x for string in configuration.SCRIPTS for x in glob.glob(string)],
        classifiers=configuration.CLASSIFIERS,
        keywords=configuration.KEYWORDS,
        install_requires=requirements(configuration.REQUIREMENTS_PATH),
        zip_safe=True
    )


def read_configuration(name: str) -> DotDict:
    properties = read_json(PROPERTIES)
    properties = {**properties[DISTRIBUTIONS][name], **properties}
    del properties[DISTRIBUTIONS]
    return DotDict(properties)


def setup_configuration(name: str) -> None:
    # TODO remove once we can publish
    forbid_publish()

    configuration = read_configuration(name)
    cleanup(configuration.BUILD_DIR_PATH)

    do_setup(configuration)


if __name__ == '__main__':
    setup_configuration('TOOLKIT')
