import sys

from _iatransfer_setup.utils import setup_configuration, cleanup, read_configuration, replace_setuppy

if __name__ == '__main__':
    config = read_configuration('RESEARCH')
    cleanup(config.DIST_DIR_PATH)
    setup_configuration('RESEARCH')

    if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
        replace_setuppy(config)
